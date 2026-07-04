import time
import os
import pandas as pd
import fire
import ot
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from numba import jit

@jit(nopython=True, fastmath=True)
def compute_accumulated_min_max(Q):
    M, K = Q.shape
    min_acc = np.zeros((K, K), dtype=np.float64)
    max_acc = np.zeros((K, K), dtype=np.float64)
    
    for m in range(M):
        q = Q[m]
        for i in range(K):
            val_i = q[i]
            for j in range(K):
                val_j = q[j]
                if val_i < val_j:
                    min_acc[i, j] += val_i
                    max_acc[i, j] += val_j
                else:
                    min_acc[i, j] += val_j
                    max_acc[i, j] += val_i
                    
    return min_acc, max_acc

def uot_cost(a, b, D, epsilon_metric=0.05, tau_metric=1.0):
    pi = ot.unbalanced.sinkhorn_knopp_unbalanced(
        a, b, D, epsilon_metric, tau_metric
    )

    transport_cost = np.sum(pi * D)

    eps = 1e-16
    entropy_term = epsilon_metric * np.sum(pi * (np.log(pi + eps) - 1.0))

    row_marg = pi.sum(axis=1)
    col_marg = pi.sum(axis=0)

    def kl(p, q, eps=1e-16):
        p_ = p + eps
        q_ = q + eps
        return np.sum(p_ * np.log(p_ / q_) - p_ + q_)

    kl_row = kl(row_marg, a)
    kl_col = kl(col_marg, b)
    kl_term = tau_metric * (kl_row + kl_col)

    cost = transport_cost + entropy_term + kl_term
    return cost

def prepare_sem_matrix(codebook_df):
    E = np.stack(codebook_df['centroid_embedding'].values, axis=0)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    cosine_sim = E_norm @ E_norm.T
    D_sem = 1.0 - cosine_sim
    return D_sem

def precompute_reference_data(ref_fpath, D_sem, topic_equivalence=False, valid_q_idxs=None):
    print(f"Pre-computing reference: {ref_fpath} ...")
    reference_df = pd.read_parquet(ref_fpath)
    
    if valid_q_idxs is not None:
        reference_df = reference_df[reference_df['q_idx'].isin(valid_q_idxs)].reset_index(drop=True)
            
    # Extract Q_ref
    if topic_equivalence:
        topic_groups = reference_df.groupby('q_idx')
        Q_ref_list = []
        for _, group in topic_groups:
            Q_topic = np.stack(group['q_z_x'].values, axis=0)  # (num_topics, K)
            Q_topic_mean = Q_topic.mean(axis=0)                 # (K,)
            Q_ref_list.append(Q_topic_mean)
        Q_ref = np.stack(Q_ref_list, axis=0)  # (M, K)
    else:
        Q_ref = np.stack(reference_df['q_z_x'].values, axis=0) # (M, K)
    
    a = Q_ref.mean(axis=0)
    a = a / a.sum()
    
    min_acc, max_acc = compute_accumulated_min_max(Q_ref)
    
    M_ref = Q_ref.shape[0]
    E_min = min_acc / M_ref
    E_max = max_acc / M_ref
    
    eps2 = 1e-8
    w = 1.0 - (E_min / (E_max + eps2))
    np.fill_diagonal(w, 0.0)
    
    D = w * D_sem
    
    return a, D, Q_ref.shape[0]

def calculate_debiased_uot(a, b, D, epsilon, tau):
    cost_p_pth = uot_cost(a, b, D, epsilon, tau)
    cost_p_p   = uot_cost(a, a, D, epsilon, tau)
    cost_pt_pt = uot_cost(b, b, D, epsilon, tau)
    
    return cost_p_pth - 0.5 * cost_p_p - 0.5 * cost_pt_pt

def process_single_model_file(file_path, ref_data_dict, epsilon, tau, topic_equivalence=False, valid_q_idxs=None):
    try:
        eval_target_df = pd.read_parquet(file_path)
        if 'q_idx' not in eval_target_df.columns:
            eval_target_df['q_idx'] = eval_target_df['source_doc_idx']
        
        if valid_q_idxs is not None:
            eval_target_df = eval_target_df[eval_target_df['q_idx'].isin(valid_q_idxs)].reset_index(drop=True)
            
        if topic_equivalence:
            topic_groups = eval_target_df.groupby('q_idx')
            Q_tgt_list = []
            for _, group in topic_groups:
                Q_topic = np.stack(group['q_z_x'].values, axis=0)
                Q_topic_mean = Q_topic.mean(axis=0)
                Q_tgt_list.append(Q_topic_mean)
            Q_tgt = np.stack(Q_tgt_list, axis=0)  # (M, K)
        else:
            Q_tgt = np.stack(eval_target_df['q_z_x'].values, axis=0)

        b = Q_tgt.mean(axis=0)
        b = b / b.sum()
        
        row = {'model_file': file_path}
        
        for lang in ['KR', 'JP', 'CN', 'US']:
            a_ref, D_ref = ref_data_dict[lang]
            score = calculate_debiased_uot(a_ref, b, D_ref, epsilon, tau)
            row[lang] = score
            
        return row
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_batch(files, ref_data_dict, epsilon, tau, topic_equivalence=False, valid_q_idxs=None, n_jobs=4):
    results = []
    
    worker = partial(process_single_model_file, ref_data_dict=ref_data_dict, epsilon=epsilon, tau=tau, topic_equivalence=topic_equivalence, valid_q_idxs=valid_q_idxs)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for res in executor.map(worker, files):
            if res is not None:
                results.append(res)
                
    return pd.DataFrame(results)

def main(
    result_dir = 'outputs/eval_target/encoding_results',
    kr_encoded_f = 'data/GPT-5.2/reference/human_docs_KR_encoded.parquet',
    jp_encoded_f = 'data/GPT-5.2/reference/human_docs_JP_encoded.parquet',
    cn_encoded_f = 'data/GPT-5.2/reference/human_docs_CN_encoded.parquet',
    en_encoded_f = 'data/GPT-5.2/reference/human_docs_US_encoded.parquet',
    codebook_fpath = 'data/GPT-5.2/codebook/finalized_codebook.parquet',
    eval_result_dir = 'outputs/eval_target/eval_output',
    valid_qid_file = 'data/resource/qids.txt',
    epsilon_metric = 0.005,
    tau_metric = 0.5,
    topic_equivalence = False,
    n_jobs = 60,
    reward_postprocess = True,
):
    os.makedirs(eval_result_dir, exist_ok=True)
    times_file = os.path.join(eval_result_dir, 'evaluation_times.log')
    
    if valid_qid_file:
        with open(valid_qid_file, 'r') as f:
            valid_q_idxs = [int(line.strip()) for line in f.readlines()]
        print(f"Loaded {len(valid_q_idxs)} valid q_idx from {valid_qid_file}.")
    else:
        valid_q_idxs = None
        
    starttime = time.time()
    print(f"Loading Codebook...")
    codebook_df = pd.read_parquet(codebook_fpath)
    endtime = time.time()
    print(f"Codebook loaded. Time taken: {endtime - starttime:.2f} seconds. ({time.ctime()})")
    with open(times_file, 'a') as f:
        f.write(f"Codebook loaded. Time taken: {endtime - starttime:.2f} seconds. ({time.ctime()})\n")
    D_sem = prepare_sem_matrix(codebook_df)
    endtime = time.time()
    print(f"Semantic matrix prepared. Time taken: {endtime - starttime:.2f} seconds. ({time.ctime()})")
    with open(times_file, 'a') as f:
        f.write(f"Semantic matrix prepared. Time taken: {endtime - starttime:.2f} seconds. ({time.ctime()})\n")
    print("Pre-computing Reference Data (Cost Matrices)... This runs once.")

    ref_data = {}
    ref_data['KR'] = precompute_reference_data(kr_encoded_f, D_sem, topic_equivalence=topic_equivalence, valid_q_idxs=valid_q_idxs)[0:2]
    ref_data['JP'] = precompute_reference_data(jp_encoded_f, D_sem, topic_equivalence=topic_equivalence, valid_q_idxs=valid_q_idxs)[0:2]
    ref_data['CN'] = precompute_reference_data(cn_encoded_f, D_sem, topic_equivalence=topic_equivalence, valid_q_idxs=valid_q_idxs)[0:2]
    ref_data['US'] = precompute_reference_data(en_encoded_f, D_sem, topic_equivalence=topic_equivalence, valid_q_idxs=valid_q_idxs)[0:2]
    endtime = time.time()
    print(f"Reference data prepared. Time taken: {endtime - starttime:.2f} seconds. ({time.ctime()})")
    with open(times_file, 'a') as f:
        f.write(f"Reference data prepared. Time taken: {endtime - starttime:.2f} seconds. ({time.ctime()})\n")
    
    print("Reference data preparation complete. Starting evaluation...")

    model_list = os.listdir(result_dir)
    model_list = [x for x in model_list if x.endswith('.parquet')]

    files = [f"{result_dir}/{x}" for x in os.listdir(result_dir) if x.endswith('.parquet') and any(m in x for m in model_list)]
    if files:
        print(f"Processing {len(files)} files)...")
        codebook_df = process_batch(files, ref_data, epsilon_metric, tau_metric, topic_equivalence=topic_equivalence, n_jobs=n_jobs)

        if reward_postprocess:
            for lang in ['KR', 'JP', 'CN', 'US']:
                codebook_df[lang] = (0.1 - codebook_df[lang]) * 1000

        codebook_df.to_csv(f"{eval_result_dir}/eval_output.csv", index=False)

    with open(times_file, 'a') as f:
        f.write(f"Finished processing. ({time.ctime()})\n")

    print(f"Result saved to {eval_result_dir}/eval_output.csv")

if __name__ == '__main__':
    fire.Fire(main)