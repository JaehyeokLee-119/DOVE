import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import fire 

def compute_topic_posterior_vectorized(embedding_df, codebook_df, device='cuda' if torch.cuda.is_available() else 'cpu', encoder_version='global_sigma'):
    K = len(codebook_df)
    
    centroids = torch.tensor(
        np.stack(codebook_df['centroid_embedding'].tolist()), 
        dtype=torch.float32
    ).to(device)
    cen_norm = F.normalize(centroids, p=2, dim=-1) # (K, D)

    all_embeddings = torch.tensor(
        np.stack(embedding_df['embedding'].tolist()),
        dtype=torch.float32
    ).to(device)
    
    doc_indices_original = embedding_df['source_doc_idx'].values
    unique_docs, doc_ids_mapped = np.unique(doc_indices_original, return_inverse=True)
    
    doc_ids_tensor = torch.tensor(doc_ids_mapped, device=device)
    num_docs = len(unique_docs)

    print("Computing Similarities ...")
    
    emb_norm = F.normalize(all_embeddings, p=2, dim=-1)
    sim = torch.matmul(emb_norm, cen_norm.T) 
    sigma = torch.tensor(codebook_df['sigma'].values, dtype=torch.float32).to(device)
    sigma_sq = (sigma ** 2) + 1e-9
    sigma_global = torch.sqrt(torch.mean(sigma ** 2)) + 1e-9
    
    if encoder_version == 'global_sigma':
        logits = (sim / (sigma_global))
    elif encoder_version == 'individual_sigma':
        logits = (sim / sigma_sq.view(1, -1))
    
    probs_all = F.softmax(logits, dim=-1)

    print("Aggregating by document (Aggregation)...")
    
    final_doc_probs = torch.zeros((num_docs, K), device=device, dtype=torch.float32)
    final_doc_probs.index_add_(0, doc_ids_tensor, probs_all)
    doc_counts = torch.bincount(doc_ids_tensor, minlength=num_docs).float().unsqueeze(1)
    
    final_doc_probs = final_doc_probs / (doc_counts + 1e-9)
    result_np = final_doc_probs.cpu().numpy()
    
    result_df = pd.DataFrame({
        'source_doc_idx': unique_docs,  # same document-row-position key as embedding_df['source_doc_idx']
        'q_z_x': list(result_np)
    })
    
    return result_df

def code_name_mapper(q_z_x, code_names, M=-1, validity_threshold=1e-9):
    if not isinstance(q_z_x, np.ndarray):
        return [], [], []
    
    probs = np.array(q_z_x)
    names = np.array(code_names)
    
    original_indices = np.arange(len(probs))
    
    valid_mask = probs > validity_threshold
    
    valid_probs = probs[valid_mask]
    valid_names = names[valid_mask]
    valid_orig_indices = original_indices[valid_mask] 
    
    sorted_sort_indices = np.argsort(valid_probs)[::-1]
    
    final_probs = valid_probs[sorted_sort_indices]
    final_names = valid_names[sorted_sort_indices]
    final_orig_indices = valid_orig_indices[sorted_sort_indices]
    
    if M != -1 and M < len(final_names):
        final_probs = final_probs[:M]
        final_names = final_names[:M]
        final_orig_indices = final_orig_indices[:M]
        
    return final_orig_indices.tolist(), final_names.tolist(), final_probs.tolist()

def main(
    ve_embedding_fpath = f'',
    codebook_fpath = f'',
    doc_fpath = '',
    result_fpath = '',
    encoder_version='global_sigma',
    validity_threshold=1e-2,
    M=15,
):
    ve_embedding_df = pd.read_parquet(ve_embedding_fpath)
    codebook_df = pd.read_parquet(codebook_fpath)
    document_df = pd.read_parquet(doc_fpath)

    document_df['source_doc_idx'] = document_df.index

    q_z_x_result_df = compute_topic_posterior_vectorized(ve_embedding_df, codebook_df, encoder_version=encoder_version)

    merged_df = pd.merge(document_df, q_z_x_result_df, on='source_doc_idx', how='left')
    code_name_list = codebook_df['code_name'].tolist()
    
    merged_df['s_indices'], merged_df['s'], merged_df['z'] = zip(*merged_df['q_z_x'].apply(lambda q: code_name_mapper(q, code_name_list, validity_threshold=validity_threshold, M=M)))
    
    if merged_df['q_z_x'].isnull().any():
        codebook_length = len(codebook_df)
        
        existing_valid_q = merged_df['q_z_x'].dropna().iloc[0] if not merged_df['q_z_x'].dropna().empty else None
        target_dtype = existing_valid_q.dtype if existing_valid_q is not None else np.float64
        
        merged_df['q_z_x'] = merged_df['q_z_x'].apply(
            lambda q: np.zeros(codebook_length, dtype=target_dtype) 
                    if not isinstance(q, np.ndarray) 
                    else q.astype(target_dtype)
        )
    if 'corresponding_codes' in merged_df.columns:
        merged_df = merged_df.rename(columns={'corresponding_codes': 've_list'})

    merged_df.to_parquet(result_fpath, index=False)

if __name__ == "__main__":
    fire.Fire(main)