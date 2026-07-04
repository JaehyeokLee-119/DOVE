import pandas as pd
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
import fire 
import os 
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from collections import defaultdict


def extension_condition(codebook_df, codebook_dir, overuse_z_score_threshold=2.0, window_size=2, decrease_ratio_threshold=0.01):
    # Load codebooks so far
    codebook_files = []
    for root, dirs, files in os.walk(codebook_dir):
        for file in files:
            if 'code_book.parquet' in file:
                codebook_files.append(os.path.join(root, file))
    df_codebook = pd.DataFrame()
    codebook_files = [f for f in codebook_files if os.path.basename(os.path.dirname(f)).startswith('t') and os.path.basename(os.path.dirname(f))[1:].isdigit()]
    
    for codebook_file in codebook_files:
        t_value = os.path.basename(os.path.dirname(codebook_file)) 
        temp_df = pd.read_parquet(codebook_file)
        
        t = int(t_value[1:])
        temp_df['t'] = t
        if 'code_id' not in temp_df.columns:
            temp_df['code_id'] = temp_df.index
        df_codebook = pd.concat([df_codebook, temp_df], ignore_index=True)
    
    df_codebook.sort_values(by='t', inplace=True)
    
    current_t = df_codebook['t'].max()
    
    df_latest_codebook = df_codebook[df_codebook['t'] == current_t].reset_index(drop=True)
    df_latest_codebook['z_score_n_k'] = (df_latest_codebook['n_k'] - df_latest_codebook['n_k'].mean()) / df_latest_codebook['n_k'].std()
    
    df_latest_codebook['is_overused'] = df_latest_codebook['z_score_n_k'] > overuse_z_score_threshold
    df_latest_codebook[f'D_k_at_{current_t}'] = df_latest_codebook['D_k']

    if current_t > 1 + window_size:
        for t in range(current_t, 1, -1):
            df_latest_codebook[f'D_k_at_{t-1}'] = None
            for idx, row in df_latest_codebook.iterrows():
                prev_code_id = row['previous_code_id']
                if pd.notna(prev_code_id):
                    matched_row = df_codebook[(df_codebook['t'] == t-1) & (df_codebook['code_id'] == prev_code_id)]
                    if not matched_row.empty:
                        df_latest_codebook.at[idx, f'D_k_at_{t-1}'] = matched_row.iloc[0]['D_k']

    df_latest_codebook['is_not_significantly_decreased'] = False

    def check_extension_target(row, current_t, window_size=2):
        decline_count = 0
        for t in range(current_t, current_t - window_size, -1):
            D_k_current = row.get(f'D_k_at_{t}')
            D_k_prev = row.get(f'D_k_at_{t-1}')
            if pd.notna(D_k_current) and pd.notna(D_k_prev):
                # improvement = D_k_prev - D_k_current  # 
                delta = D_k_current - D_k_prev  #  
                if delta / abs(D_k_prev) > -decrease_ratio_threshold:
                    decline_count += 1
        return decline_count > 0
        # return decline_count == window_size

    df_latest_codebook['is_not_significantly_decreased'] = df_latest_codebook.apply(lambda row: check_extension_target(row, current_t, window_size=window_size), axis=1)
    df_latest_codebook['is_extension_target'] = (df_latest_codebook['is_overused']) & (df_latest_codebook['is_not_significantly_decreased'])
    current_code_ids = codebook_df['code_id'].unique().tolist()
    
    overused_code_ids = list(set(current_code_ids) & set(df_latest_codebook[df_latest_codebook['is_extension_target']]['code_id'].tolist()))
    
    return overused_code_ids


def main(
    codebook_fpath = '',
    ve_embedding_fpath = '',
    result_fpath = '',
    codebook_dir = '',
    overuse_z_score_threshold = 2.0,
    low_utilization_z_score_threshold = -1.0,
    overuse_window_size = 2,
    decrease_ratio_threshold = 0.00,
    merge = True,
    extend = True,
):
    codebook_df = pd.read_parquet(codebook_fpath)
    ve_embedding_df = pd.read_parquet(ve_embedding_fpath)

    if type(merge) == str:
        merge = merge.lower() == 'true'
    if type(extend) == str:
        extend = extend.lower() == 'true'
        
    codebook_df['previous_code_id'] = codebook_df['code_id']
    codebook_df['updated'] = False
    codebook_df['absorbed_codes'] = [[] for _ in range(len(codebook_df))]
    
    if not merge and not extend:
        print("Both merge and extend are set to False. No operations will be performed.")
        print("End of iteration")
        current_codebook_iteration_finished_fname = os.path.join(codebook_dir, 'codebook_iteration_finished.txt')
        with open(current_codebook_iteration_finished_fname, 'w') as f:
            f.write("Codebook iteration finished without merge or extend operations.\n")
        return
    
    codebook_df['z_score_n_k'] = (codebook_df['n_k'] - codebook_df['n_k'].mean()) / codebook_df['n_k'].std()
    codebook_df['z_score_D_k'] = (codebook_df['D_k'] - codebook_df['D_k'].mean()) / codebook_df['D_k'].std()
    
    dead_mask_1 = codebook_df['z_score_n_k'] < low_utilization_z_score_threshold
    dead_mask_2 = codebook_df['n_k'] <= 0.0 
    
    dead_mask = (dead_mask_1 == True) & (dead_mask_2 == False) 
    alive_mask = ~dead_mask
    
    alive_codes = codebook_df[alive_mask].copy()
    dead_codes = codebook_df[dead_mask].copy()
    alive_codes['len'] = alive_codes.ve_list.apply(len)

    overused_code_ids = extension_condition(codebook_df, codebook_dir, overuse_z_score_threshold, window_size=overuse_window_size, decrease_ratio_threshold=decrease_ratio_threshold)
    
    if len(dead_codes) == 0 and len(overused_code_ids) == 0:
        print("No dead codes to merge and no overused codes to extend. Exiting.")
        print("End of iteration")
        current_codebook_iteration_finished_fname = os.path.join(codebook_dir, 'codebook_iteration_finished.txt')
        with open(current_codebook_iteration_finished_fname, 'w') as f:
            f.write("Codebook iteration finished without merge or extend operations.\n")
        return
    
    if merge: 
        if not dead_codes.empty:
            
            alive_embeddings = torch.tensor(np.stack(alive_codes['centroid_embedding'].values))
            dead_embeddings = torch.tensor(np.stack(dead_codes['centroid_embedding'].values))
            similarities = F.cosine_similarity(dead_embeddings.unsqueeze(1), alive_embeddings.unsqueeze(0), dim=2)
            most_similar_indices = torch.argmax(similarities, dim=1).tolist()

            alive_row_indices = alive_codes.index.tolist()

            for i, dead_row_idx in enumerate(dead_codes.index):
                target_alive_row_idx = alive_row_indices[most_similar_indices[i]]
                
                dead_name = dead_codes.at[dead_row_idx, 'code_name']
                codebook_df.at[target_alive_row_idx, 'absorbed_codes'] = \
                    codebook_df.at[target_alive_row_idx, 'absorbed_codes'] + [dead_name]
                
                codebook_df.at[target_alive_row_idx, 'updated'] = True

            codebook_df.drop(dead_codes.index, inplace=True)

        print(f"Remaining codes: {len(codebook_df)}")
        print(f"Updated (Merged alive) codes count: {codebook_df['updated'].sum()}")
    
    else:
        print("Merging step skipped.")
        
    if extend:
        print(f"Overused codes to extend: {len(overused_code_ids)}")

        df_extension_target = codebook_df[codebook_df['code_id'].isin(overused_code_ids)].reset_index(drop=True)
        df_extended = pd.DataFrame()

        if not df_extension_target.empty:
            new_code_id = codebook_df['code_id'].max() + 1
            for i in tqdm(range(len(overused_code_ids)), desc="Extending overused codes"):
                if 'code_id' not in ve_embedding_df.columns:
                    ve_embedding_df['code_id'] = ve_embedding_df.index
                
                current_divide_target_ve_ids = df_extension_target[df_extension_target['code_id'] == overused_code_ids[i]].ve_list.values[0]
                current_divide_target_embedding_df = ve_embedding_df[ve_embedding_df['code_id'].isin(current_divide_target_ve_ids)].reset_index(drop=True)

                embeddings = np.vstack(current_divide_target_embedding_df['embedding'].values)
                normalized_embeddings = normalize(embeddings) # embedding normalize
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(normalized_embeddings)

                current_divide_target_embedding_df['cluster'] = cluster_labels

                for cluster_label in current_divide_target_embedding_df['cluster'].unique():
                    ve_ids_in_cluster = current_divide_target_embedding_df[current_divide_target_embedding_df['cluster'] == cluster_label]['code_id'].tolist()
                    
                    new_code_entry = {
                        'code_id': new_code_id,
                        'previous_code_id': overused_code_ids[i],
                        've_list': ve_ids_in_cluster,
                        'n_k': 0,
                        'D_k': 0.0,
                        'cluster_size': len(ve_ids_in_cluster),
                    }
                    new_code_id += 1
                    df_extended = pd.concat([df_extended, pd.DataFrame([new_code_entry])], ignore_index=True)
                
            codebook_df = pd.concat([codebook_df, df_extended], ignore_index=True)
            codebook_df = codebook_df[~codebook_df['code_id'].isin(df_extended['previous_code_id'].values)]

    else:
        print("Extension step skipped.")

    rows_list = []
    for idx, row in tqdm(codebook_df.iterrows(), total=len(codebook_df), desc="Recalculating"):
        ve_list = row['ve_list']
        
        if ve_list is None or (isinstance(ve_list, (list, np.ndarray)) and len(ve_list) == 0):
            continue

        matched_embeddings = ve_embedding_df[ve_embedding_df.index.isin(ve_list)]['embedding']
        
        if matched_embeddings.empty:
            continue

        ve_embeddings = matched_embeddings.tolist()
        ve_tensor = torch.tensor(np.stack(ve_embeddings))

        centroid_embedding = torch.mean(ve_tensor, dim=0).numpy()
        cluster_size = len(ve_list)

        new_row = row.copy()
        new_row['centroid_embedding'] = centroid_embedding

        mu = ve_tensor.mean(dim=0) 
        sigma2 = torch.mean((ve_tensor - mu) ** 2)
        sigma = torch.sqrt(sigma2).item()

        new_row['sigma'] = sigma
        new_row['cluster_size'] = cluster_size
        
        rows_list.append(new_row)

    if rows_list:
        new_codebook_df = pd.DataFrame(rows_list)
    else:
        print("Empty codebook after processing. No valid codes remain.")
        new_codebook_df = pd.DataFrame()

    new_codebook_df.to_parquet(result_fpath, index=False)
    print(f"Result saved to: {result_fpath}") 

if __name__  == '__main__':
    fire.Fire(main)
