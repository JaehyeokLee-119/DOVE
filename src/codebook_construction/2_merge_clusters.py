import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import umap
import sklearn.cluster
import fire
from tqdm import tqdm

def cos_sim_matrix(a, b):
    """
    a: (N, D) tensor, b: (M, D) tensor
    Returns: (N, M) similarity matrix
    """
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def main(
    ve_embedding_fpath='',
    result_fpath='',
    tau_1 = 0.4,
    allow_new_cluster = False,
    min_cluster_size=5,
):
    allow_new_cluster = True if allow_new_cluster in [True, 'True', 'true', 1] else False

    # Load value expression embeddings
    df = pd.read_parquet(ve_embedding_fpath)
    df['index'] = df.index
    embeddings_np = np.stack(df['embedding'].values).astype('float32')
    print(f"Loaded {len(df)} embeddings.")

    if type(min_cluster_size) == str:
        if '%' in min_cluster_size:
            min_cluster_size = int(len(df) * float(min_cluster_size.replace('%','')) // 100)
        else:
            min_cluster_size = int(min_cluster_size)

        print(f"Interpreted min_cluster_size as {min_cluster_size}.")

    reducer = umap.UMAP(n_components=5, metric='cosine')
    embedding_5d = reducer.fit_transform(embeddings_np)

    clusterer = sklearn.cluster.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', n_jobs=-1)
    df['code_id'] = clusterer.fit_predict(embedding_5d)
    
    final_df = df[df['code_id'] != -1].copy()
    noise_df = df[df['code_id'] == -1].copy()

    # Summarize information by cluster
    cluster_df = final_df.groupby('code_id').agg({
        'embedding': lambda x: np.mean(np.stack(x.values), axis=0),
        'index': list
    }).reset_index().rename(columns={'embedding': 'centroid_embedding', 'index': 've_list'})
    cluster_df['cluster_size'] = cluster_df['ve_list'].apply(len)
    
    
    if not noise_df.empty:
        noise_indices = noise_df.index.tolist()
        noise_embs = torch.tensor(np.stack(noise_df['embedding'].values))
        centroid_embs = torch.tensor(np.stack(cluster_df['centroid_embedding'].values))

        sim_matrix = cos_sim_matrix(noise_embs, centroid_embs)
        max_sims, max_indices = torch.max(sim_matrix, dim=1)

        if allow_new_cluster:
            merge_mask = max_sims > tau_1
        else:
            merge_mask = torch.ones_like(max_sims, dtype=torch.bool)

        to_merge_indices = np.array(noise_indices)[merge_mask.numpy()]
        to_merge_targets = max_indices[merge_mask].numpy()

        for idx, target_rel_idx in zip(to_merge_indices, to_merge_targets):
            cluster_df.at[target_rel_idx, 've_list'].append(idx)
            cluster_df.at[target_rel_idx, 'cluster_size'] += 1
            
        to_new_indices = np.array(noise_indices)[~merge_mask.numpy()]
        
        if len(to_new_indices) > 0:
            new_rows = []
            if not cluster_df.empty:
                next_label = cluster_df['code_id'].max() + 1
            else:
                next_label = 0
                
            for idx in to_new_indices:
                new_rows.append({
                    'code_id': next_label,
                    'centroid_embedding': df.loc[idx, 'embedding'],
                    've_list': [idx],
                    'cluster_size': 1
                })
                next_label += 1
            
            if new_rows:
                cluster_df = pd.concat([cluster_df, pd.DataFrame(new_rows)], ignore_index=True)
                
    def refresh_final_centroid(row):
        embs = np.stack(df.loc[row['ve_list'], 'embedding'].values)
        return np.mean(embs, axis=0)

    cluster_df['centroid_embedding'] = cluster_df.apply(refresh_final_centroid, axis=1)
    cluster_df['cluster_size'] = cluster_df['ve_list'].apply(len)

    cluster_df.to_parquet(result_fpath, index=False)
    print(f"Final clusters: {len(cluster_df)}")
    print(f"Saved result to {result_fpath}")

if __name__ == "__main__":
    fire.Fire(main)