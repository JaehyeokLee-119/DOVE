import pandas as pd
import fire 
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def distance(emb1, emb2):
    return 1 - F.cosine_similarity(
        torch.tensor(emb1).unsqueeze(0),
        torch.tensor(emb2).unsqueeze(0),
        dim=1
    ).item()

def main(
    codebook_fpath = '',
    encoding_result_fpath = '',
    document_embedding_fpath = '',
    reconstruction_embedding_fpath = '',
    N1 = 3,
    N2 = 1,
):
    codebook_df = pd.read_parquet(codebook_fpath)
    encoded_df = pd.read_parquet(encoding_result_fpath)
    df_reconstruction_embedding = pd.read_parquet(reconstruction_embedding_fpath)
    df_doc_embedding = pd.read_parquet(document_embedding_fpath)

    df_doc_embedding['source_doc_idx'] = df_doc_embedding.index

    encoded_df = encoded_df[encoded_df['q_z_x'].apply(lambda x: x is not None)]
    
    valid_source_doc_idx = set(df_reconstruction_embedding['source_doc_idx']).intersection(
        set(encoded_df['source_doc_idx'])
    ).intersection(
        set(df_doc_embedding['source_doc_idx'])
    )
    
    df_reconstruction_embedding = df_reconstruction_embedding[
        df_reconstruction_embedding['source_doc_idx'].isin(valid_source_doc_idx)
    ]
    df_doc_embedding = df_doc_embedding[
        df_doc_embedding['source_doc_idx'].isin(valid_source_doc_idx)
    ]
    encoded_df = encoded_df[encoded_df['source_doc_idx'].isin(valid_source_doc_idx)]

    df_reconstruction_embedding = df_reconstruction_embedding.sort_values(
        by=['source_doc_idx', 'sample_idx', 'gen_idx']
    ).reset_index(drop=True)

    recon_array = np.stack(df_reconstruction_embedding['embedding'].values)
    doc_array = np.stack(df_doc_embedding['embedding'].values)
    
    N = len(doc_array)
    D = doc_array.shape[1]
    
    expected_total = N * N1 * N2
    if len(recon_array) != expected_total:
        doc_counts = df_reconstruction_embedding.groupby('source_doc_idx').size()
        complete_docs = doc_counts[doc_counts == N1 * N2].index
        
        df_reconstruction_embedding = df_reconstruction_embedding[
            df_reconstruction_embedding['source_doc_idx'].isin(complete_docs)
        ]
        df_doc_embedding = df_doc_embedding[
            df_doc_embedding['source_doc_idx'].isin(complete_docs)
        ]
        encoded_df = encoded_df[encoded_df['source_doc_idx'].isin(complete_docs)]
        
        recon_array = np.stack(df_reconstruction_embedding['embedding'].values)
        doc_array = np.stack(df_doc_embedding['embedding'].values)
        N = len(doc_array)
    
    recon_tensor = torch.from_numpy(recon_array).view(N, N1, N2, D)
    doc_tensor = torch.from_numpy(doc_array)

    recon_flat = recon_tensor.view(N, N1*N2, D)
    doc_expanded = doc_tensor.unsqueeze(1).expand(-1, N1*N2, -1)
    
    distortion_all = 1 - F.cosine_similarity(recon_flat, doc_expanded, dim=2)
    distortion_grouped = distortion_all.view(N, N1, N2)
    
    distortion_per_code_set = distortion_grouped.mean(dim=2)
    
    distortion_per_doc = distortion_per_code_set.mean(dim=1).numpy()
    
    encoded_df['distortion'] = distortion_per_doc

    sum_z_x = np.sum(np.stack(encoded_df['q_z_x'].values), axis=0)
    n_k = sum_z_x
    N_total = len(encoded_df)
    
    code_distortions = []
    for idx, row in encoded_df.iterrows():
        doc_distortion = row['distortion']
        
        doc_recons = df_reconstruction_embedding[
            df_reconstruction_embedding['source_doc_idx'] == row['source_doc_idx']
        ]
        
        for sample_idx in range(N1):
            sample_recons = doc_recons[doc_recons['sample_idx'] == sample_idx]
            if len(sample_recons) > 0:
                codes = sample_recons.iloc[0]['s_indices']
                for code_idx in codes:
                    code_distortions.append({
                        'code_idx': code_idx,
                        'distortion': doc_distortion
                    })
    
    code_dist_df = pd.DataFrame(code_distortions)
    if len(code_dist_df) > 0:
        stats_df = code_dist_df.groupby('code_idx')['distortion'].agg(['count', 'mean'])
        stats_df.rename(columns={'count': 'n_k_sampled', 'mean': 'D_k'}, inplace=True)
    else:
        stats_df = pd.DataFrame(columns=['n_k_sampled', 'D_k'])

    codebook_df['n_k'] = n_k
    codebook_df['pi_k'] = n_k / N_total
    codebook_df['D_k'] = codebook_df.index.map(stats_df['D_k']).fillna(0)

    codebook_df.to_parquet(codebook_fpath, index=False)
    print(f"Updated codebook saved to {codebook_fpath}")

if __name__ == '__main__':
    fire.Fire(main)
