

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

import fire

def compute_batch_distortion(orig_embs, recon_embs):
    N, M, D = recon_embs.shape
    orig_expanded = orig_embs.unsqueeze(1).expand(-1, M, -1)
    
    sim = F.cosine_similarity(orig_expanded, recon_embs, dim=-1)
    return 1-sim

def compute_H_q_z_x(q_z_x, eps=1e-9):
    if type(q_z_x) != np.ndarray:
        return 0
    q = np.array(q_z_x)
    H = -np.sum(q * np.log(q + eps))
    
    return H

def compute_monte_carlo_distortion(orig_tensor, recon_tensor_grouped, N1, N2):
    N, N1, N2, D = recon_tensor_grouped.shape
    recon_flat = recon_tensor_grouped.view(N, N1*N2, D)
    
    distortion_all = compute_batch_distortion(orig_tensor, recon_flat)
    distortion_grouped = distortion_all.view(N, N1, N2)
    
    distortion_per_code_set = distortion_grouped.mean(dim=2)
    distortion_per_doc = distortion_per_code_set.mean(dim=1)
    
    return distortion_per_doc.cpu().numpy()

def loss(codebook_df, df_output_embedding, alpha=0.1, beta=0.1):
    N = len(df_output_embedding)
    
    term1 = df_output_embedding['loss_distortion'].mean()
    
    term2 = (-alpha * df_output_embedding['M'] * df_output_embedding['H_q_z_x']).mean()
    
    n_k = codebook_df['n_k'].values
    q_hat = n_k / N
    H_q_hat = - np.sum(q_hat * np.log(q_hat + 1e-9))
    E_M = df_output_embedding['M'].mean()
    term3 = beta * E_M * H_q_hat
    
    loss = term1 + term2 + term3
    return loss, term1, term2, term3

def main(
    document_embedding_fpath = '',
    reconstruction_embedding_fpath = '',
    codebook_fpath = '',
    encoding_result_fpath = '',
    result_csv = '',
    beta1 = 0.3,
    beta2 = 0.08,
    N1 = 3,
    N2 = 1,
):
    df_output_embedding = pd.read_parquet(reconstruction_embedding_fpath)
    df_text_embedding = pd.read_parquet(document_embedding_fpath)
    codebook_df = pd.read_parquet(codebook_fpath)

    # document_embedding_fpath (shared/async_embedding.py) preserves document.parquet's
    # row order (it restores original order before saving), so the row position here is
    # the same 'source_doc_idx' join key that 4_encoder.py derives from document_df.index.
    # Compute it fresh rather than trusting a pre-existing column, since older document
    # sets carry an unrelated stray 'source_doc_index' column with different semantics.
    df_text_embedding['source_doc_idx'] = df_text_embedding.index

    unique_source_docs = df_output_embedding['source_doc_idx'].unique()
    df_text_embedding = df_text_embedding[df_text_embedding['source_doc_idx'].isin(unique_source_docs)]
    
    orig_array = np.stack(df_text_embedding['embedding'].values)
    N = len(orig_array)
    D = orig_array.shape[1]
    orig_tensor = torch.tensor(orig_array, dtype=torch.float32)

    
    df_output_embedding = df_output_embedding.sort_values(
        by=['source_doc_idx', 'sample_idx', 'gen_idx']
    ).reset_index(drop=True)
    
    recon_array = np.stack(df_output_embedding['embedding'].values)
    
    # Reshape to (N, N1, N2, D)
    expected_total = N * N1 * N2
    if len(recon_array) != expected_total:
        
        doc_counts = df_output_embedding.groupby('source_doc_idx').size()
        complete_docs = doc_counts[doc_counts == N1 * N2].index
        
        df_output_embedding = df_output_embedding[
            df_output_embedding['source_doc_idx'].isin(complete_docs)
        ]
        df_text_embedding = df_text_embedding[
            df_text_embedding['source_doc_idx'].isin(complete_docs)
        ]
        
        orig_array = np.stack(df_text_embedding['embedding'].values)
        N = len(orig_array)
        orig_tensor = torch.tensor(orig_array, dtype=torch.float32)
        
        recon_array = np.stack(df_output_embedding['embedding'].values)
    
    try:
        recon_tensor = torch.tensor(recon_array, dtype=torch.float32).view(N, N1, N2, D)
    except RuntimeError as e:
        raise

    distortion_per_doc = compute_monte_carlo_distortion(orig_tensor, recon_tensor, N1, N2)

    df_per_doc = df_output_embedding.groupby('source_doc_idx').first().reset_index()
    df_per_doc['loss_distortion'] = distortion_per_doc
    
    df_per_doc['H_q_z_x'] = df_per_doc['q_z_x'].apply(compute_H_q_z_x)
    df_per_doc['M'] = df_per_doc['s'].apply(len)

    # Update codebook with n_k counts
    codebook_df = codebook_df[codebook_df['sigma'] != 0]
    K = len(codebook_df)
    
    if df_per_doc['q_z_x'].isna().any():
        codebook_length = len(codebook_df)
        
        existing_valid_q = df_per_doc['q_z_x'].dropna().iloc[0] if not df_per_doc['q_z_x'].dropna().empty else None
        target_dtype = existing_valid_q.dtype if existing_valid_q is not None else np.float64
        
        df_per_doc['q_z_x'] = df_per_doc['q_z_x'].apply(
            lambda q: np.zeros(codebook_length, dtype=target_dtype) 
                    if not isinstance(q, np.ndarray) 
                    else q.astype(target_dtype)
        )
    
    try:
        q_matrix = np.stack(df_per_doc['q_z_x'].values)
    except ValueError:
        q_matrix = np.array([np.array(x) if isinstance(x, np.ndarray) else np.zeros(K) for x in df_per_doc['q_z_x']])

    n_k = q_matrix.sum(axis=0) 
    codebook_df['n_k'] = n_k

    # Compute loss
    loss_value, term1, term2, term3 = loss(codebook_df, df_per_doc, beta1, beta2)

    print(f"Loss: {loss_value:.6f}, Term1: {term1:.6f}, Term2: {term2:.6f}, Term3: {term3:.6f}")
    
    # Save results
    result_df = pd.DataFrame({
        'loss': [loss_value],
        'term1': [term1],
        'term2': [term2],
        'term3': [term3],
        'alpha': [beta1],
        'beta': [beta2],
        'K': [K],
        'E_M': [df_per_doc['M'].mean()],
        'N1': [N1],
        'N2': [N2],
    })
    
    encoding_result_df = pd.read_parquet(encoding_result_fpath)
    encoding_result_df = encoding_result_df[encoding_result_df['source_doc_idx'].isin(df_per_doc['source_doc_idx'])]
    
    distortion_map = dict(zip(df_per_doc['source_doc_idx'], df_per_doc['loss_distortion']))
    encoding_result_df['distortion'] = encoding_result_df['source_doc_idx'].map(distortion_map)
    
    encoding_result_df.to_parquet(encoding_result_fpath, index=False)
    
    result_df.to_csv(result_csv, index=False)
    print(f"Results saved to {result_csv}")

if __name__ == "__main__":
    fire.Fire(main)