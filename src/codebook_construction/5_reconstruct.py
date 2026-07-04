template = """Task: Generate a response based on the prompt below.

Critical Constraints:
1. Implicitly embody the provided values through your tone, arguments, and perspective.
2. Do not explicitly mention the value names or their associated probabilities.
3. Treat [probability] as the weight of influence. A higher probability implies a stronger dominance over the narrative and logic.

[Values List]
{value_expressions}

[Prompt]
{prompt_used}"""


template_wo_value = """{prompt_used}"""


import os
import sys

import fire
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
from util.vllm import async_VLLMClient, generate_with_resume


def sample_codes_without_replacement(q_z_x, max_codes=8, validity_threshold=1e-9):
    if not isinstance(q_z_x, np.ndarray):
        return np.array([]), np.array([])

    probs = np.array(q_z_x)
    original_indices = np.arange(len(probs))

    valid_mask = probs > validity_threshold
    valid_probs = probs[valid_mask]
    valid_indices = original_indices[valid_mask]

    if len(valid_probs) == 0:
        return np.array([]), np.array([])

    n_samples = min(max_codes, len(valid_probs))
    normalized_probs = valid_probs / valid_probs.sum()
    sampled_idx = np.random.choice(
        len(valid_probs),
        size=n_samples,
        replace=False,
        p=normalized_probs
    )

    sampled_indices = valid_indices[sampled_idx]
    sampled_probs = valid_probs[sampled_idx]

    sampled_probs = sampled_probs / sampled_probs.sum()

    return sampled_indices, sampled_probs


def build_reconstruction_inputs(
    encoding_result_fpath,
    codebook_fpath,
    ve_embedding_fpath,
    use_probability,
    use_description,
    N,
    N1,
    N2,
    max_codes,
    validity_threshold,
    random_seed,
    prompt_colname,
):
    np.random.seed(random_seed)

    df = pd.read_parquet(encoding_result_fpath)

    use_probability = True if use_probability in [True, 'true', 'True', 1, '1'] else False
    use_description = True if use_description in [True, 'true', 'True', 1, '1'] else False

    codebook_df = pd.read_parquet(codebook_fpath)

    if use_description:
        ve_df = pd.read_parquet(ve_embedding_fpath)

    code_lookup = {}
    desc_cache = {}

    if use_description:
        code_lookup = codebook_df.set_index('code_id')[['centroid_embedding', 've_list']].to_dict('index')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ve_idx_map = {idx: i for i, idx in enumerate(ve_df.index)}
        ve_embeddings = torch.tensor(np.stack(ve_df['embedding'].values), device=device)

        print(f"Pre-computing descriptions for {len(codebook_df)} codes...")
        for _, row in tqdm(codebook_df.iterrows(), total=len(codebook_df)):
            code_id = row['code_id']
            code_info = code_lookup.get(code_id)

            if not code_info:
                desc_cache[code_id] = ""
                continue

            centroid = torch.tensor(code_info['centroid_embedding'], device=device)
            ve_list = code_info['ve_list']

            target_tensor_idx = [ve_idx_map[idx] for idx in ve_list if idx in ve_idx_map]

            if not target_tensor_idx:
                desc_cache[code_id] = ""
                continue

            target_embs = ve_embeddings[target_tensor_idx]
            dot_product = torch.matmul(target_embs, centroid)
            norms = torch.norm(target_embs, dim=1) * torch.norm(centroid) + 1e-10
            sim_scores = dot_product / norms

            k = min(N, len(sim_scores))
            top_k_vals, top_k_indices = torch.topk(sim_scores, k)

            final_indices = [ve_list[i] for i in top_k_indices.cpu().numpy()]
            descriptions = ve_df.loc[final_indices, 'description'].tolist()
            desc_str = ', '.join([f'"{desc}"' for desc in descriptions])

            desc_cache[code_id] = desc_str

    expanded_rows = []

    print("Generating Monte Carlo samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        q_z_x = row['q_z_x']

        if not isinstance(q_z_x, np.ndarray):
            continue

        for sample_idx in range(N1):
            sampled_indices, sampled_probs = sample_codes_without_replacement(
                q_z_x, max_codes=max_codes, validity_threshold=validity_threshold
            )

            if len(sampled_indices) == 0:
                continue

            sampled_code_ids = codebook_df.iloc[sampled_indices]['code_id'].values
            sampled_code_names = codebook_df.iloc[sampled_indices]['code_name'].values

            if use_probability and use_description:
                value_parts = []
                for code_id, code_name, prob in zip(sampled_code_ids, sampled_code_names, sampled_probs):
                    desc_str = desc_cache.get(code_id, "")
                    value_parts.append(
                        f"{code_name} [{prob*100:.2f}%]\nDescriptions of the value '{code_name}': {desc_str}"
                    )
                value_expressions = "\n\n".join(value_parts)
            elif use_probability and not use_description:
                value_parts = [
                    f"{name} [{prob*100:.2f}%]"
                    for name, prob in zip(sampled_code_names, sampled_probs)
                ]
                value_expressions = ", ".join(value_parts)
            else:
                value_expressions = ", ".join(sampled_code_names)

            for gen_idx in range(N2):
                reconstruction_input = template.format(
                    value_expressions=value_expressions,
                    prompt_used=row[prompt_colname]
                )

                reconstruction_input_wo_value = template_wo_value.format(
                    prompt_used=row[prompt_colname]
                )

                expanded_row = {
                    'source_doc_idx': row['source_doc_idx'],
                    'sample_idx': sample_idx,
                    'gen_idx': gen_idx,
                    's_indices': sampled_indices.tolist(),
                    's': sampled_code_names.tolist(),
                    'z': sampled_probs.tolist(),
                    'q_z_x': q_z_x,
                    'reconstruction_input': reconstruction_input,
                    'reconstruction_input_wo_value': reconstruction_input_wo_value,
                    prompt_colname: row[prompt_colname],
                }

                for col in df.columns:
                    if col not in expanded_row:
                        expanded_row[col] = row[col]

                expanded_rows.append(expanded_row)

    result_df = pd.DataFrame(expanded_rows)
    print(f"Original documents: {len(df)}, Total reconstruction inputs: {len(result_df)}")
    print(f"Expected: {len(df)} * {N1} * {N2} = {len(df) * N1 * N2}")
    return result_df


def main(
    encoding_result_fpath: str = '',
    codebook_fpath: str = '',
    ve_embedding_fpath: str = '',
    result_fpath: str = '',
    use_probability: bool = True,
    use_description: bool = True,
    N: int = 5,
    N1: int = 1,
    N2: int = 1,
    max_codes: int = 15,
    validity_threshold: float = 1e-9,
    random_seed: int = 42,
    prompt_colname: str = 'assignment',
    model: str = 'gpt-4.1-nano-2025-04-14',
    base_url: str = 'https://api.openai.com/v1',
    temperature: float = 1.0,
    api_key: str = '',
    batch_size: int = 1440,
):
    df = build_reconstruction_inputs(
        encoding_result_fpath, codebook_fpath, ve_embedding_fpath,
        use_probability, use_description, N, N1, N2, max_codes,
        validity_threshold, random_seed, prompt_colname,
    )

    client = async_VLLMClient(model=model, base_url=base_url, api_key=api_key, temperature=temperature)
    checkpoint_path = f"{os.path.splitext(result_fpath)[0]}_raw.parquet"
    df = generate_with_resume(df, client, target_col='reconstruction_input', checkpoint_path=checkpoint_path, output_col='output', batch_size=batch_size)

    df = df[df['output'] != '']

    os.makedirs(os.path.dirname(result_fpath), exist_ok=True)
    df.to_parquet(result_fpath, index=False)
    print(f"Kept {len(df)} non-empty reconstructions.")
    print(f"Result saved to: {result_fpath}")


if __name__ == '__main__':
    fire.Fire(main)
