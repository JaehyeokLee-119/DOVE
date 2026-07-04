import os
import sys
from collections import defaultdict

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
from util.vllm import async_VLLMClient, generate_with_resume

PROMPT = """
You will be given a list of {criterion} about a single value concept, extracted from writings by multiple authors.
Based on the given {criterion}s, create ONE representative value code name that best captures the essence of the group.

Definition of a value:
- A value = what is considered inherently worthwhile, meaningful, or admirable.
- A value is NOT a topic, strategy, behavior, advice, or meta-importance.

Guidelines for code name:
- Use a noun or noun phrase (1–3 words).
- Capture how something is valued, not just what.
- Avoid generic or meta labels (e.g., Importance, Need, Utility).

────────────────────────────────
Examples of code names: Individual Autonomy, Relational Connectedness, Social Responsibility, Fairness, Honesty, Authenticity, Humility, Animal Welfare
────────────────────────────────
Your response should be in JSON format as follows: { "code_name": "Your Code Name Here" }.
Now evaluate the following {criterion}s, in the order of their centrality:
{given}""".strip()


def cos_sim(a, b):
    return np.dot(a, b)


def distance(a, b):
    return 1 - cos_sim(a, b)


def clean_code_names(code_name_list):
    small_code_names = [cn.lower().strip() for cn in code_name_list]
    cleaned_set = set()
    idx_to_maintain = []
    for idx, crit in enumerate(small_code_names):
        if crit not in cleaned_set:
            cleaned_set.add(crit)
            idx_to_maintain.append(idx)
    cleaned_code_names = [code_name_list[i] for i in idx_to_maintain]
    return cleaned_code_names


def merge_near_duplicate_clusters(df_cluster, tau_merge):
    embeddings = np.vstack(df_cluster['centroid_embedding'].values)
    n = len(df_cluster)

    print(f"The Number of initial clusters before merging: {n}")

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in tqdm(range(n), desc=f"Merging clusters (tau={tau_merge})"):
        emb_i = embeddings[i]
        for j in range(i + 1, n):
            emb_j = embeddings[j]
            similarity = cos_sim(emb_i, emb_j)
            if similarity >= tau_merge:
                union(i, j)

    groups = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)

    merged_rows = []
    print(f"Merging {n} clusters into {len(groups)} clusters...")

    for group_indices in groups.values():
        rep = group_indices[0]
        rep_row = df_cluster.loc[rep].copy()

        merged_codes = []
        for gi in group_indices:
            merged_codes.extend(df_cluster.loc[gi, 've_list'])

        rep_row['ve_list'] = list(dict.fromkeys(merged_codes))
        rep_row['cluster_size'] = len(rep_row['ve_list'])

        rep_row['centroid_embedding'] = np.mean(
            [df_cluster.loc[gi, 'centroid_embedding'] for gi in group_indices],
            axis=0
        )
        merged_rows.append(rep_row)

    print("\n===== MERGE RESULT =====")
    print(f"Original clusters: {n}")
    print(f"Merged clusters: {len(groups)}\n")

    return pd.DataFrame(merged_rows).reset_index(drop=True)


def build_labeling_prompts(df_cluster, df_embedding, input_column, N_sample):
    criterion_mapper = {
        'code_name': 'code names',
        'description': 'descriptions',
        'excerpt': 'excerpts',
    }
    criterion_desc = criterion_mapper[input_column]

    df_embedding['distance_to_centroid'] = None
    df_cluster[f'{input_column}_list'] = None

    for idx, row in tqdm(df_cluster.iterrows(), total=len(df_cluster)):
        target_code_index = row['ve_list']
        centroid_embedding = row['centroid_embedding']

        for code_idx in target_code_index:
            code_embedding = df_embedding.loc[code_idx, 'embedding']
            distance_to_centroid = distance(code_embedding, centroid_embedding)
            df_embedding.at[code_idx, 'distance_to_centroid'] = distance_to_centroid

        sorted_codes = sorted(
            target_code_index,
            key=lambda x: df_embedding.loc[x, 'distance_to_centroid'],
            reverse=True
        )

        df_cluster.at[idx, 've_list'] = sorted_codes

        code_name_list = [
            df_embedding.loc[code_idx, input_column]
            for code_idx in sorted_codes
        ]
        df_cluster.at[idx, f'{input_column}_list'] = code_name_list

    for idx, row in df_cluster.iterrows():
        code_names = row[f'{input_column}_list']
        cleaned_code_names = clean_code_names(code_names)[:N_sample]
        cleaned_code_names = [f'-{i+1}) "{cn}"' for i, cn in enumerate(cleaned_code_names, 1)]

        df_cluster.at[idx, 'input'] = (
            PROMPT.replace('{criterion}', criterion_desc).replace('{given}', '\n'.join(cleaned_code_names))
        )

    return df_cluster


def parse_labeling_output(text):
    text = text.replace('**', '')
    text = text.replace('*', '')
    text = text[text.find('{'):text.rfind('}') + 1]
    text = text.replace('true', 'True').replace('false', 'False')
    try:
        dict_ = eval(text)
        return dict_.get('code_name', '').strip(), dict_.get('is_value', True)
    except Exception:
        return '', True


def main(
    f_cluster: str = '',
    ve_embedding_fpath: str = '',
    result_fpath: str = '',
    input_column: str = 'description',
    N_sample: int = 20,
    tau_merge: float = 0.9,
    model: str = 'gpt-5.2',
    base_url: str = 'https://api.openai.com/v1',
    temperature: float = 1.0,
    api_key: str = '',
    batch_size: int = 150,
):
    df_cluster = pd.read_parquet(f_cluster)
    df_embedding = pd.read_parquet(ve_embedding_fpath)

    df_cluster = merge_near_duplicate_clusters(df_cluster, tau_merge)
    print(f"The Number of clusters after merging: {len(df_cluster)}")

    df_cluster = build_labeling_prompts(df_cluster, df_embedding, input_column, N_sample)

    client = async_VLLMClient(model=model, base_url=base_url, api_key=api_key, temperature=temperature)
    checkpoint_path = f"{os.path.splitext(result_fpath)[0]}_raw.parquet"
    df_cluster = generate_with_resume(df_cluster, client, target_col='input', checkpoint_path=checkpoint_path, output_col='output', batch_size=batch_size)

    df_cluster['code_name'], df_cluster['is_value'] = zip(*df_cluster['output'].apply(parse_labeling_output))

    sigma_list = []
    for code_id in tqdm(df_cluster['code_id'].unique()):
        ve_list = df_cluster[df_cluster['code_id'] == code_id]['ve_list'].values[0]
        embeddings = df_embedding.loc[ve_list, 'embedding'].tolist()

        if len(embeddings) > 1:
            emb = np.vstack(embeddings)
            mu = emb.mean(axis=0)
            sigma2 = np.mean(np.sum((emb - mu) ** 2, axis=1)) / emb.shape[1]
            sigma = np.sqrt(sigma2)
        else:
            sigma = 0.0

        sigma_list.append(sigma)

    sigma_map = dict(zip(df_cluster['code_id'].unique(), sigma_list))
    df_cluster['sigma'] = df_cluster['code_id'].map(sigma_map)

    df_cluster = df_cluster[['code_id', 'code_name', 'centroid_embedding', 'sigma', 'cluster_size', 've_list']]

    os.makedirs(os.path.dirname(result_fpath), exist_ok=True)
    df_cluster.to_parquet(result_fpath, index=False)
    print(f"Labeled {len(df_cluster)} clusters.")
    print(f"Result saved to: {result_fpath}")


if __name__ == '__main__':
    fire.Fire(main)
