import os
import sys

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
- Use a noun or noun phrase (1-3 words).
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


def build_labeling_prompts(df_codebook, df_embedding, criterion, N_sample):
    input_list = []
    for i in tqdm(range(len(df_codebook)), desc="Building labeling prompts"):
        row = df_codebook.iloc[i]
        centroid_embedding = row['centroid_embedding']
        selected_ve_df = df_embedding.loc[row['ve_list']]

        similarity_to_centroid = selected_ve_df['embedding'].apply(lambda x: cos_sim(centroid_embedding, x))
        selected_ve_df = selected_ve_df.copy()
        selected_ve_df['similarity_to_centroid'] = similarity_to_centroid

        selected_ve_df = selected_ve_df.sort_values(by='similarity_to_centroid', ascending=False)

        list_of_criterion = [x for x in selected_ve_df[criterion].tolist() if x != ''][:N_sample]

        templated_prompt = PROMPT.replace('{given}', ', '.join(list_of_criterion))
        templated_prompt = templated_prompt.replace('{criterion}s', criterion.replace('_', ' '))
        input_list.append(templated_prompt)

    df_codebook = df_codebook.copy()
    df_codebook['input'] = input_list
    return df_codebook


def parse_new_code_name(output):
    output = output[output.find('{'):output.rfind('}') + 1]
    try:
        return eval(output)['code_name']
    except Exception:
        print(f"Parsing error: {output}")
        return None


def main(
    codebook_fpath: str = '',
    ve_embedding_fpath: str = '',
    result_fpath: str = '',
    criterion: str = 'code_name',
    N_sample: int = 30,
    model: str = 'gpt-5.2',
    base_url: str = 'https://api.openai.com/v1',
    temperature: float = 1.0,
    api_key: str = '',
    batch_size: int = 150,
):
    df_codebook = pd.read_parquet(codebook_fpath)
    df_embedding = pd.read_parquet(ve_embedding_fpath)

    df_codebook = build_labeling_prompts(df_codebook, df_embedding, criterion, N_sample)

    client = async_VLLMClient(model=model, base_url=base_url, api_key=api_key, temperature=temperature)
    checkpoint_path = f"{os.path.splitext(result_fpath)[0]}_raw.parquet"
    df_codebook = generate_with_resume(df_codebook, client, target_col='input', checkpoint_path=checkpoint_path, output_col='output', batch_size=batch_size)

    df_codebook['new_code_name'] = df_codebook['output'].apply(parse_new_code_name)
    df_codebook['old_code_name'] = df_codebook['code_name']

    df_codebook['updated'] = df_codebook['updated'].apply(lambda x: True if x is None else x)
    df_codebook['code_name'] = df_codebook.apply(
        lambda x: x['new_code_name'] if (x['old_code_name'] == '' or x['updated'] == True) else x['old_code_name'],
        axis=1,
    )

    df_codebook = df_codebook[['code_id', 'code_name', 'centroid_embedding', 'sigma', 'cluster_size', 've_list', 'n_k', 'D_k', 'previous_code_id']]

    os.makedirs(os.path.dirname(result_fpath), exist_ok=True)
    df_codebook.to_parquet(result_fpath, index=False)
    print(f"Labeled {len(df_codebook)} codes.")
    print(f"Result saved to: {result_fpath}")


if __name__ == '__main__':
    fire.Fire(main)
