import asyncio
import glob
import json
import os
import shutil
import sys

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
from util.vllm import async_VLLMClient

PROMPT = """
You will be given several clusters of value expressions that were all assigned the SAME code name "{code_name}", even though they come from distinct clusters and may represent different underlying value concepts.

Definition of a value:
- A value = what is considered inherently worthwhile, meaningful, or admirable.
- A value is NOT a topic, strategy, behavior, advice, or meta-importance.

For each cluster below (identified by its Cluster ID), a sample of its most representative value expressions is listed, ordered from most to least central to the cluster.

Your task: Assign EACH cluster a new, distinct code name that stays faithful to the shared theme "{code_name}" but captures what specifically distinguishes that cluster from the other clusters listed below.

Guidelines for code names:
- Use a noun or noun phrase (1-3 words).
- Capture how something is valued, not just what.
- Avoid generic or meta labels (e.g., Importance, Need, Utility).
- The new code names must all be DIFFERENT from one another and must reflect genuine distinctions between the clusters below. Do not simply reuse "{code_name}" unchanged for any cluster.

────────────────────────────────
Examples of code names: Individual Autonomy, Relational Connectedness, Social Responsibility, Fairness, Honesty, Authenticity, Humility, Animal Welfare
────────────────────────────────
{clusters}
────────────────────────────────
Respond in JSON format only, mapping each Cluster ID (as a string) to its new code name. Example: { "{example_id}": "New Code Name", ... }
""".strip()


def cos_sim(a, b):
    return np.dot(a, b)


def sample_top_value_expressions(row, ve_df, criterion, N):
    ve_ids = row['ve_list']
    if ve_ids is None or len(ve_ids) == 0:
        return []

    selected_ve_df = ve_df.loc[ve_ids].copy()
    centroid_embedding = row['centroid_embedding']

    selected_ve_df['similarity_to_centroid'] = selected_ve_df['embedding'].apply(
        lambda x: cos_sim(centroid_embedding, x)
    )
    selected_ve_df = selected_ve_df.sort_values(by='similarity_to_centroid', ascending=False)

    texts = [x for x in selected_ve_df[criterion].tolist() if isinstance(x, str) and x.strip() != '']
    return texts[:N]


def build_group_prompt(code_name, group_df, ve_df, criterion, N):
    cluster_blocks = []
    for _, row in group_df.iterrows():
        samples = sample_top_value_expressions(row, ve_df, criterion, N)
        block = f"Cluster ID: {row['code_id']}\nValue expressions: {'; '.join(samples)}"
        cluster_blocks.append(block)

    templated_prompt = PROMPT.replace('{code_name}', code_name)
    templated_prompt = templated_prompt.replace('{clusters}', '\n\n'.join(cluster_blocks))
    templated_prompt = templated_prompt.replace('{example_id}', str(group_df.iloc[0]['code_id']))
    return templated_prompt


def parse_new_code_names(output):
    output = output[output.find('{'):output.rfind('}') + 1]
    try:
        parsed = json.loads(output)
        return {str(k): v for k, v in parsed.items() if isinstance(v, str) and v.strip() != ''}
    except Exception:
        print(f"Parsing error: {output}")
        return {}


def get_duplicate_names(codebook):
    name_counts = codebook['code_name'].value_counts()
    return name_counts[name_counts > 1].index.tolist()


def find_best_codebook_fpath(codebook_dir):
    best_loss = None
    best_fpath = None
    for loss_csv in sorted(glob.glob(os.path.join(codebook_dir, 't*', 'reconstruction_loss*.csv'))):
        t_dir = os.path.dirname(loss_csv)
        candidate_fpath = os.path.join(t_dir, 'code_book.parquet')
        if not os.path.isfile(candidate_fpath):
            continue
        loss = pd.read_csv(loss_csv)['loss'].iloc[0]
        print(f"{os.path.basename(t_dir)}: loss={loss:.6f}")
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_fpath = candidate_fpath
    return best_fpath


def run_one_pass(codebook, value_expressions, duplicate_names, N, criterion, client, batch_size):
    groups = []
    prompts = []
    for code_name in tqdm(duplicate_names, desc="Building disambiguation prompts"):
        group_df = codebook[codebook['code_name'] == code_name]
        groups.append((code_name, group_df))
        prompts.append(build_group_prompt(code_name, group_df, value_expressions, criterion, N))

    outputs = []
    for i in range(0, len(prompts), batch_size):
        results = asyncio.run(client.process_batch_async(prompts[i:i + batch_size]))
        outputs.extend(r['out_text'] for r in results)

    new_names = codebook['code_name'].copy()
    n_parse_failures = 0

    for (code_name, group_df), output in tqdm(list(zip(groups, outputs)), desc="Applying new code names"):
        parsed = parse_new_code_names(output)
        if not parsed:
            n_parse_failures += 1

        assigned_in_group = set()
        for _, row in group_df.iterrows():
            code_id = row['code_id']
            candidate = parsed.get(str(code_id))

            if not candidate:
                candidate = f"{code_name} ({code_id})"

            candidate = candidate.strip()
            if candidate in assigned_in_group:
                candidate = f"{candidate} ({code_id})"
            assigned_in_group.add(candidate)

            new_names.loc[row.name] = candidate

    codebook = codebook.copy()
    codebook['code_name'] = new_names
    return codebook, n_parse_failures


def main(
    codebook_fpath: str = '',
    codebook_dir: str = '',  # alternative to codebook_fpath: auto-pick the lowest-loss iteration under here
    result_fpath: str = '',
    ve_embedding_fpath: str = '',
    N: int = 30,  # max number of value expressions to be considered for each code
    criterion: str = 'description',  # which column of the value expression df to show the model
    model: str = 'gpt-5.2',
    base_url: str = 'https://api.openai.com/v1',
    temperature: float = 1.0,
    api_key: str = '',
    max_iterations: int = 10,  # keep re-running disambiguation until no duplicate code names remain (or this cap is hit)
    batch_size: int = 150,
):
    if not codebook_fpath:
        if not codebook_dir:
            raise ValueError("Provide either --codebook_fpath or --codebook_dir.")
        best_fpath = find_best_codebook_fpath(codebook_dir)
        if best_fpath is None:
            raise FileNotFoundError(
                f"No scored iteration codebook found under {codebook_dir}/t*/reconstruction_loss*.csv"
            )
        codebook_fpath = os.path.join(codebook_dir, 'codebook.parquet')
        shutil.copy(best_fpath, codebook_fpath)
        print(f"Using {best_fpath} (lowest reconstruction loss) as the codebook to finalize -> {codebook_fpath}")

    codebook = pd.read_parquet(codebook_fpath)
    value_expressions = pd.read_parquet(ve_embedding_fpath)
    codebook['old_code_name'] = codebook['code_name']

    print(f"Total codes: {len(codebook)}")

    client = async_VLLMClient(model=model, base_url=base_url, api_key=api_key, temperature=temperature)

    iteration = 0
    duplicate_names = get_duplicate_names(codebook)
    while duplicate_names:
        iteration += 1
        n_dup_codes = codebook['code_name'].isin(duplicate_names).sum()
        print(f"=== Iteration {iteration}: {len(duplicate_names)} duplicate code names covering {n_dup_codes} codes ===")

        if iteration > max_iterations:
            print(f"WARNING: reached max_iterations={max_iterations} with {len(duplicate_names)} duplicate code names still remaining. Stopping.")
            break

        codebook, n_parse_failures = run_one_pass(codebook, value_expressions, duplicate_names, N, criterion, client, batch_size)
        print(f"Parse failures: {n_parse_failures} / {len(duplicate_names)} groups")

        duplicate_names = get_duplicate_names(codebook)

    if not duplicate_names:
        print(f"No duplicate code names remain after {iteration} iteration(s).")

    os.makedirs(os.path.dirname(result_fpath), exist_ok=True)
    codebook.to_parquet(result_fpath, index=False)
    print(f"Result saved to: {result_fpath}")


if __name__ == '__main__':
    fire.Fire(main)
