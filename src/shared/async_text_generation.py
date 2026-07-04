import asyncio
import math
import random
from typing import List, Any, Dict
import ast
import pandas as pd
import time 
from pathlib import Path
import os
import dotenv
dotenv.load_dotenv()
from tqdm import tqdm
import httpx
HF_TOKEN = os.getenv("HF_TOKEN")
try:
    from openai import AsyncOpenAI
except ImportError:
    raise

import fire
from util.vllm import async_VLLMClient
from transformers import AutoTokenizer

# --- Config ---
MAX_CONCURRENCY = 1440
MAX_RETRIES = 50
INITIAL_BACKOFF = 0.7 
BACKOFF_JITTER = 0.25 

def main(
    fname = '',
    result_fpath = '',
    batch_size = 100,
    sample = 0,
    seed=42,
    target_col_name: str = 'input',
    output_col_name: str = 'output',
    model = "openai/gpt-oss-120b",
    temperature = 1.0,
    base_url = "",
    api_key='',
):
    if '.csv' in fname:
        df_full = pd.read_csv(fname)
    elif '.jsonl' in fname:
        df_full = pd.read_json(fname, lines=True)
    else:
        df_full = pd.read_parquet(fname)
    print(f"Original data size: {len(df_full)}")
    if sample > 0:
        df_full = df_full.sample(n=sample, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(result_fpath), exist_ok=True)

    vLLM_client = async_VLLMClient(model=model, base_url=base_url,api_key=api_key, temperature=temperature)

    start_index = 0
    if os.path.exists(result_fpath):
        existing_df = pd.read_parquet(result_fpath)
        if len(existing_df) == len(df_full):
            start_index = len(df_full)
            for i in range(len(df_full)-1, -1, -1):
                if existing_df.iloc[i][output_col_name] != '':
                    break
                start_index -= 1
            
            if start_index == len(df_full):
                print(f"All results already exist for {fname}.")
                time.sleep(0.2)
                return

    if start_index > 0:
        print(f"Resuming from row {start_index} for {fname}.")
        existing_df = pd.read_parquet(result_fpath)
        df_full[output_col_name] = existing_df[output_col_name]
    else:
        print(f"Starting fresh processing for {fname}.")
        if output_col_name not in df_full.columns:
            df_full[output_col_name] = ''

    n = len(df_full)
    n_batches = math.ceil(n / batch_size)

    start_batch_idx = start_index // batch_size

    placeholders_for_result = df_full[output_col_name].tolist()

    for b in tqdm(range(start_batch_idx, n_batches), total=n_batches, initial=start_batch_idx):
        start = b * batch_size
        end = min((b + 1) * batch_size, n)

        batch = df_full.loc[start:end-1, target_col_name].astype(str).tolist()

        results = asyncio.run(vLLM_client.process_batch_async(batch))

        placeholders_for_result[start:end] = [r['out_text'] for r in results]
        df_full[output_col_name] = placeholders_for_result
        df_full.to_parquet(result_fpath, index=False)

    print(f"{model} Inference Done")


if __name__ == "__main__":
    fire.Fire(main)