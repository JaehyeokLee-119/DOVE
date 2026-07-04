
import pandas as pd
import time 
from tqdm import tqdm 
import os 
import pyarrow as pa
import pyarrow.parquet as pq
import math
import fire
import numpy
import ast
import asyncio
import dotenv
dotenv.load_dotenv(dotenv.find_dotenv())

from openai import RateLimitError, APIError, APITimeoutError
import random
from openai import AsyncOpenAI

import tiktoken

DEFAULT_MODEL = 'text-embedding-3-large'

async def embed_batch(client, contents, deployment_name):
    resp = await client.embeddings.create(
        model=deployment_name,
        input=contents
    )
    return [d.embedding for d in resp.data]

def get_truncated_contents(contents, model_name):
    MODEL_LIMITS = {
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
        "text-embedding-ada-002": 8191,
    }
    
    max_limit = MODEL_LIMITS.get(model_name, 8191)
    
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    truncated_list = []
    for text in contents:
        tokens = encoding.encode(text)
        if len(tokens) > max_limit:
            truncated_text = encoding.decode(tokens[:max_limit])
            truncated_list.append(truncated_text)
        else:
            truncated_list.append(text)
            
    return truncated_list

async def embed_batch_with_retry(
    client,
    contents,
    deployment_name,
    max_retries=5,
    base_delay=1.0,
    max_delay=30.0,
):
    attempt = 0

    while True:
        try:
            safe_contents = get_truncated_contents(contents, deployment_name)
            
            resp = await client.embeddings.create(
                model=deployment_name,
                input=safe_contents,
            )
            return [d.embedding for d in resp.data]

        except (RateLimitError, APITimeoutError, APIError) as e:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(
                    f"Embedding failed after {max_retries} retries"
                ) from e

            # exponential backoff + jitter
            delay = min(
                max_delay,
                base_delay * (2 ** (attempt - 1))
            )
            delay *= random.uniform(0.8, 1.2)

            await asyncio.sleep(delay)
            
        except Exception as e:
                    print(f"Unexpected error, skipping batch: {e}")
                    return None
async def process_part_async(
    part_data,
    parquet_file_with_idx,
    colname,
    deployment_name,
    batch_size,
    part_idx,
    client,
    max_concurrency=5,
):
    writer = None
    sem = asyncio.Semaphore(max_concurrency)

    def _infer_col_types():
        if not part_data:
            return {}
        types = {}
        for col in part_data[0].keys():
            for row in part_data:
                v = row.get(col)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    types[col] = type(v).__name__
                    break
        types['embedding'] = 'list'
        return types

    col_types = _infer_col_types()

    def _to_arrow_array(col, col_vals):
        ctype = col_types.get(col)
        def _null(v):
            return v is None or (isinstance(v, float) and math.isnan(v))
        if ctype == 'list':
            return pa.array([v if v is not None else [] for v in col_vals],
                            type=pa.list_(pa.float64()))
        elif ctype == 'str':
            return pa.array([v if isinstance(v, str) else "" for v in col_vals],
                            type=pa.string())
        elif ctype == 'bool':
            return pa.array([v if isinstance(v, bool) else False for v in col_vals],
                            type=pa.bool_())
        elif ctype == 'int':
            return pa.array([v if isinstance(v, int) and not isinstance(v, bool) else 0
                             for v in col_vals], type=pa.int64())
        elif ctype == 'float':
            return pa.array([v if not _null(v) else 0.0 for v in col_vals],
                            type=pa.float64())
        else:
            return pa.Array.from_pandas(pd.Series(col_vals))

    async def process_one_batch(i):
        async with sem:
            max_ = min(i + batch_size, len(part_data))
            batch = part_data[i:max_]
            contents = [item[colname] for item in batch]

            embeddings = await embed_batch_with_retry(
                client,
                contents,
                deployment_name,
                max_retries=5,
            )
            for j in range(len(batch)):
                batch[j]["embedding"] = embeddings[j] if embeddings is not None else None

            return batch

    tasks = []
    for i in range(0, len(part_data), batch_size):
        tasks.append(process_one_batch(i))

    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Part {part_idx}",
    ):
        batch = await coro
        df_batch = pd.DataFrame(batch)

        arrow_arrays = {col: _to_arrow_array(col, df_batch[col].tolist())
                        for col in df_batch.columns}
        table = pa.Table.from_pydict(arrow_arrays)

        if writer is None:
            writer = pq.ParquetWriter(parquet_file_with_idx, table.schema)

        writer.write_table(table)

    if writer:
        writer.close()

    print(f"Part {part_idx} processed and saved to {parquet_file_with_idx}")


def main(
    fname = '',
    result_fpath = '',
    colname = 'output',
    n_parts = 1,
    start_n = 0,
    batch_size = 150,
    measure_cost = False,
    base_url = '',
    api_key = '',
    model_name = '',
):
    measure_cost = bool(measure_cost)

    client_kwargs = {}
    if base_url:
        client_kwargs['base_url'] = base_url
    if api_key:
        client_kwargs['api_key'] = api_key
    client = AsyncOpenAI(**client_kwargs)

    effective_model = model_name if model_name else DEFAULT_MODEL

    df = pd.read_parquet(fname)
    
    if 'original_index' not in df.columns:
        df['original_index'] = df.index

    total_cost = 0.0
    cost_per_1M_tokens = 0.065

    if os.path.exists(result_fpath):
        existing_df = pd.read_parquet(result_fpath)
        if len(existing_df) == len(df):
            print(f"All results already exist for {result_fpath}.")
            time.sleep(0.2)
            return
        else:
            df = df.iloc[len(existing_df):].reset_index(drop=True)
            print(f"Resuming from row {len(existing_df)} for {result_fpath}.")
            result_fpath = result_fpath.replace('.parquet', '_resumed.parquet')
        
    print(f"Total rows to process: {len(df)}")
    
    os.makedirs(os.path.dirname(result_fpath), exist_ok=True)
    to_dict = df.to_dict(orient='records')

    chunk_size = math.ceil(len(to_dict) / n_parts)

    # cost for all
    if measure_cost:
        encoding = tiktoken.get_encoding("cl100k_base")
        all_text = df[colname].astype(str).tolist()
        all_tokens = sum([len(encoding.encode(text)) for text in all_text])
        total_cost = (float(all_tokens) / 1_000_000.0) * cost_per_1M_tokens
        print(f"Estimated total tokens: {all_tokens}, Estimated total cost: ${total_cost:.4f}")
        return 

    for part_idx in range(start_n, n_parts):
        start = part_idx * chunk_size
        end = start + chunk_size
        part_data = to_dict[start:end]

        parquet_file_with_idx = result_fpath.replace('.parquet', f'_part_{part_idx}.parquet') if n_parts > 1 else result_fpath

        asyncio.run(
            process_part_async(
                part_data=part_data,
                parquet_file_with_idx=parquet_file_with_idx,
                colname=colname,
                deployment_name=effective_model,
                batch_size=batch_size,
                part_idx=part_idx,
                client=client,
                max_concurrency=5,
            )
        )

    if n_parts > 1:
        print("Embeddings saved to individual parquet files.")
        print("Now merging all parts into a single parquet file...")

        tables = [pq.read_table(f"embeddings_part_{i}.parquet") for i in range(n_parts)]
        final_table = pa.concat_tables(tables)

        pq.write_table(final_table, result_fpath)
        print("Final merged parquet file created:", result_fpath)


    if os.path.exists(result_fpath):
        final_df = pd.read_parquet(result_fpath)
        if 'original_index' in final_df.columns:
            print("Sorting by original index to ensure order...")
            final_df = final_df.sort_values('original_index').reset_index(drop=True)
            final_df = final_df.drop(columns=['original_index'])
            final_df.to_parquet(result_fpath)
            print("Successfully saved sorted results.")

if __name__ == "__main__":
    fire.Fire(main)