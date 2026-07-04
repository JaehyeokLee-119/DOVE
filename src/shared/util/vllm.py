# --- Config ---
MAX_CONCURRENCY = 1440
MAX_RETRIES = 300
INITIAL_BACKOFF = 0.7
BACKOFF_JITTER = 0.25

try:
    from openai import AsyncOpenAI, AsyncAzureOpenAI
except ImportError:
    raise
import asyncio
import math
import random
from typing import List, Any, Dict
import ast
import pandas as pd
import time
import os

from tqdm import tqdm

import dotenv
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def generate_with_resume(
    df: pd.DataFrame,
    client: 'async_VLLMClient',
    target_col: str,
    checkpoint_path: str,
    output_col: str = 'output',
    batch_size: int = 300,
) -> pd.DataFrame:
    """Runs client generation over df[target_col] in batches, checkpointing to
    checkpoint_path after every batch so an interrupted run can resume instead
    of re-billing already-completed rows."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    df = df.reset_index(drop=True)

    start_index = 0
    if os.path.exists(checkpoint_path):
        existing_df = pd.read_parquet(checkpoint_path)
        if len(existing_df) == len(df) and output_col in existing_df.columns:
            df[output_col] = existing_df[output_col]
            start_index = len(df)
            for i in range(len(df) - 1, -1, -1):
                if df.loc[i, output_col] != '':
                    break
                start_index -= 1

    if output_col not in df.columns:
        df[output_col] = ''

    n = len(df)
    n_batches = math.ceil(n / batch_size)
    start_batch_idx = start_index // batch_size
    placeholders = df[output_col].tolist()

    if start_index >= n:
        print(f"All {n} rows already generated. Resuming from checkpoint: {checkpoint_path}")
        return df

    if start_index > 0:
        print(f"Resuming generation from row {start_index}/{n}.")

    for b in tqdm(range(start_batch_idx, n_batches), total=n_batches, initial=start_batch_idx, desc="Generating"):
        start = b * batch_size
        end = min((b + 1) * batch_size, n)

        batch = df.loc[start:end - 1, target_col].astype(str).tolist()
        results = asyncio.run(client.process_batch_async(batch))

        placeholders[start:end] = [r['out_text'] for r in results]
        df[output_col] = placeholders
        df.to_parquet(checkpoint_path, index=False)

    return df


class async_VLLMClient:
    def __init__(self, temperature=1.0, base_url="", model="", api_key='EMPTY', api_version="2024-12-01-preview"):
        self.api_version = api_version
        self.base_url = base_url

        if base_url == "https://api.openai.com/v1" or base_url == "" or 'openai' in base_url.lower():
            print(f"Base URL: {base_url}")
            print(f"Model: {model}")
            print(f"API Key: {api_key[:4]}...{api_key[-4:]}")
            
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        elif 'azure' in base_url.lower():
            if 'gpt-4o' in model:
                api_version = "2024-12-01-preview"
                self.client = AsyncAzureOpenAI(azure_endpoint=base_url, api_key=api_key, api_version=api_version)
                
            elif 'gpt' in model:
                api_version="2025-04-01-preview"
                print("Use Azure API ()")
                print(f"Base URL: {base_url}")
                print(f"Model: {model}")
                print(f"API Key: {api_key[:4]}...{api_key[-4:]}")
                self.client = AsyncAzureOpenAI(azure_endpoint=base_url, api_key=api_key, api_version=api_version)
                
            elif 'DeepSeek' in model or 'grok' in model:
                print("Use OpenAI in Azure")
                print(f"Base URL: {base_url}")
                print(f"Model: {model}")
                print(f"API Key: {api_key[:4]}...{api_key[-4:]}")
                self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
       
        else: 
            if 'api_key' == 'EMPTY' or api_key is None:
                api_key = OPENAI_API_KEY
                print(f"Using OPENAI_API_KEY from environment variable: {api_key[:4]}...{api_key[-4:]}")
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        self.temperature = temperature
        self.model = model

    async def process_batch_async(self, texts: List[str], system: str=None) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(MAX_CONCURRENCY)
        tasks = [
            asyncio.create_task(self._call_one(t, sem, i, system=system))
            for i, t in enumerate(texts)
        ]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda r: r["index"])
        return results

    async def process_messages_batch_async(self, messages_list: List[List[Dict]], system: str = None) -> List[Dict[str, Any]]:
        """messages_list: list of message lists (multi-turn conversations)."""
        sem = asyncio.Semaphore(MAX_CONCURRENCY)
        tasks = [
            asyncio.create_task(self._call_one_messages(msgs, sem, i, system=system))
            for i, msgs in enumerate(messages_list)
        ]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda r: r["index"])
        return results

    async def _call_one_messages(self, messages: List[Dict], sem: asyncio.Semaphore, idx: int, system: str = None) -> Dict[str, Any]:
        """messages: list of {role, content} dicts (full conversation history)."""
        attempt = 0
        backoff = INITIAL_BACKOFF
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        while True:
            try:
                async with sem:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=full_messages,
                        temperature=self.temperature,
                    )
                    out_text = resp.choices[0].message.content.strip()
                return {"index": idx, "out_text": out_text, "raw": out_text}
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed for index {idx}\t: {e}")

                err_str = str(e)
                status = getattr(e, 'status_code', None)

                if status == 401 or "401" in err_str or "Unauthorized" in err_str:
                    import sys
                    print(f"\n[FATAL] 401 Unauthorized at index {idx}: {e}")
                    sys.exit(1)

                if status == 404 or "404" in err_str or "DeploymentNotFound" in err_str:
                    import sys
                    print(f"\n[FATAL] 404 DeploymentNotFound at index {idx}: {e}")
                    sys.exit(1)

                if status == 400 or any(k in err_str for k in (
                    "400", "Bad Request",
                    "content management policy",
                    "potentially violating our usage policy",
                )):
                    print(f"Bad request error at index {idx}, returning empty response.")
                    return {"index": idx, "out_text": "", "raw": f"ERROR: {e}"}
                
                if attempt > MAX_RETRIES:
                    return {"index": idx, "out_text": "", "raw": f"ERROR: {e}"}
                
                jitter = random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER)
                wait_s = max(0.1, backoff + jitter)
                await asyncio.sleep(wait_s)
                backoff *= 2.0

    async def _call_one(self, sample_text: str, sem: asyncio.Semaphore, idx: int, system: str=None) -> Dict[str, Any]:
        if system is None:
            system = "You are a helpful assistant."
        return await self._call_one_messages(
            [{"role": "user", "content": sample_text}], sem, idx, system=system
        )
