# 1. Value expression extraction: build the extraction prompt per document, call the
# LLM directly, and postprocess the output into one value-expression-per-row table -
# all in a single pass instead of separate prepare-input / generate / postprocess steps.

import os
import re
import sys
import json

import fire
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
from util.vllm import async_VLLMClient, generate_with_resume

USER_PROMPT = """
Your task is to identify and code the author's Values from a given text. There are three types of similar but distinct concepts: Values, Beliefs, and Attitudes (VBA).

Values express attributes of the reality surrounding us, regarding essential qualities like honesty, integrity, openness seeing as main values. A value is a measure of worth or importance a person attaches to something; our values are often reflected in the way we live our lives. For example: I value my family or I value freedom of speech.

Beliefs are about how we think things really are. A belief is an internal feeling that something is true, even though that belief
may be unproven or irrational. For example: I believe that crossing on the stairs brings bad luck or I believe that there is life after death.

Attitudes can be considered the response that individual have to others actions and external situations. An attitude is the way a person expresses or applies their beliefs and values,
and is expressed through words and behaviour. For example: I get really upset when I hear about any form of cruelty or I hate school.

You must only code values (V:) that express or imply a normative orientation—that is, what the author aspires to, endorses, or treats as a desirable guiding principle for life, relationships, or action, even when such values are expressed implicitly, through contrast, or via reflection on past experiences.

Each code must:
- Be 1-3 words
- Be abstract and domain-independent
- Capture a single concept
- Avoid vague descriptors (e.g., balance, process, growth, learning) unless they are reformulated into a clear normative principle
- Descriptions should not contain the word 'over' or compare different specific values, as such constructions introduce unnecessary semantic noise.

[Code name examples]
"social responsibility", "fairness", "honesty", "authenticity", "humility", "individual autonomy", "animal welfare"

[Description examples]
"The author believes that a life does not need to be ideal or perfect to be worth living well.", "The author values individual autonomy and prioritizes personal self-determination in relation to decisions imposed by abstract institutions."

First, state the author's final stance in one sentence.
Then output the codes as a Python-style list of dictionaries with this exact schema:

Only code statements that support the author's final endorsed position.
Do not code opposing, hypothetical, or illustrative viewpoints used for contrast.

```python
[
    {
        "code_name": "<1-3 word abstract normative principle>",
        "description": "<1 sentence stating the normative orientation endorsed by the author>"
    },
    ...
]
```

Target text: "{target}"

Measurement subject: "Author of the text"
""".strip()


def template_formatting(template, row, target_col_name: str = 'text') -> str:
    return template.replace("{target}", row[target_col_name])


def extract_dict_list(output_text):
    output_text = output_text.replace('**', '')
    output_text = output_text.replace('*', '')

    objects = []
    candidates = re.findall(r'\{.*?\}', output_text, flags=re.DOTALL)
    for cand in candidates:
        try:
            obj = json.loads(cand)
            objects.append(obj)
        except json.JSONDecodeError:
            continue

    return objects


def main(
    doc_fpath: str = '',
    result_fpath: str = '',
    target_col_name: str = 'text',
    model: str = 'gpt-5.2',
    base_url: str = 'https://api.openai.com/v1',
    temperature: float = 1.0,
    api_key: str = '',
    batch_size: int = 300,
    sample: int = 0,
    seed: int = 42,
):
    df = pd.read_parquet(doc_fpath)
    if sample > 0:
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)

    df['input'] = df.apply(lambda row: template_formatting(USER_PROMPT, row, target_col_name), axis=1)

    client = async_VLLMClient(model=model, base_url=base_url, api_key=api_key, temperature=temperature)
    checkpoint_path = f"{os.path.splitext(result_fpath)[0]}_raw.parquet"
    df = generate_with_resume(df, client, target_col='input', checkpoint_path=checkpoint_path, output_col='output', batch_size=batch_size)

    df['output_dict_list'] = df['output'].apply(extract_dict_list)
    dict_lists = df['output_dict_list'].tolist()

    i_list = []
    excerpt_list = []
    description_list = []
    code_name_list = []
    type_list = []

    for i, dict_list in enumerate(dict_lists):
        for entry in dict_list:
            i_list.append(i) 
            excerpt_list.append(entry.get('excerpt', ''))
            description_list.append(entry.get('description', ''))
            code_name_list.append(entry.get('code_name', ''))
            type_list.append(entry.get('code_type', ''))

    new_df = pd.DataFrame({
        'source_doc_idx': i_list,
        'excerpt': excerpt_list,
        'description': description_list,
        'code_name': code_name_list,
        'code_type': type_list,
    })

    os.makedirs(os.path.dirname(result_fpath), exist_ok=True)
    new_df.to_parquet(result_fpath, index=False)
    print(f"Extracted {len(new_df)} value expressions from {len(df)} documents.")
    print(f"Result saved to: {result_fpath}")


if __name__ == "__main__":
    fire.Fire(main)
