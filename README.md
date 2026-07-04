# Distributional Open-Ended Evaluation of LLM Cultural Value Alignment Based on Value Codebook

**DOVE (Distributional Open-ended Value Evaluation)** is an evaluation framework for cultural value alignment that compares value distributions in human- and LLM-written texts using a value codebook [ICML 2026](https://openreview.net/forum?id=z75O6LbPCF).
This repository contains the code for codebook construction and alignment evaluation.
To facilitate evaluation, we provide a codebook and human value-coding results produced using either GPT-5.2 or GPT-OSS-120B as the value extractor. You can evaluate your model using either extractor.

---

## Installation
```bash
uv sync
```
For evaluation, we provide a codebook and human value-coding results, produced with either GPT-5.2 or GPT-OSS-120B as the value extractor.
Pick whichever model you plan to use as your own value-expression extractor, and download that model's codebook and human value-encoding results from the following URL:
https://drive.google.com/drive/folders/14JBRk9eTsY2-n6jm8Phg8OXeaJL6-hin?usp=sharing

Put the downloaded files for your chosen model in `data/` as follows:
```
data/
├── codebook/
└── human_encoded/
```

---

## Usage for Evaluation
We provide a codebook and value-coding results of human documents (DOVE Set) which span four cultures (KR, JP, CN, US), available for either GPT-5.2 or GPT-OSS-120B as the value-expression extractor (see Installation above for downloading the resources for your chosen model).
The resulting scores will be stored in `outputs/<model_name>/eval_results/eval_output.csv`.
Do the following steps: 1) generate documents for the topics with your target model, 2) measure the alignment score toward the four cultures by executing:
```bash
bash scripts/measure.sh
```
To use our provided resources for evaluation, you need the same LLM you downloaded resources for (GPT-5.2 or GPT-OSS-120B) to extract value expressions from your documents, and an embedding model (OpenAI's text-embedding-3-large).

### Stage 1 - Preparing target documents
```bash
bash scripts/test_document_generation.sh
```
This step generates documents for the prepared topics in `data/resource/DOVE_topics.jsonl`. You can do this easily by running `scripts/test_document_generation.sh`, which calls the API using the model name, endpoint, and API key you set at the top of the script. The resulting document will be placed in `outputs/<model_name>/document_to_process/<model_name>.parquet`.

### Stage 2 - Run
```bash
bash scripts/measure.sh
```
This performs the evaluation in the following steps: it extracts value expressions from the target documents and embeds them, then encodes each document using a codebook, representing each document as a probability distribution over the value codes in the codebook. It then compares these distributions against the provided KR, JP, CN, and US reference value distributions, such as `data/human_encoded/human_docs_KR_encoded.parquet`.

## Usage for Codebook Construction
Given a set of training documents, construct a value codebook from scratch by extracting value expressions, clustering and naming them into codes, and iteratively refining the codebook through document reconstruction.


Before running the script, you need to:

1. Prepare a training document set.
2. Configure `config.sh` by specifying the model names and API endpoints used for codebook construction.

Optionally, you can modify the hyperparameters defined in `train_codebook.sh`.

Place the training document file at `documents/<document_name>/document.parquet`. The file must contain the following columns:

- `text`: the document text.
- `prompt`: the prompt or topic used to generate the document (used to regenerate documents during the reconstruction stage).
- `q_idx`: a unique identifier for the prompt/topic.

Then, edit `document_name` (the folder under `documents/` containing `document.parquet`), `codebook_name`, and the refinement hyperparameters (`T`, `beta1`, `beta2`, `merge`, `extend`, etc.) at the top of `train_codebook.sh`, and run:

```bash
bash scripts/train_codebook.sh
```

**`scripts/train_codebook.sh` runs the following four stages end-to-end.** Each stage can also be run individually:
### Stage 1 - Preparation
```bash
bash scripts/1_preparation.sh <src_dir> <doc_dir>
```
Embeds the training documents in `<doc_dir>/document.parquet` and extracts + embeds the value expressions they contain, producing `document_embedding.parquet`, `value_expression_extraction_postprocessed.parquet`, and `value_expression_embedding.parquet` under `<doc_dir>`.

### Stage 2 - Codebook initialization
```bash
bash scripts/2_codebook_initialization.sh <src_dir> <doc_dir> <codebook_dir> <tau1>
```
Clusters the extracted value expressions (merging near-duplicate clusters above similarity `tau1`) and prompts an LLM to name each cluster, producing the initial codebook (T=0) at `<codebook_dir>/initialization/code_book.parquet`.

### Stage 3 - Iterative refinement
```bash
bash scripts/3_reconstruction.sh <src_dir> <doc_dir> <codebook_dir> <T> <beta1> <beta2> <merge> <extend> <low_util_z_threshold> <over_util_z_threshold> <starting_t> <N1> <N2>
```
Repeats, for `t = starting_t ... T`: (1) encode each document into a probability distribution over codes, (2) sample codes per document and have an LLM reconstruct the document from them, (3) compare original vs. reconstructed embeddings to compute a reconstruction loss and update per-code usage statistics, (4) merge overused / extend underused codes when `merge`/`extend` are enabled, and (5) re-label the resulting codes to produce the next iteration's codebook (`t{t+1}/code_book.parquet`). The loop exits early if `<codebook_dir>/codebook_iteration_finished.txt` appears.

### Stage 4 - Finalize codebook
```bash
bash scripts/4_finalize_codebook.sh <src_dir> <doc_dir> <codebook_dir>
```
Disambiguates codes that ended up sharing the same name but represent different value expressions, producing the final codebook (`finalized_codebook.parquet`) used for evaluation.

---

## Repository Layout
```
DOVE/
├── data/                        # provided resources for evaluation
│   ├── codebook/                # codebook.parquet
│   ├── human_encoded/           # human_docs_{KR,JP,CN,US}_encoded.parquet
│   └── resource/                # topics used to generate test target documents (DOVE set)
├── documents/                  # training documents used to construct a codebook
├── scripts/                    # shell entry points for the pipeline
│   ├── config.sh                       # shared model / API key configuration
│   ├── test_document_generation.sh     # generate target-model documents for evaluation
│   ├── measure.sh                      # run the evaluation pipeline
│   ├── train_codebook.sh               # run the full codebook-construction pipeline end-to-end
│   └── codebook_construction/          # internal stages driven by train_codebook.sh (1_preparation.sh to 4_finalize_codebook.sh)
├── src/
│   ├── codebook_construction/   # python codes for codebook construction
│   ├── evaluation/              # python codes for alignment evaluation of given documents and codebook
│   └── shared/                  # python codes for API call
```

Runtime data created automatically while running the pipeline scripts:

```
documents/<document_name>/              # created by codebook_construction/1_preparation.sh (Stage 1)
├── document.parquet                    # input: raw training documents (column: text)
├── document_embedding.parquet          # embedding of `text`
├── value_expression_extraction_postprocessed.parquet   # extracted value expressions
└── value_expression_embedding.parquet                  # embedding of extracted value expressions

codebooks/<codebook_name>/              # working directory for codebook construction
├── initialization/
│   ├── clustering_result.parquet
│   └── code_book.parquet               # initial codebook (T=0)
├── t00/, t01/, ... t{T}/                # one directory per refinement iteration
│   ├── code_book.parquet
│   ├── encoding_result.parquet
│   └── reconstruction_output_*.parquet
└── finalized_codebook.parquet          # final codebook

outputs/<target_model_name>/                    # created by scripts/measure.sh (Usage for Evaluation)
├── document_to_process/                # target-model documents to be scored
├── targets_ve_extracted/               # value expressions extracted from those documents
├── ve_embeddings/                      # embeddings of the extracted value expressions
├── encoding_results/                   # per-document value-code distributions (q_z_x)
└── eval_results/                       # eval_output.csv -- final DOVE alignment scores
```

---

## Citation
```
@inproceedings{
    lee2026distributional,
    title={Distributional Open-Ended Evaluation of {LLM} Cultural Value Alignment Based on Value Codebook},
    author={Jaehyeok Lee and Xiaoyuan Yi and Jing Yao and Hyunjin Hwang and Roy Ka-Wei Lee and Xing Xie and JinYeong Bak},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=z75O6LbPCF}
}
```
