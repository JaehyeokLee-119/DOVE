#!/bin/bash
# Shared model / API key configuration for the codebook-construction pipeline
# (scripts/codebook_construction/*.sh) and the evaluation pipeline (scripts/measure.sh).
#
# Usage: source this file near the top of a pipeline script, e.g.
#   script_dir=$(dirname "$(realpath "$0")")
#   source "$script_dir/config.sh"            # scripts/measure.sh (same directory)
#   source "$script_dir/../config.sh"         # scripts/codebook_construction/*.sh (one level down)

# ── API key ──────────────────────────────────────────────────────────────
# Used by every pipeline call below (value-expression extraction, cluster
# labeling, reconstruction, code labeling, finalization, embedding).
# export OPENAI_API_KEY=sk-... before running any pipeline script.
API_KEY=${OPENAI_API_KEY:-YOUR_OPENAI_API_KEY}
BASE_OPENAI_URL=https://api.openai.com/v1

# ── Value-expression extraction (1_preparation.sh, 6_measure_new_file.sh) ──
VE_EXTRACTION_MODEL=gpt-5.2
VE_EXTRACTION_BASE_URL=$BASE_OPENAI_URL
VE_EXTRACTION_TEMPERATURE=1.0
VE_EXTRACTION_BATCH_SIZE=300

# ── Cluster labeling / codebook initialization (2_codebook_initialization.sh) ──
CLUSTER_LABELING_MODEL=gpt-5.2
CLUSTER_LABELING_BASE_URL=$BASE_OPENAI_URL
CLUSTER_LABELING_TEMPERATURE=1.0
CLUSTER_LABELING_BATCH_SIZE=300

# ── Reconstruction (3_reconstruction.sh) ──
RECONSTRUCTION_MODEL=gpt-4.1-nano-2025-04-14
RECONSTRUCTION_BASE_URL=$BASE_OPENAI_URL
RECONSTRUCTION_TEMPERATURE=1.0
RECONSTRUCTION_BATCH_SIZE=300

# ── Code labeling during iterative refinement (3_reconstruction.sh) ──
CODE_LABELING_MODEL=gpt-5.2
CODE_LABELING_BASE_URL=$BASE_OPENAI_URL
CODE_LABELING_TEMPERATURE=1.0
CODE_LABELING_BATCH_SIZE=300

# ── Codebook finalization (4_finalize_codebook.sh) ──
FINALIZE_MODEL=gpt-5.2
FINALIZE_BASE_URL=$BASE_OPENAI_URL
FINALIZE_TEMPERATURE=1.0
FINALIZE_BATCH_SIZE=300

# ── Embedding model (wherever shared/async_embedding.py is called) ──
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_BASE_URL=$BASE_OPENAI_URL
EMBEDDING_BATCH_SIZE=300