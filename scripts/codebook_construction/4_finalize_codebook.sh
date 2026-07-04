#!/bin/bash
set -e

src_dir=$1
doc_dir=$2
codebook_dir=$3

script_dir=$(dirname "$(realpath "$0")")
source "$script_dir/../config.sh"

# bash $script_dir/4_finalize_codebook.sh $src_dir $doc_dir $codebook_dir

base_url=$FINALIZE_BASE_URL
model_name=$FINALIZE_MODEL
temperature=$FINALIZE_TEMPERATURE
N=30
criterion=description
max_iterations=10

# scripts/codebook_construction/3_reconstruction.sh writes one codebook per iteration at
# $codebook_dir/tNN/code_book.parquet, plus that iteration's reconstruction loss at
# $codebook_dir/tNN/reconstruction_loss_*.csv. 10_finalize_codebook.py picks the
# lowest-loss iteration itself (see find_best_codebook_fpath) when --codebook_fpath
# is left empty and --codebook_dir is given instead.
ve_embedding_fpath=$doc_dir/value_expression_embedding.parquet
result_fpath=$codebook_dir/finalized_codebook.parquet

echo "=== Finalizing codebook ==="
echo "codebook_dir: $codebook_dir"
echo "ve_embedding_fpath: $ve_embedding_fpath"
echo "result_fpath: $result_fpath"

uv run "$src_dir/codebook_construction/10_finalize_codebook.py" \
    --codebook_dir "$codebook_dir" \
    --result_fpath "$result_fpath" \
    --ve_embedding_fpath "$ve_embedding_fpath" \
    --N $N \
    --criterion $criterion \
    --model $model_name \
    --base_url $base_url \
    --temperature $temperature \
    --api_key "$API_KEY" \
    --max_iterations $max_iterations \
    --batch_size $FINALIZE_BATCH_SIZE