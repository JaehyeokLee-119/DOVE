src_dir=$1
doc_dir=$2

script_dir=$(dirname "$(realpath "$0")")
source "$script_dir/../config.sh"

doc_fpath=$doc_dir/document.parquet
document_embedding_fpath=${doc_fpath%.parquet}_embedding.parquet
ve_extraction_postprocessed_fpath=$doc_dir/value_expression_extraction_postprocessed.parquet
ve_embedding_fpath=$doc_dir/value_expression_embedding.parquet


# Get document embedding
if [ ! -f "$document_embedding_fpath" ]; then
  uv run $src_dir/shared/async_embedding.py \
    --fname "$doc_fpath" \
    --result_fpath "$document_embedding_fpath" \
    --colname "text" \
    --n_parts 1 \
    --start_n 0 \
    --batch_size $EMBEDDING_BATCH_SIZE \
    --measure_cost False \
    --base_url $VE_EXTRACTION_BASE_URL \
    --api_key "$API_KEY" \
    --model_name $EMBEDDING_MODEL
fi

# Extract value expressions and get their embeddings
uv run --active $src_dir/codebook_construction/1_extract_value_expressions.py \
    --doc_fpath $doc_fpath \
    --result_fpath $ve_extraction_postprocessed_fpath \
    --model $VE_EXTRACTION_MODEL \
    --batch_size $VE_EXTRACTION_BATCH_SIZE \
    --temperature $VE_EXTRACTION_TEMPERATURE \
    --sample 0 \
    --base_url $VE_EXTRACTION_BASE_URL \
    --api_key "$API_KEY"

uv run --active $src_dir/shared/async_embedding.py \
    --fname $ve_extraction_postprocessed_fpath \
    --result_fpath $ve_embedding_fpath \
    --colname description \
    --batch_size $EMBEDDING_BATCH_SIZE \
    --base_url $VE_EXTRACTION_BASE_URL \
    --api_key "$API_KEY" \
    --model_name $EMBEDDING_MODEL