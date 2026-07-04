script_dir=$(dirname "$(realpath "$0")")
home_dir=$script_dir/..
file=$home_dir/data/resource/DOVE_topics.jsonl
# source "$script_dir/config.sh"

# ── Target model under evaluation ── #
# Edit below for evaluation run
# This is the model being *tested*, not one used to build the codebook.
TARGET_MODEL=YOUR_MODEL_NAME
TARGET_BASE_URL=YOUR_API_ENDPOINT
TARGET_API_KEY=YOUR_API_KEY
TARGET_GENERATION_BATCH_SIZE=300
TARGET_TEMPERATURE=1.0

# The evaluation target documents will be generated and stored in this directory
# : `$home_dir/outputs/$TARGET_MODEL/document_to_process/$TARGET_MODEL.parquet`
result_dir=$home_dir/outputs/$TARGET_MODEL/document_to_process
#######################################

### Setup the API for your model to be tested (edit TARGET_* in config.sh) ###
base_url=$TARGET_BASE_URL
api_key=$TARGET_API_KEY
modelname=$TARGET_MODEL

echo "Using model: $modelname"
echo "Using base URL: $base_url"

temperature=$TARGET_TEMPERATURE


mkdir -p $result_dir

filename=$(basename "$file" .jsonl)

output_path="$result_dir/${modelname}.parquet"
uv run --active $home_dir/src/shared/async_text_generation.py \
    --fname "$file" \
    --result_fpath "$output_path" \
    --batch_size $TARGET_GENERATION_BATCH_SIZE \
    --target_col_name topic \
    --output_col_name text \
    --model "$modelname" \
    --temperature $temperature \
    --base_url "$base_url" \
    --api_key "$api_key"
