
script_dir=$(dirname "$(realpath "$0")")
home_dir=$script_dir/..
src_dir=$home_dir/src
source "$script_dir/config.sh"

### Config: Value-expression extraction model (Should be same as the model used for codebook construction) ###
base_url=$VE_EXTRACTION_BASE_URL
model_name=$VE_EXTRACTION_MODEL
temperature=$VE_EXTRACTION_TEMPERATURE
embedding_model_name=$EMBEDDING_MODEL
api_key=$API_KEY
codebook_fpath=$home_dir/data/codebook/codebook.parquet

### Load the target documents and their value expressions for evaluation
TARGET_MODEL= # Update Here
document_dir=$home_dir/outputs/$TARGET_MODEL/document_to_process
ve_dir=$home_dir/outputs/$TARGET_MODEL/targets_ve_extracted
ve_embedding_dir=$home_dir/outputs/$TARGET_MODEL/ve_embeddings
encoding_result_dir=$home_dir/outputs/$TARGET_MODEL/encoding_results
eval_result_dir=$home_dir/outputs/$TARGET_MODEL/eval_results

# 1. extract value expressions (x ➡ v)
uv run $src_dir/evaluation/sorting_docs.py \
    --document_dir $document_dir

# doc file names (*.parquet)
doc_files=($(ls $document_dir/*.parquet))

for doc_fpath in "${doc_files[@]}"; do
    filename=$(basename "$doc_fpath" .parquet)
    ve_extraction_postprocessed_fpath=$ve_dir/${filename}.parquet
    ve_embedding_fpath=$ve_embedding_dir/${filename}.parquet

    mkdir -p $ve_dir
    mkdir -p $ve_embedding_dir

    # value expression extraction (build input, generate, and postprocess in one pass)
    echo "Extracting value expressions for ${filename}..."
    uv run --active $src_dir/codebook_construction/1_extract_value_expressions.py \
        --doc_fpath $doc_fpath \
        --result_fpath $ve_extraction_postprocessed_fpath \
        --model $model_name \
        --batch_size $VE_EXTRACTION_BATCH_SIZE \
        --sample 0 \
        --temperature $temperature \
        --base_url $base_url \
        --api_key "$api_key"

    # embedding value expressions (v ➡ e_v)
    uv run --active $src_dir/shared/async_embedding.py \
        --fname $ve_extraction_postprocessed_fpath \
        --result_fpath $ve_embedding_fpath \
        --colname description \
        --base_url "$base_url" \
        --api_key "$api_key" \
        --model_name $embedding_model_name \
        --batch_size $EMBEDDING_BATCH_SIZE
done



# encoding documents to values using codebook (v,c ➡ z_x)
parquets_in_eval_target_dir=($(ls $ve_embedding_dir/*.parquet))
for f in "${parquets_in_eval_target_dir[@]}"; do
    eval_target_ve_embedding_file=${f}
    target_fname=${eval_target_ve_embedding_file##*/}
    target_fname=${target_fname%.parquet}

    mkdir -p $encoding_result_dir

    if [ -f $encoding_result_dir/${target_fname}.parquet ]; then
        echo "Encoding result for ${target_fname} already exists. Skipping encoding."
        continue
    fi

    # encode documents to values using codebook (v,c ➡ z_x)
    echo "Encoding documents for ${target_fname}..."
    uv run --active $src_dir/codebook_construction/4_encoder.py \
        --ve_embedding_fpath $eval_target_ve_embedding_file \
        --codebook_fpath $codebook_fpath \
        --doc_fpath $document_dir/${target_fname}.parquet \
        --result_fpath $encoding_result_dir/${target_fname}.parquet
done

### 4. Produce DOVE alignment score
# measure alignment between human and model (z_x, v, c ➡ alignment score)
mkdir -p $eval_result_dir

uv run --active $src_dir/evaluation/UOT_measure_alignment.py \
    --result_dir $encoding_result_dir \
    --kr_encoded_f $home_dir/data/human_encoded/human_docs_KR_encoded.parquet \
    --jp_encoded_f $home_dir/data/human_encoded/human_docs_JP_encoded.parquet \
    --cn_encoded_f $home_dir/data/human_encoded/human_docs_CN_encoded.parquet \
    --en_encoded_f $home_dir/data/human_encoded/human_docs_US_encoded.parquet \
    --codebook_fpath $codebook_fpath \
    --eval_result_dir $eval_result_dir \
    --valid_qid_file $home_dir/data/resource/qids.txt