#!/bin/bash
set -e
src_dir=$1
doc_dir=$2
codebook_dir=$3
T=$4
beta1=$5
beta2=$6
merge=$7
extend=$8
low_util_z_threshold=$9
over_util_z_threshold=${10}
starting_t=${11}
N1=${12}
N2=${13}

script_dir=$(dirname "$(realpath "$0")")
source "$script_dir/../config.sh"

doc_fpath=$doc_dir/document.parquet
document_embedding_fpath=$doc_dir/document_embedding.parquet

reconstruction_base_url=$RECONSTRUCTION_BASE_URL
reconstruction_modelname=$RECONSTRUCTION_MODEL
reconstruction_temperature=$RECONSTRUCTION_TEMPERATURE

label_base_url=$CODE_LABELING_BASE_URL
label_modelname=$CODE_LABELING_MODEL
label_temperature=$CODE_LABELING_TEMPERATURE

model_name_in_filename=${label_modelname##*/}
reconstruction_model_name_in_filename=${reconstruction_modelname##*/}


initial_codebook_fpath=$codebook_dir/initialization/code_book.parquet
if [ -f "$initial_codebook_fpath" ]; then
    mkdir -p $codebook_dir/t00
    cp $initial_codebook_fpath $codebook_dir/t00/code_book.parquet
    echo "Initial codebook copied to $codebook_dir/t00/code_book.parquet"
else
    echo "Initial codebook not found at $initial_codebook_fpath. Exiting."
    exit 1
fi

for (( t=$starting_t; t<=T; t++ ))
do
    t_var=$(printf "t%02d" $t)
    echo "Start Iteration for $t_var"
    next_codebook_dir=$codebook_dir/t$(printf "%02d" $((t+1)))

    ### Encoding (calculate q_{z_x}) 
    uv run $src_dir/codebook_construction/4_encoder.py \
        --ve_embedding_fpath $doc_dir/value_expression_embedding.parquet \
        --codebook_fpath $codebook_dir/$t_var/code_book.parquet \
        --doc_fpath $doc_fpath \
        --result_fpath $codebook_dir/$t_var/encoding_result.parquet \
        --encoder_version global_sigma
    
    ### reconstruction (build reconstruction input, generate, and drop empty outputs in one pass)
    reconstruction_output_fpath=$codebook_dir/$t_var/reconstruction_output_$reconstruction_model_name_in_filename.parquet
    reconstruction_output_postprocessed_fpath=${reconstruction_output_fpath%.parquet}_postprocessed.parquet
    uv run $src_dir/codebook_construction/5_reconstruct.py \
        --encoding_result_fpath $codebook_dir/$t_var/encoding_result.parquet \
        --codebook_fpath $codebook_dir/$t_var/code_book.parquet \
        --ve_embedding_fpath $doc_dir/value_expression_embedding.parquet \
        --result_fpath $reconstruction_output_postprocessed_fpath \
        --use_description false \
        --prompt_colname prompt \
        --N1 $N1 \
        --N2 $N2 \
        --model $reconstruction_modelname \
        --temperature $reconstruction_temperature \
        --batch_size $RECONSTRUCTION_BATCH_SIZE \
        --base_url $reconstruction_base_url \
        --api_key $API_KEY

    reconstruction_embedding_fpath=${reconstruction_output_fpath%.parquet}_embedded.parquet
    uv run $src_dir/shared/async_embedding.py \
        --fname "$reconstruction_output_postprocessed_fpath" \
        --result_fpath "$reconstruction_embedding_fpath" \
        --colname "output" \
        --n_parts 1 \
        --start_n 0 \
        --batch_size $EMBEDDING_BATCH_SIZE \
        --measure_cost False \
        --api_key $API_KEY

    ### Loss
    uv run $src_dir/codebook_construction/6_calculate_loss.py \
        --document_embedding_fpath $document_embedding_fpath \
        --reconstruction_embedding_fpath $reconstruction_embedding_fpath \
        --codebook_fpath $codebook_dir/$t_var/code_book.parquet \
        --encoding_result_fpath $codebook_dir/$t_var/encoding_result.parquet \
        --result_csv $codebook_dir/$t_var/reconstruction_loss_${reconstruction_model_name_in_filename}.csv \
        --beta1 $beta1 \
        --beta2 $beta2 \
        --N1 $N1 \
        --N2 $N2

    uv run $src_dir/codebook_construction/7_update_code_book_usage.py \
        --codebook_fpath $codebook_dir/$t_var/code_book.parquet \
        --encoding_result_fpath $codebook_dir/$t_var/encoding_result.parquet \
        --document_embedding_fpath $document_embedding_fpath \
        --reconstruction_embedding_fpath $reconstruction_embedding_fpath \
        --N1 $N1 \
        --N2 $N2

    mkdir -p $next_codebook_dir
    uv run $src_dir/codebook_construction/8_codebook_refinement.py \
        --codebook_fpath $codebook_dir/$t_var/code_book.parquet \
        --ve_embedding_fpath $doc_dir/value_expression_embedding.parquet \
        --result_fpath $next_codebook_dir/code_book_should_be_renamed.parquet \
        --codebook_dir $codebook_dir \
        --merge $merge \
        --extend $extend \
        --low_utilization_z_score_threshold $low_util_z_threshold \
        --overuse_z_score_threshold $over_util_z_threshold

    # Check for codebook iteration finished file
    current_codebook_iteration_finished_fname=$codebook_dir/codebook_iteration_finished.txt
    if [ -f "$current_codebook_iteration_finished_fname" ]; then
        echo "Codebook iteration finished file found. Exiting iterative process."
        break
    fi

    ### Renaming codes for next iteration (build labeling input, generate, and apply new names in one pass)
    uv run $src_dir/codebook_construction/9_label_codes.py \
        --codebook_fpath $next_codebook_dir/code_book_should_be_renamed.parquet \
        --ve_embedding_fpath $doc_dir/value_expression_embedding.parquet \
        --result_fpath $next_codebook_dir/code_book.parquet \
        --N_sample 100 \
        --batch_size $CODE_LABELING_BATCH_SIZE \
        --temperature $label_temperature \
        --model $label_modelname \
        --base_url $label_base_url \
        --api_key $API_KEY

    echo $next_codebook_dir/code_book.parquet
done
