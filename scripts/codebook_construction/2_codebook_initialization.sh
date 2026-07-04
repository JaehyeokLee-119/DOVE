src_dir=$1
doc_dir=$2

codebook_dir=$3
tau1=$4

script_dir=$(dirname "$(realpath "$0")")
source "$script_dir/../config.sh"

tau_1=$tau1

ve_embedding_fpath=$doc_dir/value_expression_embedding.parquet

mkdir -p $codebook_dir/initialization

uv run $src_dir/codebook_construction/2_merge_clusters.py \
    --ve_embedding_fpath $ve_embedding_fpath \
    --result_fpath $codebook_dir/initialization/clustering_result.parquet \
    --tau_1 ${tau_1} \
    --min_cluster_size 5

mkdir -p $codebook_dir/t00

uv run $src_dir/codebook_construction/3_label_clusters.py \
    --ve_embedding_fpath $ve_embedding_fpath \
    --f_cluster $codebook_dir/initialization/clustering_result.parquet \
    --result_fpath $codebook_dir/initialization/code_book.parquet \
    --N_sample 100 \
    --tau_merge ${tau_1} \
    --batch_size $CLUSTER_LABELING_BATCH_SIZE \
    --temperature $CLUSTER_LABELING_TEMPERATURE \
    --model $CLUSTER_LABELING_MODEL \
    --base_url $CLUSTER_LABELING_BASE_URL \
    --api_key "$API_KEY"