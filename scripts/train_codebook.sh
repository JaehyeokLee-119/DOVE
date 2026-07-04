# Run the entire pipeline: constructing a codebook

document_name=sample # training document. You need to prepare document.parquet in documents/<document_name>/
codebook_name=sample1 # The final codebook name. The final codebook will be saved in codebooks/<codebook_name>/finalized_codebook.parquet

script_dir=$(dirname "$(realpath "$0")")

home_dir=$script_dir/..
doc_dir=$home_dir/documents/$document_name
codebook_dir=$home_dir/codebooks/$codebook_name

src_dir=$script_dir/../src
echo "Source directory: $src_dir"
echo "Document directory: $doc_dir"
echo "Codebook directory: $codebook_dir"

# Document embedding & Value-expression extraction and embedding
bash $script_dir/codebook_construction/1_preparation.sh $src_dir $doc_dir

# Codebook initialization (clustering value expressions and then naming codes)
tau1=0.8
bash $script_dir/codebook_construction/2_codebook_initialization.sh $src_dir $doc_dir $codebook_dir $tau1

# # Iterative Refinement Process
T=10
beta1=0.03
beta2=0.08
merge=true
extend=true
low_util_z_threshold=-0.5
over_util_z_threshold=1.0
N1=3
N2=1

starting_t=0
bash $script_dir/codebook_construction/3_reconstruction.sh $src_dir $doc_dir $codebook_dir $T $beta1 $beta2 $merge $extend $low_util_z_threshold $over_util_z_threshold $starting_t $N1 $N2

## Finalizing codebook
bash $script_dir/codebook_construction/4_finalize_codebook.sh $src_dir $doc_dir $codebook_dir