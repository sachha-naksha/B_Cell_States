#!/bin/bash
#SBATCH --job-name="sort_atac_bams"
#SBATCH --output="sort_bam_all_days_%A_%a.log" # Log file for each array job
#SBATCH -p RM
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks-per-node=128 # number of cores per node; memory is 2gb per node for rm-shared
#SBATCH -t 6:00:00
#SBATCH --array=0-2 # Array with three jobs (0, 1, 2)

module load anaconda3/2022.10

# Define input BAM files and output folders
BAM_FILES=(
    "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_day0_2/outs/atac_possorted_bam.bam"
    "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_day3_4/outs/atac_possorted_bam.bam"
    "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_day5_6/outs/atac_possorted_bam.bam"
)
# Define the reference expression file
REF_EXPRESSIONS=(
    "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/sorting_atac_outs/day0_2expression.tsv.gz"
    "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/sorting_atac_outs/day3_4expression.tsv.gz"
    "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/sorting_atac_outs/day5_6expression.tsv.gz"
    )
# Create scratch directories for the jobs
OUTPUT_FOLDERS=(
    "$LOCAL/sorted_bams_day0_2"
    "$LOCAL/sorted_bams_day3_4"
    "$LOCAL/sorted_bams_day5_6"
)


# Destination directory for the sorted BAM folders
DESTINATION_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/sorting_atac_outs"
# Trap to ensure output folders are copied back on script exit
trap 'cp -r ${OUTPUT_FOLDERS[$SLURM_ARRAY_TASK_ID]} $DESTINATION_DIR/' EXIT

# Transfer the corresponding BAM file and the reference expression file to the scratch folder
cp ${BAM_FILES[$SLURM_ARRAY_TASK_ID]} $LOCAL/
cp ${REF_EXPRESSIONS[$SLURM_ARRAY_TASK_ID]} $LOCAL/

# Activate the dictys environment
source activate dictys
set -eo pipefail
cd $LOCAL/

# Run the dictys_helper script on the BAM file for this array task
INPUT_BAM=$(basename ${BAM_FILES[$SLURM_ARRAY_TASK_ID]})
REF_EXPRESSION=$(basename ${REF_EXPRESSIONS[$SLURM_ARRAY_TASK_ID]})
OUTPUT_FOLDER=${OUTPUT_FOLDERS[$SLURM_ARRAY_TASK_ID]}

# Ensure the output folder exists
mkdir -p $OUTPUT_FOLDER

dictys_helper split_bam.sh $INPUT_BAM $OUTPUT_FOLDER --section "CB:Z:" --ref_expression $REF_EXPRESSION

echo "split bam done for ${INPUT_BAM}"
