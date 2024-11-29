#!/bin/bash
#SBATCH --job-name="gene_by_cell"
#SBATCH --output="slice.out"
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=16
#SBATCH --array=0-55

# Set the working directory to where data is located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Calculate the subset number based on the job array ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Construct the input and output file paths
INPUT_RNA="tmp_dynamic/Subset${SUBSET_NUM}/names_rna.txt"
OUTPUT_EXPR="tmp_dynamic/Subset${SUBSET_NUM}/expression0.tsv.gz"

####################################### RUN COMMAND #######################################
# the cmd loads the whole gene_by_cell in memory and then slices the loaded df
module load anaconda3/2022.10
source activate dictys
# Check if input file exists
if [[ -f "$INPUT_RNA" ]]; then
    # Run the command for the corresponding subset
    dictys preproc selects_rna data/expression.tsv.gz "$INPUT_RNA" "$OUTPUT_EXPR"
else
    echo "Error: Input file $INPUT_RNA not found for Subset${SUBSET_NUM}"
    exit 1
fi
