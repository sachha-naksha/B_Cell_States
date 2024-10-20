#!/bin/bash
#SBATCH --job-name="binary_priors"
#SBATCH --output="binary_priors.out"
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=128
#SBATCH --array=0-91

# Set working dir
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs"
# change dir to working dir
cd $WORK_DIR || { echo "Error: couldn't change dir to $WORK_DIR"; exit 1; }

# Calculate the subset number based on the job array ID 
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 2))

# I/O
INPUT_LINKING="tmp_dynamic/Subset${SUBSET_NUM}/linking.tsv.gz"
OUTPUT_BINLINKING="tmp_dynamic/Subset${SUBSET_NUM}/binlinking.tsv.gz"

################################ Run dictys chromatin ##############################

module load anaconda3/2022.10
source activate dictys
# top 20 TF-Target links are selected to create binary 0/1 adjacency matrix of TF-Target network from chromatin priors
dictys chromatin binlinking "$INPUT_LINKING" "$OUTPUT_BINLINKING" 20
echo "Binary TF_TG priors made for ${SUBSET_NUM}"
