#!/bin/bash
#SBATCH --job-name="homer"
#SBATCH --output="homer_windows.out"
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -t 00:30:00 # wall-time
#SBATCH --ntasks-per-node=128
#SBATCH --array=0-38

# Set the working directory to where data and makefiles are located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/tut_files/skin"
# Change directory to the working directory
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Calculate the subset number based on the job array ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Construct the input and output file paths
INPUT_GEX="tmp_dynamic/Subset${SUBSET_NUM}/expression.tsv.gz"
INPUT_FOOTPRINTS="tmp_dynamic/Subset${SUBSET_NUM}/footprints.bed"
OUTPUT_MOTIFS="tmp_dynamic/Subset${SUBSET_NUM}/motifs.bed"
OUTPUT_WELLINGTON="tmp_dynamic/Subset${SUBSET_NUM}/wellington.tsv.gz"
OUTPUT_HOMER="tmp_dynamic/Subset${SUBSET_NUM}/homer.tsv.gz"

################ RUN dictys in dynamic mode ################

module load anaconda3/2022.10
source activate dictys
dictys chromatin homer --nth 120 "$INPUT_FOOTPRINTS" data/JASPAR2020_vertebrates.motif data/genome "$INPUT_GEX" "$OUTPUT_MOTIFS" "$OUTPUT_WELLINGTON" "$OUTPUT_HOMER"
