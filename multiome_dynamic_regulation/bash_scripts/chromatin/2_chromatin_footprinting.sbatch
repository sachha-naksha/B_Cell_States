#!/bin/bash
#SBATCH --job-name="footprinting"
#SBATCH --output="footprint_windows.out"
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -t 1:00:00 # wall-time
#SBATCH --ntasks-per-node=32
#SBATCH --array=0-60

# Set the working directory to where data and makefiles are located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added"
# Change directory to the working directory
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Calculate the subset number based on the job array ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Construct the input and output file paths
INPUT_BAM="tmp_dynamic/Subset${SUBSET_NUM}/reads.bam"
INPUT_BAI="tmp_dynamic/Subset${SUBSET_NUM}/reads.bai"
INPUT_PEAKS="tmp_dynamic/Subset${SUBSET_NUM}/peaks.bed"
OUTPUT_FOOTPRINTS="tmp_dynamic/Subset${SUBSET_NUM}/footprints.bed"

################ RUN dictys in dynamic mode ################
# bash runs commands sequentially, so we can run the commands in the order they are written also in a subshell (so activate and deactivate whatever)
module load anaconda3/2022.10
source activate dictys
dictys chromatin wellington --nth 30 --fi_blacklist data/blacklist.bed "$INPUT_BAM" "$INPUT_BAI" "$INPUT_PEAKS" "$OUTPUT_FOOTPRINTS"
echo "Done with footprinting for Subset${SUBSET_NUM}"
