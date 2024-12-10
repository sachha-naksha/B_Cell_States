#!/bin/bash
#SBATCH --job-name="bam_peaks"
#SBATCH --output="window_bam_peaks.out"
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -t 1:00:00 # wall-time
#SBATCH --ntasks-per-node=128
#SBATCH --array=0-60

# Set the working directory to where data and makefiles are located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added"
# Change directory to the working directory
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Calculate the subset number based on the job array ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Construct the input and output file paths
INPUT_ATAC="tmp_dynamic/Subset${SUBSET_NUM}/names_atac.txt"
OUTPUT_BAM="tmp_dynamic/Subset${SUBSET_NUM}/reads.bam"
OUTPUT_BAI="tmp_dynamic/Subset${SUBSET_NUM}/reads.bai"
OUTPUT_PEAKS="tmp_dynamic/Subset${SUBSET_NUM}/peaks.bed"

################ RUN dictys in dynamic mode ################
# bash runs all commands sequentially
module load anaconda3/2022.10
source activate dictys
dictys chromatin macs2 --nth 100 "$INPUT_ATAC" data/bams "$OUTPUT_BAM" "$OUTPUT_BAI" "$OUTPUT_PEAKS" hs
echo "Done with peak calling for Subset${SUBSET_NUM}"