#!/bin/bash
#SBATCH --job-name="indirect_array"
#SBATCH --output="indirect_all.out"  
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -t 1:00:00  # 30 mins for each subset
#SBATCH --array=0-52  # Create job array for all subsets

# Set the working directory
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/tut_files/skin"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Load necessary modules and activate the environment
module load anaconda3/2022.10
source activate dictys

# Calculate the window number based on the array task ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Combined log file for all indirect effect and normalization jobs
LOG_FILE="$SLURM_SUBMIT_DIR/indirect_combined.log"

# Add a header to distinguish logs from different array jobs
echo "===================================" | tee -a "$LOG_FILE"
echo "Starting Subset${SUBSET_NUM} (Array Job ${SLURM_ARRAY_TASK_ID})" | tee -a "$LOG_FILE"
echo "===================================" | tee -a "$LOG_FILE"

# Run indirect effect for the specific window
echo "Running indirect effect for Subset${SUBSET_NUM}" | tee -a "$LOG_FILE"
dictys network indirect --nth 32 --fi_meanvar tmp_dynamic/Subset${SUBSET_NUM}/net_meanvar.tsv.gz tmp_dynamic/Subset${SUBSET_NUM}/net_weight.tsv.gz tmp_dynamic/Subset${SUBSET_NUM}/net_covfactor.tsv.gz tmp_dynamic/Subset${SUBSET_NUM}/net_iweight.tsv.gz | tee -a "$LOG_FILE"

# Log completion of the subset
echo "Done with Subset${SUBSET_NUM}" | tee -a "$LOG_FILE"
