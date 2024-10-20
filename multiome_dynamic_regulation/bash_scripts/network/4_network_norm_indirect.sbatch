#!/bin/bash
#SBATCH --job-name="indirect_normalize_array"
#SBATCH --output="indirect_normalize_all.out"  # Combined log file for all arrays
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -t 3:00:00  # Adjust time as needed
#SBATCH --array=2-93  # Create job array from Subset1 to Subset93

# Set the working directory
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }
# Calculate the window number based on the array task ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID))
# Combined log file for all indirect effect and normalization jobs
LOG_FILE="$SLURM_SUBMIT_DIR/indirect_normalize_combined.log"
# Add a header to distinguish logs from different array jobs
echo "===================================" | tee -a "$LOG_FILE"
echo "Starting Subset${SUBSET_NUM} (Array Job ${SLURM_ARRAY_TASK_ID})" | tee -a "$LOG_FILE"
echo "===================================" | tee -a "$LOG_FILE"

# Load necessary modules and activate the environment
module load anaconda3/2022.10
source activate dictys
# Run normalization after indirect effect
echo "Normalizing indirect effect for Subset${SUBSET_NUM}" | tee -a "$LOG_FILE"
dictys network normalize --nth 32 tmp_dynamic/Subset${SUBSET_NUM}/net_iweight.tsv.gz tmp_dynamic/Subset${SUBSET_NUM}/net_meanvar.tsv.gz tmp_dynamic/Subset${SUBSET_NUM}/net_covfactor.tsv.gz tmp_dynamic/Subset${SUBSET_NUM}/net_inweight.tsv.gz | tee -a "$LOG_FILE"
# Log completion of the subset
echo "Done with Subset${SUBSET_NUM}" | tee -a "$LOG_FILE"
