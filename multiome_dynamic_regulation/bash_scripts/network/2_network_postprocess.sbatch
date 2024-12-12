#!/bin/bash
#SBATCH --job-name="auto_normalization_submit"
#SBATCH --output="normalize_windows_submit.log"  
#SBATCH -p RM-shared           
#SBATCH -N 1            
#SBATCH --ntasks-per-node=1     
#SBATCH -t 1:00:00         

# Set working directory
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/tut_files/skin"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Load necessary modules and activate the environment
module load anaconda3/2022.10
source activate dictys

# Define a single log file in the submission directory
LOG_FILE="$SLURM_SUBMIT_DIR/normalize_log.log"

#################################### Define windows and batch size ####################################
start_window=1
end_window=53
batch_size=4  # Maximum batch size is 4 windows (can change based on requirements)

#################################### Submit jobs in batches ####################################
# Calculate total windows to process
total_windows=$((end_window - start_window + 1))
current_window=$start_window

# Loop through windows and submit jobs in batches for normalization
while [ $current_window -le $end_window ]; do
    # Calculate the upper limit for this batch (adjust batch size if fewer windows remain)
    upper_limit=$((current_window + batch_size - 1))
    if [ $upper_limit -gt $end_window ]; then
        upper_limit=$end_window
    fi

    ################################# Calculate time required for each batch of 4 windows (20 minutes per window for 32 cores) #################################
    num_windows_in_batch=$((upper_limit - current_window + 1))
    time_required=$(echo "scale=2; $num_windows_in_batch * 20 / 60" | bc)
    hours=$(echo "scale=0; $time_required / 1" | bc)
    minutes=$(echo "scale=0; ($time_required - $hours) * 60 / 1" | bc)

    # Submit the SLURM job for this batch to the RM-shared partition
    echo "Submitting normalization batch for windows $current_window to $upper_limit (Time: $hours:$minutes:00)"
    sbatch --partition=RM-shared --time="${hours}:${minutes}:00" --output="slurm-%j.out" <<-EOF
#!/bin/bash
#SBATCH --job-name="normalize_batch_${current_window}_to_${upper_limit}"
#SBATCH --output="slurm-%j.out"  # Temporary log for this batch
#SBATCH --ntasks-per-node=32      # Use 32 cores as specified in the original postprocess script
#SBATCH --time=${hours}:${minutes}:00
#SBATCH -N 1

# Change to working directory
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Load modules and activate the environment
module load anaconda3/2022.10
source activate dictys

# Run normalization for this batch of windows
for window in \$(seq $current_window $upper_limit); do
    echo "Running normalization for Subset\${window}" | tee -a "$LOG_FILE"
    dictys network normalize --nth 32 tmp_dynamic/Subset\${window}/net_weight.tsv.gz tmp_dynamic/Subset\${window}/net_meanvar.tsv.gz tmp_dynamic/Subset\${window}/net_covfactor.tsv.gz tmp_dynamic/Subset\${window}/net_nweight.tsv.gz | tee -a "$LOG_FILE"
done
EOF

    # Update current window for the next batch
    current_window=$((upper_limit + 1))

    # Sleep for a short while before checking for the next batch (optional)
    sleep 2
done

echo "All normalization batches submitted." | tee -a "$LOG_FILE"
