#!/bin/bash
#SBATCH --job-name="network_to_file"
#SBATCH --output="network_to_file.out"
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 3:00:00  # Adjust time as needed

# Set the working directory
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/tut_files/skin"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Ensure the output directory exists
OUTPUT_DIR="output"
mkdir -p $OUTPUT_DIR  # Create the output directory if it does not exist

# Load necessary modules and activate the environment
module load anaconda3/2022.10
source activate dictys

# Run the dictys network command
dictys network tofile --dynamic data tmp_dynamic tmp_dynamic/subsets.txt $OUTPUT_DIR/dynamic.h5

# Log completion
echo "Done"
