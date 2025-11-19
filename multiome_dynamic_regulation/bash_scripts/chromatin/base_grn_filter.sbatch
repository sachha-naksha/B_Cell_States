#!/bin/bash
#SBATCH --job-name="base_grn_filter"
#SBATCH --output="base_grn_filter.out"
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -t 00:30:00 
#SBATCH --ntasks-per-node=64
#SBATCH --array=1-24

# Set the working directory to where data and makefiles are located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/outs/dictys/rbpj_ntc"
# Change directory to the working directory
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Calculate the subset number based on the job array ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Construct the input and output file paths
INPUT_GEX="tmp_dynamic/Subset${SUBSET_NUM}/expression.tsv.gz"
INPUT_WELLINGTON="tmp_dynamic/Subset${SUBSET_NUM}/wellington.tsv.gz"
INPUT_HOMER="tmp_dynamic/Subset${SUBSET_NUM}/homer.tsv.gz"
BINDING="tmp_dynamic/Subset${SUBSET_NUM}/binding.tsv.gz"
TSSDIST="tmp_dynamic/Subset${SUBSET_NUM}/tssdist.tsv.gz"
LINKING="tmp_dynamic/Subset${SUBSET_NUM}/linking.tsv.gz"
BINLINKING="tmp_dynamic/Subset${SUBSET_NUM}/binlinking.tsv.gz"

################ RUN dictys in dynamic mode ################
# bash runs all commands sequentially, so does python
module load anaconda3/2022.10
source activate dictys

python -c "
# make sure to import all functions
from dictys.chromatin import binding, tssdist, linking, binlinking
