#!/bin/bash
#SBATCH --job-name="priors"
#SBATCH --output="tf_tg_priors.out"
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -t 00:30:00 
#SBATCH --ntasks-per-node=128
#SBATCH --array=0-60

# Set the working directory to where data and makefiles are located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added"
# Change directory to the working directory
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Calculate the subset number based on the job array ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Construct the input and output file paths
INPUT_GEX="tmp_dynamic/Subset${SUBSET_NUM}/expression.tsv.gz"
INPUT_WELLINGTON="tmp_dynamic/Subset${SUBSET_NUM}/wellington.tsv.gz"
INPUT_HOMER="tmp_dynamic/Subset${SUBSET_NUM}/homer.tsv.gz"
OUTPUT_BINDING="tmp_dynamic/Subset${SUBSET_NUM}/binding.tsv.gz"
OUTPUT_TSSDIST="tmp_dynamic/Subset${SUBSET_NUM}/tssdist.tsv.gz"
OUTPUT_LINKING="tmp_dynamic/Subset${SUBSET_NUM}/linking.tsv.gz"

################ RUN dictys in dynamic mode ################
# bash runs all commands sequentially
module load anaconda3/2022.10
source activate dictys
dictys chromatin binding "$INPUT_WELLINGTON" "$INPUT_HOMER" "$OUTPUT_BINDING"
dictys chromatin tssdist "$INPUT_GEX" "$INPUT_WELLINGTON" data/gene.bed "$OUTPUT_TSSDIST"
dictys chromatin linking "$OUTPUT_BINDING" "$OUTPUT_TSSDIST" "$OUTPUT_LINKING"
echo "TF_TG priors made for ${SUBSET_NUM}"
