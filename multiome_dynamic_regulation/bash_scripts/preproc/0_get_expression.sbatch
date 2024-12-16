#!/bin/bash
#SBATCH --job-name="get_expression"
#SBATCH --output="get_expression_%A_%a.log" # Log file for each array job
#SBATCH -p RM
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks-per-node=128 # number of cores per node; memory is 2gb per node for rm-shared
#SBATCH -t 6:00:00
#SBATCH --array=0-6 # Array with number of jobs = number of days

module load anaconda3/2022.10

# Define all input paths
MATRIX_DIR=(
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM1/filtered_feature_bc_matrix"
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM2/filtered_feature_bc_matrix"
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM3/filtered_feature_bc_matrix"
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM4/filtered_feature_bc_matrix"
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM5/filtered_feature_bc_matrix"
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM6/filtered_feature_bc_matrix"
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM7/filtered_feature_bc_matrix"
)

# Define the output reference expression files (on scratch)
OUTPUT_REF_EXPRESSIONS=(
    "$LOCAL/day1_expression.tsv.gz"
    "$LOCAL/day2_expression.tsv.gz"
    "$LOCAL/day3_expression.tsv.gz"
    "$LOCAL/day4_expression.tsv.gz"
    "$LOCAL/day5_expression.tsv.gz"
    "$LOCAL/day6_expression.tsv.gz"
    "$LOCAL/day7_expression.tsv.gz"
)

# Python script
PYTHON_FILE="/ocean/projects/cis240075p/asachan/bio_informatics_analysis/B_Cells_human_analysis/analysis_repo/multiome_dynamic_regulation/py_scripts/dynamic_grn/get_main_expression.py"

# Destination directory for the sorted BAM folders
DESTINATION_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_2nd_donor/expression_mtx"

# Trap to ensure output folders are copied back on script exit
trap 'cp -r ${OUTPUT_REF_EXPRESSIONS[$SLURM_ARRAY_TASK_ID]} $DESTINATION_DIR/' EXIT

# Transfer the corresponding input folders to the scratch folder
cp ${MATRIX_DIR[$SLURM_ARRAY_TASK_ID]} $LOCAL/

# Activate the dictys environment
source activate dictys
set -eo pipefail
cd $LOCAL/

# Run the python script
INPUT_FOLDER=$LOCAL/
OUTPUT_REF_EXPRESSION=$(basename ${OUTPUT_REF_EXPRESSIONS[$SLURM_ARRAY_TASK_ID]})

python $PYTHON_FILE $INPUT_FOLDER $OUTPUT_REF_EXPRESSION
