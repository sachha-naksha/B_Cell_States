#!/bin/bash
#SBATCH --job-name="QC"
#SBATCH --output="QC.out"
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=16 #mem=2gb per core
#SBATCH --array=0-60

# Set the working directory to where data is located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Calculate the subset number based on the job array ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Construct the input and output file paths
INPUT_GEX="tmp_dynamic/Subset${SUBSET_NUM}/expression0.tsv.gz"
INPUT_ATAC="tmp_dynamic/Subset${SUBSET_NUM}/names_atac0.txt"
OUTPUT_GEX="tmp_dynamic/Subset${SUBSET_NUM}/expression.tsv.gz"
OUTPUT_ATAC="tmp_dynamic/Subset${SUBSET_NUM}/names_atac.txt"

####################################### RUN COMMAND #######################################
module load anaconda3/2022.10
source activate dictys
# Check if input file exists
if [[ -f "$INPUT_GEX" ]]; then
    # QC gene_by_cell per subset {min cell and gene depth 10, 10} {min cell and gene occurence 1, 1} {min cell and gene proportion 0, 0}
    # 	n_gene:
	# 	Lower bound on total read counts for gene QC
	# nc_gene:
	# 	Lower bound on number of expressed cells for gene QC
	# ncp_gene:
	# 	Lower bound on proportion of expressed cells for gene QC
	# n_cell:
	# 	Lower bound on total read counts for cell QC
	# nt_cell:
	# 	Lower bound on number of expressed genes for cell QC
	# ntp_cell:
	# 	Lower bound on proportion of expressed genes for cell QC

    dictys preproc qc_reads "$INPUT_GEX" "$OUTPUT_GEX" 50 30 0 10 10 0
else
    echo "Error: Input file $INPUT_GEX not found for Subset${SUBSET_NUM}"
    exit 1
fi

# get the atac filed to make a combined bam from the cells post QC (excluding dropped out cells)
if [[ -f "$INPUT_ATAC" ]]; then
    dictys preproc selects_atac "$OUTPUT_GEX" "$INPUT_ATAC" "$OUTPUT_ATAC"
else
    echo "Error: Input file $INPUT_ATAC not found for Subset${SUBSET_NUM}"
    exit 1
fi