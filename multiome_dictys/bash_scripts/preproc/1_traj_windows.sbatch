#!/bin/bash
#SBATCH --job-name="dynamic"
#SBATCH --output="dictys_dynamic.log"

#SBATCH -p RM
#SBATCH -N 1 # by default for gpu-shared you can only request 1 node
#SBATCH -t 12:00:00 # wall-time
#SBATCH --ntasks-per-node 128 # how many cards do you want?

# Set the working directory to where data and makefiles are located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs"
# Change directory to the working directory
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

################ RUN dictys in dynamic mode ################

module load anaconda3/2022.10
source activate dictys
mkdir -p tmp_dynamic
if ! [ -e tmp_dynamic/subsets.txt ] || ! [ -e tmp_dynamic/subset_locs.h5 ] || ! [ -e tmp_dynamic/subset_edges.tsv.gz ]; then python3 -m dictys dynamic subsets_rna data/traj_node.h5 data/traj_cell_rna.h5 data/coord_rna.tsv.gz tmp_dynamic/subsets.txt tmp_dynamic/subset_locs.h5 tmp_dynamic tmp_dynamic/subset_edges.tsv.gz 3000 2625 0.05; fi

