#!/bin/bash
#SBATCH --job-name="ee_blimp1_ko_gc"
#SBATCH --output="ee_blimp1_ko_gc.log"  
#SBATCH -p RM-shared           
#SBATCH -N 1            
#SBATCH --ntasks-per-node=64 #for memory     
#SBATCH -t 00:30:00 
#SBATCH --array=5-8

# Set working directory where the python script is located
WORK_DIR="/ocean/projects/cis240075p/asachan/bio_informatics_analysis/B_Cells_human_analysis/src/multiome_dynamic_regulation/py_scripts/analysis"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Load necessary modules and activate the environment
module load anaconda3/2022.10
source activate dictys

# I/O file names
dictys_dynamic_object_path="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2/output/dynamic.h5"
output_folder="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2/output/intermediate_tmp_files/blimp1_ko/gc_98"
latent_factor_folder="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/other_files/latent_factors"
lf_blimp1_file="${latent_factor_folder}/feature_list_Z5_PRDM1_KO.txt"
lf_irf4_file="${latent_factor_folder}/feature_list_Z4_IRF4_KO.txt"

# Calculate the window number based on the array task ID
episode_idx=$((SLURM_ARRAY_TASK_ID))
time_slice_start=$(( (episode_idx - 1) * 5 ))
time_slice_end=$(( episode_idx * 5 ))
traj_start=1
traj_end=3
num_points=40
dist=0.001
sparsity=0.01

# Run enrichment 
echo "Running enrichment for episode ${episode_idx}" 

########################## Python script ##########################
python -c "
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import dictys
from dictys.net import stat
import joblib
import pickle
from scipy.stats import median_abs_deviation, hypergeom
import math
from utils_custom import *
from local_dynamics import *

lf_blimp1 = pd.read_csv(\"${lf_blimp1_file}\", sep='\t')['names'].tolist()
lf_irf4 = pd.read_csv(\"${lf_irf4_file}\", sep='\t')['names'].tolist()
lf_genes = list(set(lf_blimp1 + lf_irf4))

run_episode(
    episode_idx=${episode_idx},
    dictys_dynamic_object_path=\"${dictys_dynamic_object_path}\",
    output_folder=\"${output_folder}\",
    latent_factor_folder=\"${latent_factor_folder}\",
    time_slice_start=${time_slice_start},
    time_slice_end=${time_slice_end},
    lf_genes=lf_genes,
    traj_start=${traj_start},
    traj_end=${traj_end},
    num_points=${num_points},
    dist=${dist},
    sparsity=${sparsity}
)
"

# Log completion of the subset
echo "Done with enrichment for episode ${episode_idx}"
