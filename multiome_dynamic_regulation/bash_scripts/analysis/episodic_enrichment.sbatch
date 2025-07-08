#!/bin/bash
#SBATCH --job-name="episodic_enrichment"
#SBATCH --output="episodic_enrichment.log"  
#SBATCH -p RM-shared           
#SBATCH -N 1            
#SBATCH --ntasks-per-node=64 #for memory     
#SBATCH -t 00:30:00 
#SBATCH --array=1-8

# Set working directory
WORK_DIR="/ocean/projects/cis240075p/asachan/bio_informatics_analysis/B_Cells_human_analysis/src/multiome_dynamic_regulation/py_scripts/analysis"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Load necessary modules and activate the environment
module load anaconda3/2022.10
source activate dictys

# Calculate the window number based on the array task ID
episode_idx=$((SLURM_ARRAY_TASK_ID))
time_slice_start=$(( (episode_idx - 1) * 5 ))
time_slice_end=$(( episode_idx * 5 ))

# I/O file names
dictys_dynamic_object_path="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2/output/dynamic.h5"
output_folder="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2/output/intermediate_tmp_files/blimp_ko/gc_repressions_added"
latent_factor_folder="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/other_files/latent_factors"
lf_gene_file="${latent_factor_folder}/feature_list_Z5_PRDM1_KO.txt"

# Run enrichment 
echo "Running enrichment for episode ${episode_idx}" 
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

run_episode(
    episode_idx=${episode_idx},
    dictys_dynamic_object_path=\"${dictys_dynamic_object_path}\",
    output_folder=\"${output_folder}\",
    latent_factor_folder=\"${latent_factor_folder}\",
    time_slice_start=${time_slice_start},
    time_slice_end=${time_slice_end},
    lf_gene_file=\"${lf_gene_file}\"
)
"

# Log completion of the subset
echo "Done with enrichment for episode ${episode_idx}"
