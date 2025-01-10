#!/bin/bash
#SBATCH --job-name="animate"
#SBATCH --output="animate.log"  # Single log file for the submission job
#SBATCH -p RM-shared           # Use a CPU-only partition (e.g., RM)
#SBATCH -N 1            # Request 1 node
#SBATCH  --ntasks-per-node=64
#SBATCH -t 3:00:00 # Time for the submission script (can be adjusted)

module load anaconda3/2022.10
source activate dictys

python /ocean/projects/cis240075p/asachan/bio_informatics_analysis/B_Cells_human_analysis/analysis_repo/multiome_dynamic_regulation/py_scripts/analysis/animate.py