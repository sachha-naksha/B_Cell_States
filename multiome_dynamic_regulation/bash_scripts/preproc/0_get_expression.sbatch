#!/bin/bash
#SBATCH --job-name="sort_atac_bams"
#SBATCH --output="sort_bam_all_days_%A_%a.log" # Log file for each array job
#SBATCH -p RM
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks-per-node=128 # number of cores per node; memory is 2gb per node for rm-shared
#SBATCH -t 6:00:00
#SBATCH --array=0-2 # Array with three jobs (0, 1, 2)