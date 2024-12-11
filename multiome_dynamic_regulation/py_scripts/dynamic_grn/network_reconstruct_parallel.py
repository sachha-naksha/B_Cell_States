import os
import logging
import sys
from subprocess import Popen

# Setup logging configuration
logging.basicConfig(
    filename='reconstruction_log.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Working directory
if len(sys.argv) < 4:
    print("Usage: python script.py start_window end_window work_dir")
    sys.exit(1)
    
WORK_DIR = sys.argv[3]
os.chdir(WORK_DIR)

def run_reconstruction(subset_num):
    input_expression = f"tmp_dynamic/Subset{subset_num}/expression.tsv.gz"
    input_binlinking = f"tmp_dynamic/Subset{subset_num}/binlinking.tsv.gz"
    output_net_weight = f"tmp_dynamic/Subset{subset_num}/net_weight.tsv.gz"
    output_net_meanvar = f"tmp_dynamic/Subset{subset_num}/net_meanvar.tsv.gz"
    output_net_covfactor = f"tmp_dynamic/Subset{subset_num}/net_covfactor.tsv.gz"
    output_net_loss = f"tmp_dynamic/Subset{subset_num}/net_loss.tsv.gz"
    output_net_stats = f"tmp_dynamic/Subset{subset_num}/net_stats.tsv.gz"

    # Log the start of the job
    logging.info(f"Starting reconstruction for Subset {subset_num}")
    
    # Run dictys reconstruction command
    # Note: We don't specify the GPU ID, letting CUDA use the SLURM-assigned GPU
    command = [
        "dictys", "network", "reconstruct",
        "--device=cuda",  # Let CUDA use the GPU that SLURM assigned
        "--nth", "5",
        input_expression, input_binlinking, output_net_weight,
        output_net_meanvar, output_net_covfactor, output_net_loss, output_net_stats
    ]

    try:
        process = Popen(command)
        process.wait()  # Wait for the job to finish
        logging.info(f"Finished reconstruction for Subset {subset_num}")

    except Exception as e:
        logging.error(f"Error running reconstruction for Subset {subset_num}: {e}")

def main():
    start_subset = int(sys.argv[1])
    end_subset = int(sys.argv[2])

    # Log the start of the main job
    logging.info(f"Starting reconstruction for subsets {start_subset} to {end_subset}")

    # Process each subset sequentially
    for subset_num in range(start_subset, end_subset + 1):
        run_reconstruction(subset_num)

    logging.info(f"All jobs from subset {start_subset} to {end_subset} completed.")

if __name__ == "__main__":
    main()