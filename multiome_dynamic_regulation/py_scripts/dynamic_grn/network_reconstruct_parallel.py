import os
import multiprocessing
from subprocess import Popen, PIPE
import logging
import sys

# Setup logging configuration
logging.basicConfig(
    filename='reconstruction_log.log',  # Log file name
    filemode='w',  # Overwrite the log file each time
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)

# Working directory
if len(sys.argv) < 5:
    print("Usage: python script.py start_window end_window num_gpus work_dir")
    sys.exit(1)
    
WORK_DIR = sys.argv[4]
os.chdir(WORK_DIR)

# Function to get an available GPU
def get_available_gpu():
    try:
        # Check CUDA_VISIBLE_DEVICES environment variable first
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_devices is not None:
            # Return the first available GPU from SLURM allocation
            return 0  # We return 0 because in the SLURM context, this will map to the allocated GPU
        
        # Fallback to nvidia-smi only if CUDA_VISIBLE_DEVICES is not set
        result = Popen(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], stdout=PIPE)
        output, _ = result.communicate()
        
        # Parse the output
        gpu_status = output.decode('utf-8').strip().split("\n")
        
        for gpu_info in gpu_status:
            gpu_id, memory_used = gpu_info.split(", ")
            if int(memory_used) == 0:  # GPU is available if memory used is 0
                return int(gpu_id)
        return None  # No available GPU found
    except Exception as e:
        logging.error(f"Failed to query GPUs: {e}")
        return None

# Define the command template
def run_reconstruction(subset_num, gpu_id):
    input_expression = f"tmp_dynamic/Subset{subset_num}/expression.tsv.gz"
    input_binlinking = f"tmp_dynamic/Subset{subset_num}/binlinking.tsv.gz"
    output_net_weight = f"tmp_dynamic/Subset{subset_num}/net_weight.tsv.gz"
    output_net_meanvar = f"tmp_dynamic/Subset{subset_num}/net_meanvar.tsv.gz"
    output_net_covfactor = f"tmp_dynamic/Subset{subset_num}/net_covfactor.tsv.gz"
    output_net_loss = f"tmp_dynamic/Subset{subset_num}/net_loss.tsv.gz"
    output_net_stats = f"tmp_dynamic/Subset{subset_num}/net_stats.tsv.gz"

    # Log the start of the job
    logging.info(f"Starting reconstruction for Subset {subset_num} on GPU {gpu_id}")
    
    # Run dictys reconstruction command
    command = [
        "dictys", "network", "reconstruct",
        f"--device=cuda:{gpu_id}",
        "--nth", "5",
        input_expression, input_binlinking, output_net_weight,
        output_net_meanvar, output_net_covfactor, output_net_loss, output_net_stats
    ]

    try:
        process = Popen(command)
        process.wait()  # Wait for the job to finish

        # Log successful completion of the job
        logging.info(f"Finished reconstruction for Subset {subset_num} on GPU {gpu_id}")

    except Exception as e:
        # Log any errors that occur during the process execution
        logging.error(f"Error running reconstruction for Subset {subset_num} on GPU {gpu_id}: {e}")

def main():
    # Start and end subset indices and number of GPUs are passed as arguments to the script
    start_subset = int(sys.argv[1])
    end_subset = int(sys.argv[2])
    num_gpus = int(sys.argv[3])  # Number of GPUs to use

    # Log the start of the main job
    logging.info(f"Starting reconstruction for subsets {start_subset} to {end_subset} on {num_gpus} GPUs")

    # If only one GPU is provided, find an available GPU
    if num_gpus == 1:
        available_gpu = get_available_gpu()
        if available_gpu is None:
            logging.error("No available GPU found!")
            sys.exit(1)
        for subset_num in range(start_subset, end_subset + 1):
            run_reconstruction(subset_num, available_gpu)  # Use the available GPU
    else:
        ###### Create a process pool with the specified number of GPUs (not possible to find more than 1 gpu per node, hence this part is not used)
        pool = multiprocessing.Pool(num_gpus)

        # Assign each subset to a GPU in a round-robin fashion
        jobs = []
        for subset_num in range(start_subset, end_subset + 1):
            gpu_id = (subset_num - start_subset) % num_gpus  # Assign GPUs in round-robin
            jobs.append(pool.apply_async(run_reconstruction, args=(subset_num, gpu_id)))

        # Wait for all jobs to finish
        for job in jobs:
            job.get()

        # Log completion of all jobs
        logging.info(f"All jobs from subset {start_subset} to {end_subset} completed successfully.")

        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
