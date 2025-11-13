#!/bin/bash
#SBATCH --job-name=upper_bounding
#SBATCH --partition=staging
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7:00:00
#SBATCH --output=/home/jvrijn/code/rs/rs_rd_research/runs/slurm_outputs/upper_bounding_%j.out
#SBATCH --error=/home/jvrijn/code/rs/rs_rd_research/runs/slurm_outputs/upper_bounding_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

echo "Loaded modules:"
module list 2>&1

cd /home/jvrijn/code/rs/VERONA

source /gpfs/home2/jvrijn/miniforge3/etc/profile.d/conda.sh
conda activate verona_jair

echo "Python location: $(which python)"
echo "Python version: $(python --version)"
# echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
# echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
# Concern: runtime is not comparable with certify, as adv_attacks on VERONA are not tuned optimally for data parallelism

# TODO: Add command-line argument parsing once implemented in rs_rd_pgdl2.py
# Parameters to be configurable in the future:
#   - DATASET_NAME (e.g., "CIFAR-10", "MNIST", "ImageNet")
#   - SPLIT (e.g., "train", "test")
#   - SAMPLE_SIZE (e.g., 500, 1000)
#   - SAMPLE_CORRECT_PREDICTIONS (e.g., True, False)
#   - EPSILON_LIST (e.g., start, stop, step for np.arange)
#   - EXPERIMENT_NAME (e.g., "pgd_l2", "pgd_linf")
#
# Future usage example:
#   python rs_rd/rs_rd_pgdl2.py \
#     --dataset_name CIFAR-10 \
#     --split test \
#     --sample_size 500 \
#     --sample_correct_predictions True \
#     --epsilon_start 0.00 \
#     --epsilon_stop 0.4 \
#     --epsilon_step 0.0039 \
#     --experiment_name pgd_l2


python rs_rd/rs_rd_pgdl2.py

# Print completion information
echo "Job completed at: $(date)"
echo "Exit code: $?"

# Clean up resources
echo "Cleaning up resources..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Deactivate conda environment
conda deactivate

echo "Final resource usage:"
echo "Memory usage: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
echo "GPU memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'N/A')"

echo "Job finished successfully."

# Example usage (current):
#   sbatch run_upper_bounding.slurm
#
# Example usage (future with arguments):
#   sbatch run_upper_bounding.slurm CIFAR-10 test 500 True 0.00 0.4 0.0039 pgd_l2

