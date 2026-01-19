#!/bin/bash
#SBATCH --job-name=sdp_crown
#SBATCH --partition=gpu_a100 
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/prjs1681/runs/slurm_outputs/sdpcrown_%j.out
#SBATCH --error=/projects/prjs1681/runs/slurm_outputs/sdpcrown_%j.err
#SBATCH --export=ALL

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

echo "Loaded modules:"
module list 2>&1

cd /home/jvrijn/code/rs/verif/VERONA_rs_rd/rs_rd/scripts

source /gpfs/home2/jvrijn/miniforge3/etc/profile.d/conda.sh
conda activate verona_jair

echo "Python location: $(which python)"
echo "Python version: $(python --version)"

#python rs_rd/rs_rd_pgdl2.py
python sdp_crown.py 

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
