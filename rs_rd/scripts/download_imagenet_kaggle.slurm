#!/bin/bash
#SBATCH --job-name=imagenet_download
#SBATCH --partition=staging
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/prjs1681/runs/slurm_outputs/imagenet_download_%j.out
#SBATCH --error=/projects/prjs1681/runs/slurm_outputs/imagenet_download_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

echo "Loaded modules:"
module list 2>&1

# Create output directory if it doesn't exist
mkdir -p /gpfs/work2/0/prjs1681/datasets/ImageNet

source /gpfs/home2/jvrijn/miniforge3/etc/profile.d/conda.sh
conda activate verona_new

echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "Kaggle CLI location: $(which kaggle)"

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json"
    echo "Please set up Kaggle API credentials first."
    exit 1
fi

# Ensure proper permissions on kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "Starting ImageNet-1k download from Kaggle..."
echo "Download location: /gpfs/work2/0/prjs1681/datasets/ImageNet"
echo "Dataset size: ~150GB (this may take several hours)"

# Download ImageNet-1k dataset
kaggle competitions download -c imagenet-object-localization-challenge -p /gpfs/work2/0/prjs1681/datasets/ImageNet

DOWNLOAD_EXIT_CODE=$?

if [ $DOWNLOAD_EXIT_CODE -eq 0 ]; then
    echo "Download completed successfully at: $(date)"
    echo "Checking downloaded files:"
    ls -lh /gpfs/work2/0/prjs1681/datasets/ImageNet/ | head -20
    
    echo ""
    echo "Downloaded files summary:"
    du -sh /gpfs/work2/0/prjs1681/datasets/ImageNet/*
else
    echo "ERROR: Download failed with exit code: $DOWNLOAD_EXIT_CODE"
    echo "Check the error log for details."
    exit $DOWNLOAD_EXIT_CODE
fi

# Print completion information
echo "Job completed at: $(date)"
echo "Exit code: $DOWNLOAD_EXIT_CODE"

# Deactivate conda environment
conda deactivate

echo "Final disk usage:"
df -h /gpfs/work2/0/prjs1681/datasets/ | tail -1

echo "Job finished successfully."

# Usage:
#   sbatch download_imagenet_kaggle.slurm

