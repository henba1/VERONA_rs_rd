#!/bin/bash
#SBATCH --job-name=pgdl2
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=/projects/prjs1681/runs/slurm_outputs/pgdl2_%j.out
#SBATCH --error=/projects/prjs1681/runs/slurm_outputs/pgdl2_%j.err
#SBATCH --export=ALL

SCRIPT_DIR="/home/jvrijn/code/rs/verif/VERONA_rs_rd/rs_rd/scripts/RS_Baseline"
SCRIPT="${SCRIPT_DIR}/pgdl2_imagenet_beit.py"
# pgdl2_cifar10_onnx.py | pgdl2_cifar10_vit_p16_224.py | pgdl2_imagenet_beit.py

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

echo "Loaded modules:"
module list 2>&1

cd "${SCRIPT_DIR}"

source /gpfs/home2/jvrijn/miniforge3/etc/profile.d/conda.sh
conda activate verona_new

echo "Python location: $(which python)"
echo "Python version: $(python --version)"

python "${SCRIPT}"

echo "Job completed at: $(date)"
echo "Exit code: $?"

python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
conda deactivate

echo "Final resource usage:"
echo "Memory usage: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
echo "GPU memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'N/A')"
