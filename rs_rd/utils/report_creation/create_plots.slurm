#!/bin/bash
#SBATCH --job-name=create_plots
#SBATCH --partition=staging
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/projects/prjs1681/runs/slurm_outputs/create_plots_%j.out
#SBATCH --error=/projects/prjs1681/runs/slurm_outputs/create_plots_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

echo "Loaded modules:"
module list 2>&1

cd /home/jvrijn/code/rs/verif/VERONA_rs_rd

echo "Activating conda environment verona_new..."
source /gpfs/home2/jvrijn/miniforge3/etc/profile.d/conda.sh
conda activate verona_new

echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# Configuration variables
DATASET=${1:-"CIFAR-10"}
OUTPUT_DIR=${2:-"/gpfs/work2/0/prjs1681/runs/results/CIFAR-10/plots"}
HUE_BY=${3:-"network"}  #"network" or "verifier"


#use below as template when not using compile_by_tag.py (e.g. when comparing accross different tags)
CSV_FILES=(
    "/gpfs/work2/0/prjs1681/runs/results/CIFAR-10/certification_cifar_10_vit_base_patch16_224_in21k_finetuned_cifar10_20251223_163234_711830/result_df.csv"
    "/gpfs/work2/0/prjs1681/runs/results/CIFAR-10/certification_cifar_10_vit_base_patch16_224_in21k_finetuned_cifar10_20251223_163234_709977/result_df.csv"
    "/gpfs/work2/0/prjs1681/runs/results/CIFAR-10/certification_cifar_10_vit_base_patch16_224_in21k_finetuned_cifar10_20251223_163244_883647/result_df.csv"

)
CUSTOM_LABELS=(
    "rs_ViT-B/16_0.25"
    "rs_ViT-B/16_0.5"
    "rs_ViT-B/16_1"
)
CUSTOM_COLORS=(
    "#d7bde2"
    "#bb8fce"  
    "#7d3c98"  
)

python rs_rd/utils/report_creation/agg_report_cli.py \
    "${CSV_FILES[@]}" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --hue-by "$HUE_BY" \
    --custom-labels "${CUSTOM_LABELS[@]}" \
    --custom-colors "${CUSTOM_COLORS[@]}"

echo "Job completed at: $(date)"
echo "Exit code: $?"


conda deactivate

echo "Final resource usage:"
echo "Memory usage: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"

echo "Job finished successfully."
