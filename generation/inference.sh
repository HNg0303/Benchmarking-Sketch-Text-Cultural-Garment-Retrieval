#!/bin/bash
#SBATCH --job-name=JangWonyoung
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --nodelist=gpu01
#SBATCH --output=./outputs/logs/inference_sana_%j.out

echo "Starting job ..."
# 1. Setup Environment
export WORKING_DIR="/media02/ltnghia31/HNguyen/ICMR2026"
source /media02/ltnghia31/HNguyen/miniconda3/bin/activate

conda activate ../miniconda3/envs/sana
# hf download black-forest-labs/FLUX.2-klein-4B --local-dir /media02/ltnghia31/models/FLUX2_4B
#pip install -r "$WORKING_DIR/requirements.txt"
echo "Environment setup completed."
echo "Starting inference..."
python ./sana_inference.py 

conda deactivate