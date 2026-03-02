#!/bin/bash
#SBATCH --job-name=WonyoungNo1
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00           # Time limit hrs:min:sec
#SBATCH --output=./outputs/logs/searleXL_validate_%j.out

echo "Starting job ..."
# 1. Setup Environment
export WORKING_DIR="/media02/ltnghia31/HNguyen/BM_ICMR2026/SEARLE"
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
source /media02/ltnghia31/HNguyen/miniconda3/bin/activate
conda activate ../../miniconda3/envs/searle

python src/validate.py --eval-type searle-xl --dataset aodai --dataset-path /media02/ltnghia31/HNguyen/BM_ICMR2026/aodai