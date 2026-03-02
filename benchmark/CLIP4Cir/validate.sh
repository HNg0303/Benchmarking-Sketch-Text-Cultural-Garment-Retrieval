#!/bin/bash
#SBATCH --job-name=WonyoungNo1
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=12:00:00           # Time limit hrs:min:sec
#SBATCH --output=./outputs/logs/clip4cir_validate_%j.out

echo "Starting job ..."
# 1. Setup Environment
export WORKING_DIR="/media02/ltnghia31/HNguyen/BM_ICMR2026/CLIP4Cir"
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
source /media02/ltnghia31/HNguyen/miniconda3/bin/activate
conda activate ../../miniconda3/envs/benchmark


python src/validate.py \
   --dataset aodai \
   --combining-function combiner \
   --combiner-path /media02/ltnghia31/HNguyen/BM_ICMR2026/CLIP4Cir/models/combiner_trained_on_aodai_RN50x4_2026-02-03_15_37_55/saved_models/combiner.pt \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --clip-model-name RN50x4 \
   --target-ratio 1.25 \
   --transform targetpad

conda deactivate