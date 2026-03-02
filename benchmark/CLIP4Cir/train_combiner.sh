#!/bin/bash
#SBATCH --job-name=WonyoungNo1
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00           # Time limit hrs:min:sec
#SBATCH --output=./outputs/logs/train_combiner_%j.out

echo "Starting job ..."
# 1. Setup Environment
export WORKING_DIR="/media02/ltnghia31/HNguyen/BM_ICMR2026"
source /media02/ltnghia31/HNguyen/miniconda3/bin/activate
conda activate ../miniconda3/envs/benchmark

python src/combiner_train.py \
   --dataset aodai \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --num-epochs 10 \
   --clip-model-name RN50x4 \
   --combiner-lr 2e-5 \
   --batch-size 1024 \
   --clip-bs 32 \
   --transform targetpad \
   --target-ratio 1.25 \
   --save-training \
   --save-best \
   --validation-frequency 1

conda deactivate