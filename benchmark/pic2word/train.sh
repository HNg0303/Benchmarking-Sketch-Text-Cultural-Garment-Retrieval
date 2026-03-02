#!/bin/bash
#SBATCH --job-name=WonyoungNo1
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00           # Time limit hrs:min:sec
#SBATCH --output=./outputs/logs/train_pic2word_%j.out

echo "Starting job ..."
# 1. Setup Environment
export WORKING_DIR="/media02/ltnghia31/HNguyen/BM_ICMR2026/pic2word"
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
source /media02/ltnghia31/HNguyen/miniconda3/bin/activate
conda activate ../../miniconda3/envs/pic2word

python -u src/main.py \
    --save-frequency 20 \
    --train-data="/media02/ltnghia31/HNguyen/BM_ICMR2026/aodai/sketches"  \
    --dataset-type aodai \
    --warmup 10000 \
    --batch-size=256 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=100 \
    --workers=8 \
    --openai-pretrained \
    --model ViT-L/14

conda deactivate