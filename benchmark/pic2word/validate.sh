#!/bin/bash
#SBATCH --job-name=WonyoungNo1
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00           # Time limit hrs:min:sec
#SBATCH --output=./outputs/logs/pic2word_validate_%j.out

echo "Starting job ..."
# 1. Setup Environment
export WORKING_DIR="/media02/ltnghia31/HNguyen/BM_ICMR2026/pic2word"
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
source /media02/ltnghia31/HNguyen/miniconda3/bin/activate
conda activate ../../miniconda3/envs/pic2word


python src/eval_retrieval.py --openai-pretrained --resume /media02/ltnghia31/HNguyen/BM_ICMR2026/pic2word/logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2026-02-11-09-47-34/checkpoints/epoch_100.pt  --eval-mode aodai --model ViT-L/14

conda deactivate