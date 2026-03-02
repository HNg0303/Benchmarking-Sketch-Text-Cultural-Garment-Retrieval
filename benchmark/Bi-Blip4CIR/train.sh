#!/bin/bash
#SBATCH --job-name=WonyoungNo1
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00           # Time limit hrs:min:sec
#SBATCH --output=./outputs/logs/blip4cir_train_%j.out

echo "Starting job ..."
# 1. Setup Environment
export WORKING_DIR="/media02/ltnghia31/HNguyen/BM_ICMR2026/Bi-Blip4CIR"
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
source /media02/ltnghia31/HNguyen/miniconda3/bin/activate
conda activate ../../miniconda3/envs/blip

python src/combiner_train.py --dataset AoDai \
                             --num-epochs 30 --batch-size 64 --blip-bs 32 \
                             --projection-dim 2560 --hidden-dim 5120  --combiner-lr 2e-5 \
                             --transform targetpad --target-ratio 1.25 \
                             --save-training --save-best --validation-frequency 1 \
                             --experiment-name Combiner_loss_r.50_2e-5__BLIP_cos10_loss_r_.40_5e-5
