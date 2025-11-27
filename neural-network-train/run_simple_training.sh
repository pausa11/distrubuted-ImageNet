#!/bin/bash

# Script to run simple Tiny-ImageNet training
# Usage: ./run_simple_training.sh [additional args]

# Activate conda environment if needed
# Uncomment and modify if you use conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your-env-name

# Or activate virtualenv if needed:
# source venv/bin/activate

# Run the training script
python3 train_simple.py \
    --data_dir data/tiny-imagenet-200 \
    --batch_size 64 \
    --epochs 10 \
    --lr 0.001 \
    --num_workers 0 \
    --checkpoint_dir checkpoints_simple \
    "$@"
