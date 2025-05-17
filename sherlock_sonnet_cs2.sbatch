#!/bin/bash

#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output={{OUTPUT_FILE}}
#SBATCH --error={{ERROR_FILE}}
#SBATCH --time={{TIME_LIMIT}}
#SBATCH --gres=gpu:{{GPUS}}
#SBATCH --partition={{PARTITION}}

module load python/3.8
# source activate your_env_here
python train_model.py
