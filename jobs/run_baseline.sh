#!/usr/bin/env bash
###############################################################################
# Slurm directives
###############################################################################
#SBATCH --job-name=231n_job
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:59:00            # 2 days + 23:59 hrs
#SBATCH --mem=128G
#SBATCH --output=job_output_%j.out
#SBATCH --chdir=/home/groups/roxanad/eric/CS231N   # run from repo root

###############################################################################
# Environment setup
###############################################################################
set -euo pipefail                            # stop on first error

# 1) Use Python 3.11 (wheels for numpy, PyTorch/TensorFlow, etc. exist)
module load python/3.11

# 2) (Optional) Load cluster-provided CUDA/cuDNN so you don’t need to
#    pin nvidia-cudnn-cu12 in requirements.txt.  Comment out if not needed.
module load cuda/12.2 cudnn/9

###############################################################################
# Virtual-env creation + deps
###############################################################################
python -m venv venv
source venv/bin/activate

pip install --upgrade pip

# If your requirements.txt hard-pins “nvidia-cudnn-cu12==…”, filter it out
# because the cluster module already supplies cuDNN.
grep -v '^nvidia-cudnn-cu12' requirements.txt | pip install --no-cache-dir -r /dev/stdin

###############################################################################
# Launch your training / evaluation script
###############################################################################
python src/models/model_comparison_baseline.py
