#!/usr/bin/env bash
###############################################################################
# Slurm directives
###############################################################################
#SBATCH --job-name=231n_job
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:59:00
#SBATCH --mem=128G
#SBATCH --output=job_output_%j.out
#SBATCH --chdir=/home/groups/roxanad/eric/CS231N   # repo root

###############################################################################
# Environment setup
###############################################################################
set -euo pipefail

# ---- 1.  Load the Python version that actually exists on Sherlock ----
PYTHON_MOD=python/3.9.0          # ← from `module spider python`
module load "${PYTHON_MOD}"  || { echo "❌ Could not load ${PYTHON_MOD}"; exit 1; }

# ---- 2.  CUDA & cuDNN modules supplied by the cluster ---------------
module load cuda/12.2 cudnn/9    # adjust if Sherlock uses different names

###############################################################################
# Virtual-env creation + dependencies
###############################################################################
python -m venv venv
source venv/bin/activate
pip install --upgrade pip

# If your requirements pin cuDNN, drop that line because the module already
# provides the libraries; leaving it in can break the install.
grep -v '^nvidia-cudnn-cu12' requirements.txt | \
    pip install --no-cache-dir -r /dev/stdin

###############################################################################
# Launch training / evaluation
###############################################################################
python src/models/model_comparison_baseline.py
