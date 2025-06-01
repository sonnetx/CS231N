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

# ---- 1.  Load the Python module that actually exists on Sherlock ----------
module load python/3.9.0        || { echo "❌ Could not load python/3.9.0"; exit 1; }

# (Optional) GPU libraries that the cluster provides
module load cuda/12.2 cudnn/9    # adjust if versions differ

# Capture the module’s interpreter so we don’t accidentally call /usr/bin/python
PY=python3                       # everything below uses this variable

###############################################################################
# Virtual-env creation + dependencies
###############################################################################
$PY -m venv venv                 # <-- now runs the module’s python3
source venv/bin/activate
pip install --upgrade pip

# Remove any hard-pinned cuDNN wheel; cluster libs already supply it
grep -v '^nvidia-cudnn-cu12' requirements.txt | \
    pip install --no-cache-dir -r /dev/stdin

###############################################################################
# Launch your script
###############################################################################
$PY src/models/model_comparison_baseline.py
