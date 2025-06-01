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

set -euo pipefail

###############################################################################
# 1.  Load modules
###############################################################################
module purge                                     # start clean
module load python/3.9.0       || { echo "❌ cannot load python/3.9.0"; exit 1; }
module load cuda/12.2 cudnn/9                    # adjust to Sherlock’s names

PY=$(which python3)                              # path to module’s interpreter
echo "Using $PY"

###############################################################################
# 2.  Create venv *without* ensurepip, then add pip manually
###############################################################################
$PY -m venv --without-pip venv
source venv/bin/activate

# tiny bootstrap script (~100 kB) → installs pip & setuptools inside venv
curl -sS https://bootstrap.pypa.io/get-pip.py | python -

###############################################################################
# 3.  Install project requirements
###############################################################################
pip install --upgrade pip

# If requirements.txt hard-pins cuDNN wheels, drop that line (cluster libs already loaded)
grep -v '^nvidia-cudnn-cu12' requirements.txt | \
    pip install --no-cache-dir -r /dev/stdin

###############################################################################
# 4.  Run your code
###############################################################################
python src/models/model_comparison_baseline.py
