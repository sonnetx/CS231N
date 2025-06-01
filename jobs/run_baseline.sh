#!/usr/bin/env bash
#SBATCH --job-name=231n_job
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:59:00
#SBATCH --mem=128G
#SBATCH --output=job_output_%j.out
#SBATCH --chdir=/home/groups/roxanad/eric/CS231N

set -euo pipefail

# ------------------------------------------------------------------
# 1.  Modules
# ------------------------------------------------------------------
module load python/3.9.0              || { echo "❌ cannot load python/3.9.0"; exit 1; }
module load cuda/12.2 cudnn/9         # adjust if Sherlock uses other names

PY=python3                            # interpreter from the module

# ------------------------------------------------------------------
# 2.  Virtual-env  (skip ensurepip, add pip manually)
# ------------------------------------------------------------------
$PY -m venv --without-pip venv        # <-- no ensurepip call
source venv/bin/activate

# Bootstrap pip (10 KB download)
curl -sS https://bootstrap.pypa.io/get-pip.py | python -

# ------------------------------------------------------------------
# 3.  Dependencies
# ------------------------------------------------------------------
pip install --upgrade pip
grep -v '^nvidia-cudnn-cu12' requirements.txt | \
    pip install --no-cache-dir -r /dev/stdin

# ------------------------------------------------------------------
# 4.  Run
# ------------------------------------------------------------------
$PY src/models/model_comparison_baseline.py
