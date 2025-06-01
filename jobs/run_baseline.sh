#!/usr/bin/env bash
#SBATCH --job-name=231n_job
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:59:00
#SBATCH --mem=128G
#SBATCH --output=job_output_%j.out
#SBATCH --chdir=/home/groups/roxanad/eric/CS231N

set -euo pipefail

###############################################################################
# 1.  Wipe the default modules and load ours
###############################################################################
module --force purge                     # unload devel, math, etc.

module load python/3.9.0                 || { echo "❌ cannot load python/3.9.0"; exit 1; }
module load cuda/12.2 cudnn/9            # adjust names/versions if needed

PY=$(which python3)                      # exact path to the 3.9 interpreter
echo "Using ${PY}"

###############################################################################
# 2.  Create venv *without* ensurepip and bootstrap pip by hand
###############################################################################
$PY -m venv --without-pip venv
source venv/bin/activate

curl -sS https://bootstrap.pypa.io/get-pip.py | python -

###############################################################################
# 3.  Install requirements
###############################################################################
pip install --upgrade pip
grep -v '^nvidia-cudnn-cu12' requirements.txt | \
    pip install --no-cache-dir -r /dev/stdin

###############################################################################
# 4.  Run your script
###############################################################################
python src/models/model_comparison_baseline.py
