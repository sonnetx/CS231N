#!/bin/bash

#SBATCH --job-name=231n_job
#SBATCH --time=2-23:59:00
#SBATCH --output=job_output_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p roxanad
#SBATCH --mem=128G


# Loading python version
ml python/3.9.0

# Creating venv and installing requirements
python3 -m venv venv      
source venv/bin/activate
pip install -r /home/groups/roxanad/eric/CS231N/requirements.txt

# Running the script
python3 src/models/model_comparison_baseline.py