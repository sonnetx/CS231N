#!/bin/bash
#SBATCH --job-name=img_probe
#SBATCH --output=logs/img_probe_%j.log
#SBATCH --error=logs/img_probe_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH -p roxanad


# Loading python version
ml python/3.12

# Creating venv and installing requirements
python3 -m venv venv      
source venv/bin/activate
pip install -r /home/groups/roxanad/eric/CS231N/requirements.txt

# Running the script
python3 src/evaluation/img_probing.py