#!/bin/bash -l
## Example run script for SLURM


## General configuration options
#SBATCH -J Manu_LandLab
#SBATCH -e LandLab_e%j
#SBATCH -o LandLab_o%j
#SBATCH --mail-user=manuel.schmid@uni-tuebingen.de
#SBATCH --mail-type=ALL


## Machine and CPU configuration
## Number of tasks per job:
#SBATCH -n 8
## Number of nodes:
#SBATCH -N 1
## Run only on this node(s):
#SBATCH -w u-005-s039

python my_script.py
