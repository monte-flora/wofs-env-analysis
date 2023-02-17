#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --job-name="Step1_2013"
#SBATCH --mail-user=randychase@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out
#SBATCH --error=/home/randychase/slurmouts/R-%x.%j.err
#SBATCH --array=2%4

#source  my python envs
source ~/.bashrc
bash 

#activate your python environment 
conda activate python_env_name 
 
#QC and convert the sparse gridrad data to grids 
python name_of_your_python_script