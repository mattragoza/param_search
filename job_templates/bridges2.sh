#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -A bio170034p
#SBATCH --partition=BatComputer
#SBATCH --gres=gpu:1
#SBATCH --dependency=singleton
#SBATCH --mail-type=fail
#SBATCH --mail-user={email}
#SBATCH -o %J.stdout
#SBATCH -e %J.stderr
{job_params}
hostname
nvidia-smi
source activate {conda_env}
echo hello, world

