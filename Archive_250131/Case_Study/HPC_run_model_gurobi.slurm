#!/bin/sh
#SBATCH --qos=normal
#SBATCH --account=markers
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --job-name=hypstat_case_study
#SBATCH --mail-user=yijin.li@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A markers
#SBATCH -o hypstat_case_study.log

module purge
module load anaconda3
module load gurobi
conda activate /kfs2/projects/h2nyserda/envs/pyomo_env

echo 'running HYPSTAT open sourcing: Pyomo function for linear optimization'

python HPC_run_model_case_study.py
