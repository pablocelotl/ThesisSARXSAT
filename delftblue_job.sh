#!/bin/bash
# DelftBlue batch job for run_identify.py. Submit it from the run folder in /scratch:
#   cd /scratch/$USER/runs/<run-name> && sbatch ~/ThesisSARXSAT/delftblue_job.sh

#SBATCH --job-name=hysisat
#SBATCH --partition=compute-p1
#SBATCH --account=education-me-msc-sc
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB

source ~/miniconda3/bin/activate hysisat
export PYTHONPATH=$HOME/ThesisSARXSAT

srun python $HOME/ThesisSARXSAT/run_identify.py \
    --input Data/inputHVAC_first_1000.txt --output Data/outputHVAC_first_1000.txt \
    --nu 1 --ny 1 --dw 15 --delta 0.8 --blocks 2 > run.log 2>&1
