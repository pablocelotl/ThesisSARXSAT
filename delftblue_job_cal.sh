#!/bin/bash
# DelftBlue calibration job: one block on a trimmed dataset (Data/in.txt, Data/out.txt),
# to measure time and memory before the full run. Submit it from the run folder in /scratch:
#   cd /scratch/$USER/runs/sleep8x5_cal1000 && sbatch ~/ThesisSARXSAT/delftblue_job_cal.sh
# Afterwards: sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS

#SBATCH --job-name=hysisat_cal
#SBATCH --partition=compute-p1
#SBATCH --account=education-me-msc-sc
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900MB

source ~/miniconda3/bin/activate hysisat
export PYTHONPATH=$HOME/ThesisSARXSAT

srun python $HOME/ThesisSARXSAT/run_identify.py \
    --input Data/in.txt --output Data/out.txt \
    --nu 1 --ny 1 --dw 15 --delta 0.1 --blocks 1 > run.log 2>&1
