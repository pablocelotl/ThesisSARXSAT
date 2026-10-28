#!/bin/bash
# DelftBlue job for the combined cortex boxes + hippocampal LFP dataset (1 kHz, z-scored),
# made with make_combined_dataset.py. Submit it from the run folder in /scratch:
#   cd /scratch/$USER/runs/<run-name> && sbatch ~/ThesisSARXSAT/delftblue_job_combined.sh
# DELTA, DW and BLOCKS can be overridden per submission, e.g. for runs side by side:
#   sbatch --export=ALL,DELTA=0.5 ~/ThesisSARXSAT/delftblue_job_combined.sh

#SBATCH --job-name=hysisat_comb
#SBATCH --partition=compute-p1
#SBATCH --account=education-me-msc-sc
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900MB

DELTA=${DELTA:-0.8}
DW=${DW:-100}
BLOCKS=${BLOCKS:-2}

source ~/miniconda3/bin/activate hysisat
export PYTHONPATH=$HOME/ThesisSARXSAT

srun python $HOME/ThesisSARXSAT/run_identify.py \
    --input Data/in.txt --output Data/out.txt \
    --nu 1 --ny 1 --dw $DW --delta $DELTA --blocks $BLOCKS > run.log 2>&1
