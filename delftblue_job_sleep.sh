#!/bin/bash
# DelftBlue job for the downsampled grouped hippocampal LFP (hpc_g3, made with lfp_groups.py and
# downsample_dataset.py), to look for awake / REM / NREM modes with a 60 s dwell time.
# Submit it from a run folder in /scratch that contains Data/<NAME>_output.txt and
# Data/<NAME>_input_dummy.txt. DW is in samples, so 60 s is 60 at 1 Hz and 15000 at 250 Hz:
#   sbatch --export=ALL,NAME=hpc_g3_1Hz,NY=1,DW=60,DELTA=2.0 --cpus-per-task=1 --time=04:00:00 \
#       ~/ThesisSARXSAT/delftblue_job_sleep.sh
#   sbatch --export=ALL,NAME=hpc_g3_250Hz,NY=4,DW=15000,DELTA=1.5 ~/ThesisSARXSAT/delftblue_job_sleep.sh

#SBATCH --job-name=hysisat_sleep
#SBATCH --partition=compute-p1
#SBATCH --account=education-me-msc-sc
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900MB

NAME=${NAME:-hpc_g3_250Hz}
NY=${NY:-4}
DW=${DW:-15000}
DELTA=${DELTA:-1.5}
BLOCKS=${BLOCKS:-1}

source ~/miniconda3/bin/activate hysisat
export PYTHONPATH=$HOME/ThesisSARXSAT

echo "NAME=$NAME NY=$NY DW=$DW DELTA=$DELTA BLOCKS=$BLOCKS" > settings.txt
srun python $HOME/ThesisSARXSAT/run_identify.py \
    --input Data/${NAME}_input_dummy.txt --output Data/${NAME}_output.txt \
    --nu 1 --ny $NY --dw $DW --delta $DELTA --blocks $BLOCKS > run.log 2>&1
