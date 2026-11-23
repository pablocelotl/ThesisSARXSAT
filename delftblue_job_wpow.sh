#!/bin/bash
# DelftBlue job for the wavelet band-power dataset of the NREM bout (comb3x2_g3_nrem_wpow: 5 cortex boxes x 3 bands
# + 3 hippocampal groups x 7 bands = 36 features at 20 Hz, made with frequency_analysis/wavelet_power.py).
# Submit it from a run folder in /scratch that contains Data/<NAME>_output.txt and Data/<NAME>_input_dummy.txt.
# DW is in samples at 20 Hz, so 150 = 7.5 s. Calibrate on the first 60 s, then the full bout:
#   sbatch --export=ALL,NAME=comb3x2_g3_nrem_wpow_cal60s --time=04:00:00 ~/ThesisSARXSAT/delftblue_job_wpow.sh
#   sbatch --export=ALL,DELTA=2.5 ~/ThesisSARXSAT/delftblue_job_wpow.sh

#SBATCH --job-name=hysisat_wpow
#SBATCH --partition=compute-p1
#SBATCH --account=education-me-msc-ro
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900MB

NAME=${NAME:-comb3x2_g3_nrem_wpow}
NY=${NY:-1}
DW=${DW:-150}
DELTA=${DELTA:-2.0}
BLOCKS=${BLOCKS:-1}

source ~/miniconda3/bin/activate hysisat
export PYTHONPATH=$HOME/ThesisSARXSAT

echo "NAME=$NAME NY=$NY DW=$DW DELTA=$DELTA BLOCKS=$BLOCKS" > settings.txt
srun python $HOME/ThesisSARXSAT/run_identify.py \
    --input Data/${NAME}_input_dummy.txt --output Data/${NAME}_output.txt \
    --nu 1 --ny $NY --dw $DW --delta $DELTA --blocks $BLOCKS > run.log 2>&1
