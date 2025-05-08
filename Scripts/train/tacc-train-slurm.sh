#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Lonestar6 AMD Milan nodes
#
#   *** Serial Job in Normal Queue***
# 
# Last revised: October 22, 2021
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch milan.serial.slurm" on a Lonestar6 login node.
#
#  -- Serial codes run on a single node (upper case N = 1).
#       A serial code ignores the value of lower case n,
#       but slurm needs a plausible value to schedule the job.
#
#  -- Use TACC's pylauncher utility to run multiple serial 
#       executables at the same time, execute "module load pylauncher" 
#       followed by "module help pylauncher".
#----------------------------------------------------

#SBATCH -J plb-train           # Job name
#SBATCH -o slurm/%j.out       # Name of stdout output file
#SBATCH -e slurm/%j.err       # Name of stderr error file
#SBATCH -p gpu-a100-small  # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 10:00:00        # Run time (hh:mm:ss)s
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A DBS24006 # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=sagnikraychoudhury@gmail.com

# Any other commands must follow all #SBATCH directives...
module load cuda/12.2
source ~/.bashrc

export CUDA_HOME=/opt/apps/cuda/12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo $1, $2
${WORK}/anaconda3/envs/polbias/bin/python finetuning-LLM-classification.py --model_n $1 --dataset_n $2
# Launch serial code...
#./myprogram         # Do not use ibrun or any other MPI launcher
