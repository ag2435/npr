#!/bin/bash
#SBATCH -J npr_susy                         # Job name
#SBATCH -o slurm/npr_susy_%j.out                  # output file (%j expands to jobID)
#SBATCH -e slurm/npr_susy_%j.err                  # error log file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 2                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=32000                           # server memory (MBs) requested (per node)
#SBATCH -t 8:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=default_partition       # Request partition
#SBATCH --constraint=sr                     #request specific CPU
./real.sh output_rebuttal-test