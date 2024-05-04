#!/bin/bash
#SBATCH -J test_file                         # Job name
#SBATCH -o out/test_file_%j.out                  # output file (%j expands to jobID)
#SBATCH -e err/test_file_%j.err                  # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email 
#SBATCH --mail-user=ag2435@cornell.edu        # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 16                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=100000                           # server memory (MBs) requested (per node)
#SBATCH -t 8:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=default_partition       # Request partition
#SBATCH --constraint=sr                     #request specific CPU
/home/ag2435/goodpoints/examples/classification/run.sh
