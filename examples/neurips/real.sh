#!/bin/bash
# %%%%%%%% Submission Experiments %%%%%%%%
# 
# California Housing dataset (n=20,640, d=8)
# 
# python run_real.py -m krr -thin full -k gaussian -sig 10 -alpha 1e-3 -t 1 -op $1
# python run_real.py -m krr -thin st -k gaussian -sig 10 -alpha 1e-3 -op $1
# python run_real.py -m krr -thin kt -k gaussian -sig 10 -alpha 1e-3 -op $1
# python run_real.py -m krr -thin rpcholesky -k gaussian -sig 10 -alpha 1e-5 -op $1

# %%%%%%%% Rebuttal Experiments %%%%%%%%
# 
# Feature learning via Recursive Feature Machines on California Housing dataset (n=20,640, d=8)
# 
# TODO: warm up run so that we only time the KRR fitting part, not the feature matrix fitting part
# 1. Do we still need to do cross validation after the warm up run?
#   a. sigma = 10
#   b. alpha = 1e-3

# # do one trial for full since it's deterministic
# python run_real.py -m krr -thin full -k gaussian_M -sig 10 -alpha 1e-3 -t 1 -op $1
# python run_real.py -m krr -thin st -k gaussian_M -sig 10 -alpha 1e-3 -op $1
# python run_real.py -m krr -thin kt -k gaussian_M -sig 10 -alpha 1e-3 -op $1

# 
# SUSY dataset (n=5,000,000, d=18)
# 
# python run_real.py -d susy -m krr -thin full -k gaussian -sig 4 -alpha 1e-3 -t 1 -op $1
python run_real.py -d susy -m krr -thin st -k gaussian -sig 4 -alpha 1e-3 -op $1
# python run_real.py -d susy -m krr -thin kt -k gaussian -sig 4 -alpha 1e-3 -op $1
# python run_real.py -d susy -m krr -thin rpcholesky -k gaussian -sig 4 -alpha 1e-5 -op $1