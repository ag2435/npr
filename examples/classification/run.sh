#!/bin/bash
conda init bash
conda activate krr2
papermill bench45.ipynb housing.ipynb -p n_jobs 16 -p n_repeats 40 -p save False -p use_cross_validation True
