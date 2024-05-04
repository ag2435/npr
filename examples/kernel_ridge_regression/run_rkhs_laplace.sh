#!/bin/bash
conda init bash
conda activate krr2
papermill kernel_ridge_regression_rkhs.ipynb kernel_ridge_regression_rkhs_laplace_$1.ipynb -p n_jobs 4 -p k $1 -p use_cross_validation True -p f_name sum-laplace -p kernel laplace
