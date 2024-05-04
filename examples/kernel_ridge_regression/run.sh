#!/bin/bash
conda init bash
conda activate krr2
papermill kernel_ridge_regression_housing.ipynb kernel_ridge_regression_housing_1.ipynb -p n_jobs 2 -p n_repeats 100 -p alpha 0.001 -p save True
