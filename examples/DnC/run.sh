#!/bin/bash
conda init bash
conda activate krr2
# papermill dnc_paper.ipynb dnc_paper_sobolev_$1.ipynb -p n_jobs 8 -p n_repeats 100 -p save True -p kernel sobolev -p X_name unif-[0,1] -p f_name dnc-paper
papermill dnc_paper.ipynb dnc_paper_gauss_$1.ipynb -p n_jobs 8 -p n_repeats 100 -p save True -p kernel gauss -p X_name unif -p f_name sum-gauss -p alpha_pow -1
