# Non-Parametric Regression

## Setup

Python version: 3.10

NOTE: for some some reason, sklearn grid search with n_jobs>1 doesn't work with python versions >3.10

Goodpoints package:
```
git clone https://github.com/ag2435/goodpoints
cd goodpoints
pip install -e .
```

## Estimators

Nadaraya-Watson

Kernel ridge regression

## Kernels

(Implemented in `goodpoints` repo, see above)

Gaussian

TODO:
- Laplace
- Boxcar
- Epanechnikov