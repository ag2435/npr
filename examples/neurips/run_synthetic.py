# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: goodpoints
#     language: python
#     name: python3
# ---

# %% [markdown]
# Run Nadaraya-Watson experiments
#
# ```
# python run_synthetic.py -m nw -thin full
# ```
#
# Timing: https://openml.github.io/openml-python/develop/examples/30_extended/fetch_runtimes_tutorial.html

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--method", "-m", default='nw', type=str,
                    help="non-parametric regression method", choices=['nw', 'krr', 'rpcholesky'])
parser.add_argument("--thin", "-thin", default='full', type=str,
                    help="thinning method", choices=['full', 'st', 'kt', 'rpcholesky'])
parser.add_argument("--kernel", "-k", default='epanechnikov', type=str,
                    help="kernel function", choices=['epanechnikov', 'gaussian', 'laplace'])
parser.add_argument("--sigma", "-sig", default=0.1, type=float,
                    help="bandwidth for kernel")
parser.add_argument("--alpha", "-alpha", default=1e-3, type=float,
                    help="regularization parameter for kernel ridge regression")
parser.add_argument("--ground_truth", "-gt", default='sum_gauss', type=str,
                    help="ground-truth regression function", choices=['sum_gauss', 'sum_laplace'])
parser.add_argument("--logn_lo", "-lo", default=8, type=int,
                    help="minimum log2(n) (inclusive)")
parser.add_argument("--logn_hi", "-hi", default=14, type=int,
                    help="maximum log2(n) (inclusive)")
parser.add_argument("--n_trials", "-t", default=100, type=int,
                    help="number of trials to run each setting")
# parser.add_argument("--use_crossval", "-cv", action='store_true',
#                     help="if set, use cross-validation")
parser.add_argument("--output_path", "-op", default='output', type=str,
                    help="directory for storing output")
parser.add_argument("--seed", "-s", default=123, type=int,
                    help="seed for random number generator when generating synthetic data")
parser.add_argument("--ablation", default=0, type=int,
                    help="kernel ablation study", choices=[0, 1, 2])

# %%
args, opt = parser.parse_known_args()
method = args.method
thin = args.thin
kernel = args.kernel
sigma = args.sigma
alpha = args.alpha
ground_truth = args.ground_truth
task = 'regression'
refit = 'neg_mean_squared_error'
logn_lo = 2* (args.logn_lo // 2) # make sure n is a power of 4
logn_hi = 2* (args.logn_hi // 2) # make sure n is a power of 4
n_trials = args.n_trials
n_jobs = 1
use_cross_validation = False # args.use_crossval
output_path = args.output_path
seed = args.seed
ablation = args.ablation

# %%
# remaining imports
import pandas as pd
# from copy import deepcopy
# from joblib import Parallel, delayed
import pickle
import numpy as np
from tqdm import tqdm
import os
from time import time
from numpy.linalg import LinAlgError

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV

# utils for nadaraya watson estimators
from npr import estimator_factory
# utils for generate samples from the data distribution
from npr.util_sample import ToyData

# %%
# sample synthetic data
toy_data_noise = ToyData(
    X_name='unif', 
    f_name=ground_truth,
    noise=0.1,
    d=1, 
    k=8, #number of anchor points
)

# X_train, y_train = toy_data_noise.sample(n)
X_test, y_test = toy_data_noise.sample(10000, seed=seed, shuffle=False)
# print('debug: X_test>', X_test[0], 'y_test>', y_test[0])
# validation set used for cross validation
# set different seed so that val and test data and different
X_val, y_val = toy_data_noise.sample(10000, seed=seed*2, shuffle=True)

# %%
param_grid = {
    "sigma": np.logspace(-4, 0, 5),
}

# %%
# Run experiment (depending on experiment_type)

results = []

i = 0
# only even logn (i.e., n is a power of 4)
for logn in range(logn_lo, logn_hi+1, 2):
    trials = n_trials # (1 if method in ['full'] else n_trials)

    # get data
    X, y = toy_data_noise.sample(2**logn, seed=seed, shuffle=False)
    # print('debug: X>', X[0], 'Y>', y[0])

    # get estimator
    # keep kernel fixed
    model = estimator_factory(task, method, thin, 
                              kernel=kernel,
                              ablation=ablation,)
    
    if model is None:
        print(f"Skipping {thin}-{method} with {kernel} kernel")
        continue
    print(f'i={i+1}: logn={logn}, model={model}')

    # STEP 2: Get optimal parameters through grid search
    # NOTE: we do something slightly better than k-fold cross validation.
    # Namely, we are trying to get rid of randomness in the Kernel Thinning (or Standard Thinning) routine,
    # but if we did 100-fold CV, then the validation set would be 1% of the data
    # (which is too small to get a good estimate of the validation score).
    # Instead we use the same train-val split for each parameter setting and repeat `trials` times
    if use_cross_validation:
        X_concat, y_concat = np.concatenate([X, X_val]), np.concatenate([y, y_val])
        split = [(np.arange(len(X)), np.arange(len(X), len(X)+len(X_val))) for _ in range(trials)]
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            return_train_score=True,
            cv=split,
            scoring=refit,
            refit=False,
            n_jobs=n_jobs,
        ).fit(X_concat, y_concat)
        # get validation scores
        cv_results = pd.DataFrame(grid_search.cv_results_)
        val_scores = []
        for i in range(trials):
            val_scores.append( cv_results.iloc[grid_search.best_index_][f'split{i}_test_score'] )

        # get optimal parameters
        best_params = grid_search.best_params_
    else:
        # Dummy values
        val_scores = [1,] * trials
        
        best_params = {
            'sigma' : sigma,
            'alpha' : alpha,
        }

    # print(f"best params: {best_params}")
    best_model = estimator_factory(
        task, 
        method=method,
        thin=thin,
        kernel=kernel,
        sigma=best_params['sigma'],
        alpha=best_params['alpha'],
        ablation=ablation,
    )

    scores = []
    train_times = []
    test_times = []
    pbar = tqdm(range(trials))
    for _ in pbar:
        # training
        try:
            start = time()
            best_model.fit(X, y)
            train_time = time() - start
        except LinAlgError:
            # print(f"LinAlgError: {e}")
            continue

        # testing
        start = time()
        test_pred = best_model.predict(X_test).squeeze()
        test_time = time() - start

        if refit == 'neg_mean_squared_error':
            test_scores = mean_squared_error(y_test, test_pred)
        elif refit == 'accuracy':
            test_scores = accuracy_score(y_test, test_pred)
        else:
            raise ValueError(f"invalid refit metric: {refit}")

        scores.append( np.mean(test_scores) )
        train_times.append(train_time)
        test_times.append(test_time)
        pbar.set_description(f"test score: {np.mean(scores):.4f}")

    results.append({
        "logn": logn, 
        # "model": model_name, 
        "cv_results": pd.DataFrame(grid_search.cv_results_) if use_cross_validation else None,
        "best_index_" : grid_search.best_index_ if use_cross_validation else 0,
        "scores" : scores,
        "train_times" : train_times,
        "test_times" : test_times,
        # for printing
        "_mean_score": f"{np.mean(scores):.4f} ± {np.std(scores):.4f}",
        "_mean_train_time": f"{np.mean(train_times):.4f} ± {np.std(train_times):.4f}",
        "_mean_test_time": f"{np.mean(test_times):.4f} ± {np.std(test_times):.4f}",
    })

    i += 1

# %%
# save to output path
score_dir = os.path.join(
    output_path,
    'scores',
)
os.makedirs(score_dir, exist_ok=True)
ablation_str = f"-ablation{ablation}" if ablation > 0 else ""
score_file = os.path.join(
    score_dir,
    f"{thin}-{method}-k={kernel}-gt={ground_truth}-logn={logn_lo}_{logn_hi}-t{n_trials}{ablation_str}.pkl"
)
df = pd.DataFrame(results, columns=['logn', '_mean_score', '_mean_train_time', '_mean_test_time'])
print(f"Saving to {score_file}:\n{df}")
with open(score_file, 'wb') as f:
    pickle.dump(results, f)

# %%
