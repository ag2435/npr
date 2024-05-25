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
# To run:
# ```
# python run_housing.py -m krr -thin full
# ```

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--method", "-m", default='krr', type=str,
                    help="non-parametric regression method", choices=['nw', 'krr'])
parser.add_argument("--thin", "-thin", default='full', type=str,
                    help="thinning method", choices=['full', 'st', 'kt', 'rpcholesky'])
parser.add_argument("--kernel", "-k", default='gaussian', type=str,
                    help="kernel function", choices=['epanechnikov', 'gaussian', 'laplace'])
parser.add_argument("--sigma", "-sig", default=10, type=float,
                    help="bandwidth for kernel")
parser.add_argument("--alpha", "-alpha", default=1e-3, type=float,
                    help="regularization parameter for kernel ridge regression")
parser.add_argument("--n_trials", "-t", default=100, type=int,
                    help="number of trials to run each setting")
# parser.add_argument("--use_crossval", "-cv", action='store_true',
#                     help="if set, use cross-validation")
parser.add_argument("--output_path", "-op", default='output', type=str,
                    help="directory for storing output")
parser.add_argument("--seed", "-s", default=123, type=int,
                    help="seed for random number generator when generating synthetic data")
parser.add_argument('--ablation', default=0, type=int,
                    help="kernel ablation study", choices=[0, 1, 2, 3])

# %%
args, opt = parser.parse_known_args()
dataset = 'housing'
k_fold = 5
method = args.method
thin = args.thin
kernel = args.kernel
sigma = args.sigma
alpha = args.alpha
task = 'regression'
refit = 'neg_mean_squared_error'
n_trials = args.n_trials
n_jobs = 1
use_cross_validation = False # args.use_crossval
output_path = args.output_path
seed = args.seed
ablation = args.ablation

# %%
# remaining imports
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import os
from time import time

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, KFold

# utils for nadaraya watson estimators
from npr import estimator_factory
# utils for generate samples from the data distribution
from npr.util_load_data import get_real_dataset

# %%
print(f"Loading {dataset} dataset")
X, y = get_real_dataset(dataset)
# print(X.shape, y.shape)
# remove values corresponding to y>= 5
X = X[y < 5]
y = y[y < 5]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=(k_fold-1)/k_fold, 
                                                    shuffle=True, random_state=seed)

# %%
param_grid = {
    "sigma" :   [1,2,5,10,20], 
    "alpha" :   [1e-3,1e-4,1e-5],
}

# %%
results = []

# NOTE: full and rfm are deterministic, so we only need to run them once
trials = (1 if thin in ['full', 'rfm'] else n_trials)

# STEP 1: Get data
# use X_train, y_train, X_test, y_test from above
model = estimator_factory(task, method, thin,
                          kernel=kernel,
                          ablation=ablation,)

# STEP 2: Get optimal parameters through grid search
# NOTE: we want to get rid of randomness in the Kernel Thinning (or Standard Thinning) routine
# so we do k-fold cross validation `trials` times using the *same* split.
# This is different from sklearn's repeated k-fold implementation which uses a 
# different random split each time.            
if use_cross_validation:
    split = list(KFold(n_splits=k_fold).split(X_train)) * trials
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        return_train_score=True,
        cv=split,
        scoring=refit,
        refit=False,
        n_jobs=n_jobs,
    ).fit(X_train, y_train)
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

# %%
print(f"best params: {best_params}")
best_model = estimator_factory(
    task, 
    method=method, 
    thin=thin,
    kernel=kernel,
    sigma=best_params['sigma'],
    alpha=best_params['alpha'],
    ablation=ablation,
)
print(best_model)

# %%
train_scores = []
test_scores = []
train_times = []
test_times = []

pbar = tqdm(range(trials))
for _ in pbar:
    # training
    start = time()
    best_model.fit(X_train, y_train)
    train_time = time() - start

    # testing
    # compute train score
    train_pred = best_model.predict(X_train).squeeze()
    # compute test score
    start = time()
    test_pred = best_model.predict(X_test).squeeze()
    test_time = time() - start

    if refit == 'neg_mean_squared_error':
        train_score = mean_squared_error(y_train, train_pred)
        test_score = mean_squared_error(y_test, test_pred)
    elif refit == 'accuracy':
        train_score = 1- accuracy_score(y_train, train_pred)
        test_score = 1- accuracy_score(y_test, test_pred)
    else:
        raise ValueError(f"invalid refit metric: {refit}")

    train_scores.append( train_score )
    test_scores.append( test_score )
    train_times.append(train_time)
    test_times.append(test_time)
    pbar.set_description(f"test score: {np.mean(test_scores):.4f}")

results.append({
    "dataset": dataset, 
    # "model": model_name, 
    "cv_results": pd.DataFrame(grid_search.cv_results_) if use_cross_validation else None,
    "best_index_" : grid_search.best_index_ if use_cross_validation else 0,
    "best_params_" : best_params,
    "val_scores" : val_scores,
    "train_scores" : train_scores,
    "test_scores" : test_scores,
    "train_times" : train_times,
    "test_times" : test_times,
    # for printing
    "_mean_train_score": f"{np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}",
    "_mean_test_score": f"{np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}",
    "_mean_train_time": f"{np.mean(train_times):.4f} ± {np.std(train_times):.4f}",
    "_mean_test_time": f"{np.mean(test_times):.4f} ± {np.std(test_times):.4f}",
})

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
    f"{thin}-{method}-k={kernel}-dataset={dataset}-t{n_trials}{ablation_str}.pkl"
)
df = pd.DataFrame(results, columns=['_mean_train_score', '_mean_test_score', 
                                    '_mean_train_time', '_mean_test_time'])
print(f"Saving to {score_file}:\n{df}")
with open(score_file, 'wb') as f:
    pickle.dump(results, f)

# %%
