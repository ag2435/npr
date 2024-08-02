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
#     display_name: npr
#     language: python
#     name: npr
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
# package imports
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import os
from time import time
from numpy.linalg import LinAlgError
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV

# utils for nadaraya watson estimators
from npr import estimator_factory
# utils for generate samples from the data distribution
from npr.util_sample import ToyData
# utils for experiments
from helpers import get_base_parser, parse_param_str

# %%
parser = get_base_parser()
parser.add_argument("--ground_truth", "-gt", default='sum_gauss', type=str,
                    help="ground-truth regression function", 
                    choices=['sum_gauss', 'sum_laplace', 'sin', 'tanh', 'sinexp'])
parser.add_argument("--logn_lo", "-lo", default=8, type=int,
                    help="minimum log2(n) (inclusive)")
parser.add_argument("--logn_hi", "-hi", default=14, type=int,
                    help="maximum log2(n) (inclusive)")
parser.add_argument('--dimension', '-d', default=1, type=int,
                    help="X dimension of toy dataset")
parser.add_argument("--seed", "-s", default=123, type=int,
                    help="seed for random number generator when generating synthetic data")

# %%
args = parser.parse_args()
method = args.method
thin = args.thin
kernel = args.kernel
sigma = parse_param_str(args.sigma)
alpha = parse_param_str(args.alpha)
ground_truth = args.ground_truth
task = 'regression'
refit = 'neg_mean_squared_error'
logn_lo = 2* (args.logn_lo // 2) # make sure n is a power of 4
logn_hi = 2* (args.logn_hi // 2) # make sure n is a power of 4
n_trials = args.n_trials
n_jobs = args.n_jobs
output_path = args.output_path
seed = args.seed
ablation = args.ablation
no_swap = args.no_swap
d = args.dimension

# set boolean flag for using cross validation
use_sigma_cv = not isinstance(sigma, float) 
use_alpha_cv = not isinstance(alpha, float)
use_cross_validation = use_sigma_cv or use_alpha_cv

# %%
# Output path
os.makedirs(output_path, exist_ok=True)
# path to save scores
score_dir = os.path.join(
    output_path,
    f'd={d}',
    'scores_cv' if use_cross_validation else 'scores',
)
os.makedirs(score_dir, exist_ok=True)
ablation_str = f"-ablation{ablation}" if ablation > 0 else ""
score_file = os.path.join(
    score_dir,
    f"{thin}-{method}-k={kernel}-gt={ground_truth}-logn={logn_lo}_{logn_hi}-t{n_trials}{ablation_str}.pkl"
)
# path to save figures
figure_dir = os.path.join(output_path,f'd={d}','figures')
os.makedirs(figure_dir, exist_ok=True)

# Get saved filed if possible
if os.path.exists(score_file) and not args.force:
    print(f"Loading from {score_file}")
    with open(score_file, 'rb') as f:
        results = pickle.load(f)
    df = pd.DataFrame(results, columns=['logn', '_mean_score', '_mean_train_time', '_mean_test_time'])
    print(df)
    raise SystemExit

# %%
# sample synthetic data
X_name = 'unif'
toy_data_noise = ToyData(
    X_name=X_name, 
    f_name=ground_truth,
    noise=1.,
    d=d, 
    # k=1, #number of anchor points
    # k=8,
    # scale=8,
)

# X_train, y_train = toy_data_noise.sample(n)
X_test, y_test = toy_data_noise.sample(10000, seed=seed, shuffle=False)
print(X_test.shape, y_test.shape)
# validation set used for cross validation
# set different seed so that val and test data and different
X_val, y_val = toy_data_noise.sample(10000, seed=seed*2, shuffle=True)
print(X_val.shape, y_val.shape)

# %%
if d == 1:
    fig = px.scatter(x=X_val[:,0], y=y_val,
                title=f'X={X_name}, f={ground_truth}, std[f]={np.std(y_val):.4f}',)
else:
    fig = px.scatter_3d(x=X_val[:,0], y=X_val[:,1], z=y_val, opacity=0.5,)
# save fig as image
fig.write_image(os.path.join(figure_dir, f"X={X_name}_f={ground_truth}.png"))

# %%
param_grid = {}
if use_cross_validation:
    assert d in [1, 4, 8], f"dimension {d} not supported for cross validation"
    if use_sigma_cv: param_grid['sigma'] = sigma
    if use_alpha_cv: param_grid['alpha'] = alpha
    
    # if d == 1:
    #     grid_configs = {
    #     'full': grid(0.05, 0.2, 0.05),# / 5,
    #         'st': grid(0.1, 1.0, 0.1),
    #         'kt': grid(0.1, 1.0, 0.1),
    #         'rpcholesky': grid(0.1, 1.0, 0.1),
    #     }
    # elif d==4:
    #     delta = 0.1
    #     param_grid = {
    #         "sigma": grid(delta, 2., delta),
    #     }
    # elif d == 8:
    #     delta = 0.1
    #     param_grid = {
    #         "sigma": grid(delta, 2., delta),
    #     }
    # else:
    #     raise ValueError(f"experiment with dimension {d} not tested")
print('param grid', param_grid)

# %%
# Run experiment (depending on experiment_type)

results = []

def plot_design_points(logn, model, X, y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=model.X_fit_.flatten(), 
        y=model.y_fit_,
        mode='markers',
        opacity=0.5,
        marker=dict(
            color='red',
        )
    ))
    fig.add_trace(go.Scatter(
        x=X.flatten(), 
        y=y,
        mode='markers',
        opacity=0.1,
        marker=dict(
            color='blue',
        )
    ))
    path = os.path.join(
        figure_dir, 
        f"{thin}-{method}-k={kernel}-gt={ground_truth}-logn={logn}{ablation_str}.png"
    )
    # print(f"Saving design points figure to {path}")
    fig.write_image(path)

# only even logn (i.e., n is a power of 4)
for logn in range(logn_lo, logn_hi+1, 2):
    trials = n_trials # (1 if method in ['full'] else n_trials)
    if thin in ['full',] and trials > 1:
        print(f"Warning: trials={trials} for method {method} even though it's deterministic.")

    # get data
    X, y = toy_data_noise.sample(2**logn, seed=seed, shuffle=False)
    # print('debug: X>', X[0], 'Y>', y[0])

    # get estimator
    # keep kernel fixed
    model = estimator_factory(task, method, thin, 
                              kernel=kernel,
                              ablation=ablation, no_swap=no_swap)
    
    if model is None:
        print(f"Skipping {thin}-{method} with {kernel} kernel")
        continue
    print(f'logn={logn}: model={model}')

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

    print(f"best params: {best_params}")
    best_model = estimator_factory(
        task, 
        method=method,
        thin=thin,
        kernel=kernel,
        sigma=best_params['sigma'] if use_sigma_cv else sigma,
        alpha=best_params['alpha'] if use_alpha_cv else alpha,
        ablation=ablation,
        no_swap=no_swap,
    )

    scores = []
    train_times = []
    test_times = []
    pbar = tqdm(range(trials))
    for i in pbar:
        # training
        try:
            start = time()
            best_model.fit(X, y)
            train_time = time() - start
        except LinAlgError:
            # print(f"LinAlgError: {e}")
            continue
            # Note: sklearn automatically prints out a summary of errors during cross-validation
            # including the number of fits that failed and the error type.

        # plot design points
        if i == 0: plot_design_points(logn, best_model, X, y)

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
        # "_mean_score": f"{np.mean(scores):.4f} ± {np.std(scores):.4f}",
        # "_mean_train_time": f"{np.mean(train_times):.4f} ± {np.std(train_times):.4f}",
        # "_mean_test_time": f"{np.mean(test_times):.4f} ± {np.std(test_times):.4f}",
        "_mean_score": f"{np.mean(scores):.8f} ± {np.std(scores):.8f}",
        "_mean_train_time": f"{np.mean(train_times):.8f} ± {np.std(train_times):.8f}",
        "_mean_test_time": f"{np.mean(test_times):.8f} ± {np.std(test_times):.8f}",
    })

# %%
# save to output path
df = pd.DataFrame(results, columns=['logn', '_mean_score', '_mean_train_time', '_mean_test_time'])
print(f"Saving to {score_file}:\n{df}")
with open(score_file, 'wb') as f:
    pickle.dump(results, f)

# %%
