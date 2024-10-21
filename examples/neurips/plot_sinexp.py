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

# %%
import numpy as np
# sample synthetic data
from npr.util_sample import ToyData
# plot using matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 24,
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'axes.linewidth': 0.8,
    'lines.linewidth': 3,
    'lines.markersize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# %%

X_name = 'unif'
ground_truth = 'sinexp'
d=1
toy_data_noise = ToyData(
    X_name=X_name, 
    f_name=ground_truth,
    noise=1.,
    d=d, 
)
toy_data_no_noise = ToyData(
    X_name=X_name, 
    f_name=ground_truth,
    noise=0,
    d=d, 
)

# %%
seed = 123
X_val, y_val = toy_data_noise.sample(1000, seed=seed*2, shuffle=True)
print(X_val.shape, y_val.shape)

# %%
# regression function
X, y = toy_data_no_noise.sample(1000, seed=seed*2, shuffle=False)
# sort X
idx = np.argsort(X[:,0])
X = X[idx]
y = y[idx]

# %%
fig, ax = plt.subplots()
# plot validation data
ax.scatter(X_val[:,0], y_val, alpha=0.1, color='green')
# plot regression line
ax.plot(X[:,0], y, alpha=1, color='green', 
# linewidth
linewidth=3)
plt.show()

# %%
# save as pdf
import os
# output_dir = 'output_rebuttal_fig2-2-copy-camera_ready'
output_dir = 'output_presentation'
fig_dir = os.path.join(output_dir, f'd={d}', 'figures') # directory to save figures

os.makedirs(fig_dir, exist_ok=True)
fig.savefig(os.path.join(fig_dir, f'{ground_truth}.pdf'), bbox_inches='tight')

# %%
