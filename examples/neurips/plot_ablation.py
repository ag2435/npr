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
# %load_ext autoreload
# %autoreload 2

# %%
import os
import pickle
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from matplotlib.patches import Patch

# utils for plotting
from matplotlib import pyplot as plt
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

from helpers import (get_ablation_scores, 
                     get_scores,
                     plot_matplotlib,
                     plot_plotly)

# %%
ground_truth ='sinexp' # 'sum_gauss'
d = 1
logn_lo = 8
logn_hi = 14
n_trials = 100
# !!! MAKE SURE TO SET THE RIGHT BASELINE LOSS !!! 
baseline_loss = 1 # = noise**2

# output_dir = 'output_rebuttal_fig2-2-copy-camera_ready'
output_dir = 'output_rebuttal_fig2-2-copy-camera_ready-presentation'
fig_dir = os.path.join(output_dir, f'd={d}', 'figures') # directory to save figures
score_dir = os.path.join(output_dir, f'd={d}', 'scores_cv') # directory with saved scores

# %%
# KT-Nadaraya Watson
method = 'nw'
alias = 'NW'
kernel = 'wendland'

# # KT-KRR
# method = 'krr'
# alias = 'RR'
# kernel = 'gaussian'

ablation = True
ablation_scores, ablation_names = get_ablation_scores(score_dir, method, kernel, ground_truth, logn_lo, logn_hi, n_trials)
scores, names = get_scores(score_dir, method, kernel, ground_truth, logn_lo, logn_hi, n_trials)
all_scores = ablation_scores + scores
all_names = ablation_names + names
print('loaded these names:', all_names)
if ablation:
    include_names = [
        'k(x1,x2)', 
        'k((x1,x2), (y1,y2))', 
        f'k_{alias}'
    ]
else:
    include_names = ['full', 'st', 'rpcholesky', f'k_{alias}']
print('only plotting these names:', include_names)
number_of_names_to_actually_plot = 3

# %%
color_map = {
    'full': 'black',
    'st': 'green',
    'rpcholesky': 'red',
    'k(x1,x2)': 'orange',
    'k((x1,x2), (y1,y2))': 'red',
    'k_RR': 'blue',
    'k_NW': 'blue',
}
hatch_map = {
    'full': '',
    'st': '',
    'rpcholesky': '',
    'k(x1,x2)': '.',
    'k((x1,x2), (y1,y2))': '//',
    'k_RR': '',
    'k_NW': '',
}
legend_names = {
    'full': 'Full',
    'st': 'ST',
    'rpcholesky': 'RPCholesky',
    'k(x1,x2)': '$\mathbf{k}(x_1,x_2)$',
    'k((x1,x2), (y1,y2))': '$\mathbf{k}((x_1\oplus y_1), (x_2\oplus y_2))$',
    'k_RR': '$\mathbf{k}_{\mathrm{RR}}((x_1,y_1),(x_2,y_2))$',
    'k_NW': '$\mathbf{k}_{\mathrm{NW}}((x_1,y_1),(x_2,y_2))$',
}

# %%
interval = 2.5
colors = []
ys = []
positions = []
hatches = []
alphas = []
logn_list = []
for result in all_scores:
    logn = result['logn']
    name = result['name']
    # only plot 2^10 and 2^14
    if logn not in [10, 14]:
        continue
    # only plot the names in include_names
    if name not in include_names[:number_of_names_to_actually_plot]:
        continue

    if logn not in logn_list:
        logn_list.append(logn)
        # sort logn_list
    
    i = include_names.index(name)
    color = color_map[name]
    hatch = hatch_map[name]
    # excess_risk = np.array(result["scores"]) - baseline_loss
    excess_risk = np.array(result["scores"]) - baseline_loss
    # print(name, result['logn'], i)
    # define offset to be the number of indices away from the median
    offset = i - (len(include_names)-1) / 2
    position = logn_list.index(result['logn']) * interval * len(include_names) + offset*1.5
    alpha = 0.75
    # alpha=1

    y = np.log2(excess_risk)
    ys.append(y)
    positions.append(position)
    colors.append(color)
    hatches.append(hatch)
    alphas.append(alpha)


# %%
positions

# %%
fig, ax = plt.subplots(figsize=(10, 6))

bplot = ax.boxplot(
    ys,
    patch_artist=True,  # fill with color
    positions=positions,
    widths=1.2,
    # don't show outliers
    showfliers=False,
    showmeans=True,
    meanline=True,
    meanprops=dict(
        color='black', # set mean line color
    ),  
    medianprops=dict(color='black'),  # set median line color
)
# fill with colors
for patch, color, hatch, alpha in zip(bplot['boxes'], colors, hatches, alphas):
    patch.set_facecolor(color)
    patch.set_hatch(hatch)
    patch.set_alpha(alpha)
    # set mean line color

ax.set_xlabel('Input sample size $n$')
# set the x-axis labels
xticks = [logn_list.index(logn) * interval * len(include_names) for logn in logn_list]
print(xticks)
ax.set_xticks(xticks)
ax.set_xticklabels(["$2^{%d}$" % (logn) for logn in logn_list])
# set title
# add dashed vertical lines
# for logn in logn_list:
#     ax.axvline((logn+0.)*len(all_names), color='black', linestyle='--', linewidth=0.5, dashes=(5, 5))

# add legend
ax.legend(
    handles=[Patch(color=color_map[name], label=legend_names[name], alpha=0.5, hatch=hatch_map[name]) for i,name in enumerate(include_names[:number_of_names_to_actually_plot])],
    loc='upper right',
    # bbox_to_anchor=(1, 1),
    fontsize=20,
)

# add grid lines along y axis
ax.yaxis.grid(True)
# set ylim
# ax.set_ylim(-9, 2)
ax.set_ylabel(f"Mean Squared Error")
ymin, ymax = ax.get_ylim()
yticks = np.arange(int(ymin), int(ymax)+1, 2)
# print(yticks)
# set y ticks
ax.set_yticks(yticks)
ax.set_yticklabels([f'$2^{{{i}}}$' for i in yticks])

plt.show()

# %%
# save as pdf
os.makedirs(fig_dir, exist_ok=True)
fig.savefig(os.path.join(fig_dir, f"{ground_truth}_{method}_{kernel}_ablation{int(ablation)}{'_' + str(number_of_names_to_actually_plot) if number_of_names_to_actually_plot is not None else ''}.pdf"), bbox_inches='tight')

# %%
