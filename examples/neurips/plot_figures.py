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
import numpy as np
import pandas as pd

# utils for plotting
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

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

from helpers import (get_ablation_scores, 
                     get_scores,)

# %%
ground_truth ='sinexp' # 'sum_gauss'
d = 1
logn_lo = 8
logn_hi = 14
n_trials = 100
# !!! MAKE SURE TO SET THE RIGHT BASELINE LOSS !!! 
baseline_loss = 1 # = noise**2

# output_dir = 'output_rebuttal_fig2-2-copy-camera_ready'
output_dir = 'output_presentation'
fig_dir = os.path.join(output_dir, f'd={d}', 'figures') # directory to save figures
score_dir = os.path.join(output_dir, f'd={d}', 'scores_cv') # directory with saved scores

# %%
# # KT-Nadaraya Watson
# method = 'nw'
# alias = 'NW'
# kernel = 'wendland'

# KT-KRR
method = 'krr'
alias = 'RR'
kernel = 'gaussian'

ablation = False
ablation_scores, ablation_names = get_ablation_scores(score_dir, method, kernel, ground_truth, logn_lo, logn_hi, n_trials)
scores, names = get_scores(score_dir, method, kernel, ground_truth, logn_lo, logn_hi, n_trials)
all_scores = ablation_scores + scores
all_names = ablation_names + names
print('loaded these names:', all_names)
if ablation:
    include_names = ['k(x1,x2)', 'k((x1,x2), (y1,y2))', f'k_{alias}']
else:
    include_names = ['full', 'st', 'rpcholesky', f'k_{alias}']
print('only plotting these names:', include_names)


# %%
def with_column(results, column_name, f):
    for result in results:
        result[column_name] = f(result)
    return results


# %%
# note that all_nw_scores is a list of dictionaries
# where each dictionary has keys 'logn', '_mean_score', '_mean_train_time', '_mean_test_time', 'name'
def log2_mean_std_excess_risk(result): 
    excess_risk = np.array(result['scores']) - baseline_loss
    log2_excess_risk = np.log2(excess_risk)
    return np.mean(log2_excess_risk), np.std(log2_excess_risk,ddof=1)
all_scores = with_column(all_scores, 's', log2_mean_std_excess_risk)
# create a line plot for the scores using matplotlib
df = pd.DataFrame([{key: d[key] for key in ['logn', 's', 'name']} for d in all_scores])
df['mean'] = df['s'].apply(lambda x: x[0])
df['std'] = df['s'].apply(lambda x: x[1])
# drop _mean_score
df = df.drop(columns=['s'])
# drop irrelevant names
df = df[df['name'].isin(include_names)]

# %%
df

# %%
colors = {
    'full': 'black',
    'st': 'green',
    'rpcholesky': 'red',
    'k(x1,x2)': 'orange',
    'k((x1,x2), (y1,y2))': 'red',
    'k_RR': 'blue',
    'k_NW': 'blue',
}
markers = {
    'full': 's',
    'st': '<',
    'rpcholesky': '>',
    'k(x1,x2)': '>',
    'k((x1,x2), (y1,y2))': '<',
    'k_RR': 'o',
    'k_NW': 'o',
}
line_styles = {
    'full': 'dashed',
    'st': 'dashed',
    'rpcholesky': 'dashed',
    'k(x1,x2)': 'dashed',
    'k((x1,x2), (y1,y2))': 'dashed',
    'k_RR': 'dotted',
    'k_NW': 'dotted',
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
# group by name
grouped = df.groupby('name')
# rearrange grouped to have order: k(x1,x2), k((x1,x2), (y1,y2)), k_NW
grouped = [(name, grouped.get_group(name)) for name in include_names]

# create a plot
fig, ax = plt.subplots(figsize=(10, 6))
for name, group in grouped:
    ax.errorbar(x=group['logn'], y=group['mean'], yerr=group['std'], 
        label=legend_names[name], 
        fmt='o', 
        marker=markers[name],
        capsize=5, 
        color=colors[name],
        # set marker size
    )
    if method == 'krr' and name in ['rpcholesky']:
        # rpcholesky has no clear line
        continue

    try:
        # filter out nan values
        logn, mean = group['logn'], group['mean']
        logn, mean = logn[~np.isnan(mean)], mean[~np.isnan(mean)]
        # draw trend line for non-nan
        z = np.polyfit(logn, mean, 1)
        p = np.poly1d(z)
        ax.plot(logn, p(logn), color=colors[name], linestyle=line_styles[name])
        # add equation of the line
        # y={z[0]:.2f}x+{z[1]:.2f}
        ax.text(logn.iloc[-1]+0.1, p(logn.iloc[-1]), f'$n^{{{z[0]:.2f}}}$', color=colors[name])
    except:
        # numpy is unable to fit a line
        pass
ax.set_xlabel('Input sample size $n$')
# set ticks
ax.set_xticks(np.arange(logn_lo, logn_hi+1, 2))
ax.set_xticklabels([f'$2^{{{i}}}$' for i in range(logn_lo, logn_hi+1, 2)])

ax.set_ylabel('Mean Squared Error')
# get the y limits
ymin, ymax = ax.get_ylim()
yticks = np.arange(int(ymin), int(ymax)+1, 2)
# print(yticks)
# set y ticks
ax.set_yticks(yticks)
ax.set_yticklabels([f'$2^{{{i}}}$' for i in yticks])

if ablation:
    # add legend
    ax.legend(
        # handles=[Patch(color=colors_list[i], label=legend_names[name], alpha=0.25 if name in ablation_names else 1) for i,name in enumerate(all_names)],
        loc='lower left',
        # bbox_to_anchor=(1, 1),
        fontsize=22,
    )
plt.grid()
plt.show()

# %%
# save as pdf
os.makedirs(fig_dir, exist_ok=True)
fig.savefig(os.path.join(fig_dir, f'{ground_truth}_{method}_{kernel}_ablation{int(ablation)}.pdf'), bbox_inches='tight')

# %%
if ablation:
    exit()

# %% [markdown]
# # Train-test time tradeoff

# %%
legend_names = {
    'full': 'Full',
    'st': 'Subsample',
    'rpcholesky': 'RPCholesky',
    'k_RR': 'KT (Ours)',
    'k_NW': 'KT (Ours)',
}


# %%
def mean_std_train_time(result): 
    train_times = np.array(result['train_times'])
    if method == 'krr':
        train_times = np.log2(train_times)
    return np.mean(train_times), np.std(train_times,ddof=1)
all_scores = with_column(all_scores, 'train', mean_std_train_time)

def mean_std_test_time(result): 
    test_times = np.array(result['test_times'])
    # train_times = np.log2(train_times)
    return np.mean(test_times), np.std(test_times,ddof=1)
all_scores = with_column(all_scores, 'test', mean_std_test_time)

# %%
# create a line plot for the scores using matplotlib
df_time = pd.DataFrame([{key: d[key] for key in ['logn', 'train', 'test', 'name']} for d in all_scores])
# note that _mean_train_time is in the format mean+-std
df_time['mean_train'] = df_time['train'].apply(lambda x: x[0])
df_time['std_train'] = df_time['train'].apply(lambda x: x[1])
# note that _mean_test_time is in the format mean+-std
df_time['mean_test'] = df_time['test'].apply(lambda x: x[0])
df_time['std_test'] = df_time['test'].apply(lambda x: x[1])
# drop _mean_score
df_time = df_time.drop(columns=['train', 'test'])
# drop irrelevant names
df_time = df_time[df_time['name'].isin(include_names)]
# drop logn besides 10 and 14
df_time = df_time[df_time['logn'].isin([10, 14])]

# %%
# group by name
grouped = df_time.groupby('name')
# rearrange grouped to have order: k(x1,x2), k((x1,x2), (y1,y2)), k_NW
grouped = [(name, grouped.get_group(name)) for name in ['full', 'st', 'rpcholesky', f'k_{alias}']]

# create a plot
fig, ax = plt.subplots(figsize=(10, 6))
for name, group in grouped:
    # draw error bars
    ax.errorbar(
        x=group['mean_train'], y=group['mean_test'], 
        xerr=group['std_train'], yerr=group['std_test'],
        fmt='o', 
        capsize=5, 
        color=colors[name],
        markersize=0,
    )
    marker_sizes = group['logn'].apply(lambda x: 5 + 1000 * ((x - logn_lo) / (logn_hi - logn_lo))**2)
    print(marker_sizes)
    ax.scatter(group['mean_train'], group['mean_test'], 
                color=colors[name], s=marker_sizes,
               marker='o',# markers[name],
               label=legend_names[name], alpha=0.25)
    # add text labels next to each marker
    for i, logn in enumerate(group['logn']):
        if method == 'nw':
            if name == 'full':# and logn > 10:
                xoffset = 0
                yoffset = 0
                ax.annotate(f'$n=2^{{{logn}}}$', (group['mean_train'].iloc[i]+xoffset, group['mean_test'].iloc[i]+yoffset), fontsize=20, color=colors[name])
            elif name == 'rpcholesky':# and logn > 10:
                xoffset = 0
                yoffset = 0
                ax.annotate(f'$n=2^{{{logn}}}$', (group['mean_train'].iloc[i]+xoffset, group['mean_test'].iloc[i]+yoffset), fontsize=20, color=colors[name])
            # elif name in ['kt','st'] and logn > 10:
            #     xoffset = 0
            #     yoffset = 0.1
            #     ax.annotate(f'$n=2^{{{logn}}}$', (group['mean_train'].iloc[i]+xoffset, group['mean_test'].iloc[i]+yoffset), fontsize=20, color=colors[name])
        elif method == 'krr':
            if name == 'full': # and logn > 10:
                xoffset = 0
                yoffset = 0
                ax.annotate(f'$n=2^{{{logn}}}$', (group['mean_train'].iloc[i]+xoffset, group['mean_test'].iloc[i]+yoffset), fontsize=20, color=colors[name])
            
ax.set_xlabel('Train time (s)')
if method == 'krr':
    # use linear x axis for NW
    # use log x axis for KRR

    # get current xticks
    xticks = ax.get_xticks()
    print(xticks)
    # ignore first and last tick
    xticks = xticks[1:-1]
    ax.set_xticks(xticks)
    xticklabels = []
    for i in xticks:
        two_to_i = np.power(2., i)
        if two_to_i < 1:
            # get position of first sig fig
            pos = int(np.floor(np.log10(two_to_i)))
            # round to the first sig fig
            two_to_i = np.round(two_to_i, -pos)
            xticklabels.append(f'${two_to_i}$')
        else:
            xticklabels.append(f'${int(two_to_i)}$')
    ax.set_xticklabels(xticklabels)

ax.set_ylabel('Test time (s)')
# get the y limits
ymin, ymax = ax.get_ylim()
yticks = np.arange(int(ymin), int(ymax)+1, 1)
# print(yticks)
# set y ticks
# ax.set_yticks(yticks)
# ax.set_yticklabels([f'$2^{{{i}}}$' for i in yticks])

# add legend
ax.legend(
    # set patch to have alpha=1
    handles=[Patch(color=colors[name], label=legend_names[name], alpha=1) for i,name in enumerate(include_names)],
    loc='upper left' if method == 'krr' else 'upper right',
    # bbox_to_anchor=(1, 1),
    fontsize=20,
)
plt.grid()
# plt.subplots_adjust(top=1)  # Adjust 'top' to allow more space at the top
plt.show()

# %%
# save as pdf
os.makedirs(fig_dir, exist_ok=True)
fig.savefig(os.path.join(fig_dir, f'{ground_truth}_{method}_{kernel}_train_test_times.pdf'), bbox_inches='tight')

# %%
