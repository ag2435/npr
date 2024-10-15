import pickle
import os
import argparse
import numpy as np

################ Helper functions for argument parsing ################

def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", "-m", default='nw', type=str,
                        help="non-parametric regression method", choices=['nw', 'krr', 'rpcholesky'])
    parser.add_argument("--thin", "-thin", default='full', type=str,
                        help="thinning method", choices=['full', 'st', 'kt', 'rpcholesky'])
    parser.add_argument("--kernel", "-k", default='epanechnikov', type=str,
                        help="kernel function", 
                        choices=['epanechnikov', 'wendland', 'gaussian', 'laplace'])
    parser.add_argument("--sigma", "-sig", default='0.1', type=str,
                        help="bandwidth for kernel"\
                            "Note: if a single number, then use that for all experiments;" \
                            "otherwise, if a list with format [lo,hi,delta], then use it to perform grid search.")
    parser.add_argument("--alpha", "-alpha", default='1e-3', type=str,
                        help="regularization parameter for kernel ridge regression"\
                            "Note: if a single number, then use that for all experiments;" \
                            "otherwise, if a list with format [lo,hi,delta], then use it to perform grid search.")
    parser.add_argument("--n_trials", "-t", default=100, type=int,
                        help="number of trials to run each setting")
    parser.add_argument("--output_path", "-op", default='output', type=str,
                        help="directory for storing output")
    parser.add_argument("--ablation", default=0, type=int,
                        help="kernel ablation study", choices=[0, 1, 2, 3])
    parser.add_argument('--no_swap', action='store_true', default=False,
                        help='if set, do not perform swap step of kernel thinning')
    parser.add_argument('--n_jobs', '-j', default=1, type=int,
                        help="number of parallel jobs during cross validation")
    parser.add_argument('--force', '-f', action='store_true',
                        help="force overwrite of existing files")
    return parser

def parse_param_str(param_str):
    """
    Parse a string to a float or a list of floats.
    param_str could be `val` (e.g., '0.1') or [lo,hi,delta] (e.g., '[0.1,1.0,0.1]')

    In the first case, we return the float value.
    In the second case, we return a list of floats from lo to hi (inclusive) with delta step size.
    """
    if param_str[0] == '[':
        lo, hi, delta = map(float, param_str[1:-1].split(','))
        # return lo, hi, delta
        return np.linspace(lo, hi, int(np.round((hi-lo)/delta) + 1))
    else:
        return float(param_str)

################ Helper functions for plotting figures ################

def get_ablation_scores(score_dir, method, kernel, ground_truth, logn_lo, logn_hi, n_trials):
    results = []
    if method == 'nw':
        names = {
            3: "k_RR", #"k_RR = k^2(x1,x2) + k(x1,x2) * y1*y2", # 3
            2: "k(x1,x2)", # 2
            1: "k((x1,x2), (y1,y2))",  # 1
            0: "k_NW" # = k(x1,x2) * (1+ y1*y2)", # 0
        }
        # convert the above to a dictionary where the key is the number
        # of the ablation and the value is the name
        
    elif method == 'krr':
        names = {
            3:"k_NW", # = k(x1,x2) * (1+ y1*y2)", # 3
            2:"k(x1,x2)", # 2
            1:"k((x1,x2), (y1,y2))", # 1
            0:"k_RR" # = k^2(x1,x2) + k(x1,x2) * y1*y2", # 0
        }
    else:
        raise ValueError()

    for a, name in names.items():
        ablation_str = f"-ablation{a}" if a > 0 else ""
        score_file = os.path.join(
            score_dir,
            f"kt-{method}-k={kernel}-gt={ground_truth}-logn={logn_lo}_{logn_hi}-t{n_trials}{ablation_str}.pkl"
        )
        with open(score_file, 'rb') as f:
            result = pickle.load(f)
            
        # name = names[a]
        for r in result:
            r['name'] = name
            results.append(r)
    return results, list(names.values())

def get_scores(score_dir, method, kernel, ground_truth, logn_lo, logn_hi, n_trials):
    results = []

    if method == 'nw':
        names = ['full', 'rpcholesky', 'st']
        # names = ['full', 'st']
    elif method == 'krr':
        names = ['full', 'rpcholesky', 'st']
        # names = ['full', 'st']
    else:
        raise ValueError()

    for name in names:
        trials = (1 if name == 'full' else n_trials)
        lo = logn_lo
        hi = logn_hi
        # hi = 10 if thin == 'rpcholesky' else logn_hi
        score_file = os.path.join(
            score_dir,
            f"{name}-{method}-k={kernel}-gt={ground_truth}-logn={lo}_{hi}-t{trials}.pkl"
        )
        with open(score_file, 'rb') as f:
            result = pickle.load(f)
            
        for r in result:
            # name = f"{n}-{method}"
            # if name not in names:
            #     names.append(name)
            r['name'] = name
            results.append(r)
    return results, names


################ Helper functions for plotting ################

# utils for plotting
import plotly.graph_objects as go
import plotly.colors as colors

def plot_plotly(
        all_scores, 
        f,
        include_names=None, 
        scale = 'log2', 
    ):
    """
    Make Plotly Box plots

    Args:
        all_scores: list of dictionaries containing `logn`, `name`, and $key
        f: function that takes in a result and outputs an 1D array
        include_names: list of names to include (if None, include all names)
        scale: scale of y-axis ('log2' or 'linear')
        kwargs: additional arguments to pass to `update_layout`
    """
    all_names = [result['name'] for result in all_scores]
    display_names = set(include_names) if include_names is not None else set(all_names)
    display_names = list(display_names)

    fig = go.Figure()
    colors_list = colors.qualitative.Plotly * (
        len(display_names) // len(colors.qualitative.Plotly) + 1
    )
    colors_used = set()
    for result in all_scores:
        name = result['name']
        if name not in display_names:
            continue
        
        color = colors_list[display_names.index(name)]
        if scale == 'log2':
            y = np.log2(np.abs(f(result)))
        elif scale == 'linear':
            y = np.abs(f(result))
        else:
            raise ValueError(f"Unknown scale: {scale}")

        trace = go.Box(
            x=[result['logn']]*len(y),
            y=y,
            name=name,
            # opacity=0.5,
            legendgroup=name,
            line_color=color,
            offsetgroup=name,
            showlegend=color not in colors_used,
            boxmean=True,
        )

        fig.add_trace(trace)
        colors_used.add(color)

    # fig.update_yaxes(title_text=f"{scale}({key})")
    fig.update_xaxes(title_text="log2(n)", type='linear')
    fig.update_layout(
        boxmode='group',
    )
    return fig

# make a matplotlib version of the above figure
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_matplotlib(
        ax, 
        all_scores, 
        f,
        all_names, 
        ablation_names, 
        legend_names, 
        colors_list):
    """
    Plot box plot in matplotlib
    
    Args:
    all_scores: list of dictionaries with keys 'name', 'logn', 'scores'
    all_names: list of names of methods
    ablation_names: list of names of ablation methods
    legend_names: dictionary mapping method names to legend names
    ax: matplotlib axis
    f: function that takes in a result and outputs an 1D array
    colors_list: list of colors
    """
    offset = 2
    colors = []
    ys = []
    positions = []
    hatches = []
    alphas = []
    logn_list = []
    for result in all_scores:
        logn = result['logn']
        if logn not in logn_list:
            logn_list.append(logn)
        name = result['name']

        i = all_names.index(name)
        color = colors_list[i]
        hatch = '.' if name in ablation_names else None
        # excess_risk = np.array(result["scores"]) - baseline_loss
        excess_risk = f(result)
        # print(name, result['logn'], i)
        position = result['logn'] * len(all_names) + 1.5*i + offset
        alpha = 0.25 if name in ablation_names else 1.
        # alpha=1

        y = np.log2(excess_risk)
        ys.append(y)
        positions.append(position)
        colors.append(color)
        hatches.append(hatch)
        alphas.append(alpha)

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
        # patch.set_hatch(hatch)
        patch.set_alpha(alpha)
        # set mean line color

    ax.set_xlabel("$n$", fontsize=16)
    # set the x-axis labels
    xticks = [(logn +1)* len(all_names) for logn in logn_list]
    # print(xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["$2^{%d}$" % (logn) for logn in logn_list])
    # set title
    # ax.set_title(f"Excess risk vs n (ground_truth={ground_truth})")
    # add dashed vertical lines
    for logn in logn_list:
        ax.axvline((logn+0.)*len(all_names), color='black', linestyle='--', linewidth=0.5, dashes=(5, 5))

    # add legend
    ax.legend(
        handles=[Patch(color=colors_list[i], label=legend_names[name], alpha=0.25 if name in ablation_names else 1) for i,name in enumerate(all_names)],
        loc='lower left',
        # bbox_to_anchor=(1, 1),
        fontsize='large',
    )

    # # add grid lines along y axis
    # ax.yaxis.grid(True)
    # # set ylim
    # ax.set_ylim(-9, 2)
    # ax.set_ylabel(f"$\log_2(excess~risk)$", fontsize=16)