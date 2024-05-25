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
                        help="kernel function", choices=['epanechnikov', 'gaussian', 'laplace'])
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
            3: "k^2(x1,x2) + k(x1,x2) * y1*y2", # 3
            2: "k(x1,x2)", # 2
            1: "k((x1,x2), (y1,y2))",  # 1
            0: "k(x1,x2) * (1+ y1*y2)", # 0
        }
        # convert the above to a dictionary where the key is the number
        # of the ablation and the value is the name
        
    elif method == 'krr':
        names = {
            3:"k(x1,x2) * (1+ y1*y2)", # 3
            2:"k(x1,x2)", # 2
            1:"k((x1,x2), (y1,y2))", # 1
            0:"k^2(x1,x2) + k(x1,x2) * y1*y2", # 0
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
        names = ['full', 'st']
    elif method == 'krr':
        # names = ['full', 'rpcholesky', 'st']
        names = ['full', 'st']
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