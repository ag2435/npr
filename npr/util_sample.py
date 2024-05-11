# import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import numpy.linalg as npl
import os
import pickle as pkl
from scipy import stats

from .util_k import laplace, gaussian, euclidean_distances

'''
File containing helper functions for details about target P and
drawing samples / loading mcmc samples from file

Albert: Copied from examples/gkt/util_sample.py
'''
######## functions related to setting P and sampling from it ########

def grid(k=100, d=1, mag=0.5):
    """
    Form grid of k^d points in [-mag, mag]^d
    """
    points_1d = np.linspace(-mag, mag, k)
    mesh = np.meshgrid(*[points_1d for _ in range(d)])
    points = np.stack(mesh, axis=-1).reshape(-1, d)
    return points

def sample(n, params_p, seed=None):
    """Returns n sample points drawn iid from a specified distribution
    
    Args:
      n: Number of sample points to generate
      params_p: Dictionary of distribution parameters including
        name: Distribution name in {"gauss"}
        var: Variance parameter
        d: Dimension of generated vectors
      seed: (Optional) Random seed to set prior to generation; if None,
        no seed will be set
    """
    name = params_p["name"]
    if name == "gauss":
        sig = np.sqrt(params_p["var"])
        return(sig * rand.default_rng(seed).standard_normal(size=(n, params_p["d"])))
    elif name == "unif":
        d = params_p["d"]
        sig = np.sqrt(params_p["var"])
        # NOTE: 
        # - Unif[-1,1] has variance 1/3
        # - Unif[-sqrt(3), sqrt(3)] has variance 3
        # uncomment to use i.i.d. sampling
        # return(rand.default_rng(seed).random(size=(n, params_p["d"])) - 0.5) * np.sqrt(12) * sig
        
        # "sample" deterministic points to get rid of any randomness in the data
        #   this is useful for debugging and precisely estimating population (i.e., test) loss
        sqrt_3 = np.sqrt(3)
        # generate at least n iid samples from Unif[-1,1]^d
        k = int(np.ceil(np.power(n, 1/d)))
        return(sig * grid(k, d, mag=sqrt_3))
    elif name == "unif-[0,1]":
        return np.linspace(0, 1, n)[:, np.newaxis]
    elif name == "mog":
        rng = rand.default_rng(seed)
        w = params_p["weights"]
        n_mix = rng.multinomial(n, w)
        for i, ni in enumerate(n_mix):
            mean = params_p["means"][i, :]
            cov = params_p["covs"][i, :, :]
            temp = rng.multivariate_normal(mean=mean, cov=cov, size=ni)
            if i == 0:
                x = temp
            else:
                x = np.vstack((x, temp))
        rng.shuffle(x)
        return(x)
    elif params_p["name"] == "diag_mog":
        rng = rand.default_rng(seed)
        w = params_p["weights"]
        d = params_p["d"]
        n_mix = rng.multinomial(n, w)
        for i, ni in enumerate(n_mix):
            mean = params_p["means"][i, :]
            cov = params_p["covs"][i] * np.eye(d)
            temp = rng.multivariate_normal(mean=mean, cov=cov, size=ni)
            if i == 0:
                x = temp
            else:
                x = np.vstack((x, temp))
        rng.shuffle(x)
        return(x)
    elif params_p["saved_samples"] == True:
        if 'Hinch' in params_p["name"]: # for this case, samples are all preloaded
            assert(params_p["include_last"])
            filename = os.path.join(params_p["data_dir"], "{}_samples_n_{}.pkl".format(params_p["name"], n))
            with open(filename, 'rb') as file:
                return(pkl.load(file))
        else:
            if '_float_step' in params_p["name"]:
                assert(params_p["include_last"])
                end = params_p["X"].shape[0]
                sample_idx = np.linspace(0, end-1, n,  dtype=int, endpoint=True)
                return(params_p["X"][sample_idx])
            else:
                end = params_p["X"].shape[0]
                # compute thinning parameter
                step_size = int(end / n)
                start = end-step_size*n
                assert(step_size>=1)
                assert(start>=0)
                if params_p["include_last"]:
                    return(params_p["X"][end-1:start:-step_size][::-1])
                else:
                    return(params_p["X"][start:end:step_size])
        
    raise ValueError("Unrecognized distribution name {}".format(params_p["name"]))

def pdf(x, params_p):
    """Returns pdf (at x) from a specified distribution
    
    Args:
      x: either scalar or array of shape (n,d)
      params_p: Dictionary of distribution parameters including
        name: Distribution name in {"gauss"}
        var: Variance parameter
        d: Dimension of generated vectors
    """
    if params_p['name'] == "gauss":
        X_var = params_p['var']
        d = params_p['d']
        return stats.multivariate_normal(mean=np.zeros(d), cov=np.diag(X_var * np.ones(d))).pdf(x)
    
    elif params_p['name'] == "diag_mog":
        w = params_p["weights"]
        d = params_p["d"]

        pdfs = []
        for i in range(len(w)):
            mean = params_p["means"][i, :]
            cov = params_p["covs"][i] * np.eye(d)
            pdfs.append( stats.multivariate_normal(mean=mean, cov=cov).pdf(x) )

        result = np.average(np.stack(pdfs, axis=-1), axis=-1, weights=w, keepdims=True)
        return result

    raise ValueError("Unrecognized distribution name {}".format(params_p["name"]))


def compute_diag_mog_params(M=int(4), snr=3.):
    """Returns diagonal mixture of Gaussian target distribution settings for d=2
    
    Args:
      M: (Optional) Integer, number of components
      snr: (Optional) Scaling of the means    
    """
    d = int(2)
    weights = np.ones(M)
    weights /= np.sum(weights)

    # change this to set the means apart
    means = np.zeros((M, d))
    if M == 3:
        means = snr*np.array([[1., 1.], [-1., 1], [-1., -1.]])
    if M == 4: 
        means = snr*np.array([[1., 1.], [-1., 1], [-1., -1.], [1., -1.]])
    if M == 6:
        means = snr*np.array([[1., 1.], [-1., 1], [-1., -1.], [1., -1.], [0, 2.], [-2, 0.]])
    if M == 8:
        means = snr*np.array([[1., 1.], [-1., 1], [-1., -1.], [1., -1.], [0, 2.], [-2, 0.], [2, 0.], [0, -2.]])
    covs = np.ones(M)


    # compute the expected value of E[||X-Y||^2] for X, Y iid from P
    mean_sqdist = 0.
    for i in range(M):
        for j in range(M):
            temp = npl.norm(means[i])**2 + npl.norm(means[j])**2 - 2 * np.dot(means[i], means[j])
            temp += d*(covs[i]+ covs[j])
            mean_sqdist += weights[i] * weights[j] * temp
            
    params_p = {"name": "diag_mog", 
                 "weights": weights,
                "means": means,
                "covs": covs,
                "d": int(d),
                "mean_sqdist" : mean_sqdist,
               "saved_samples": False,
               "flip_Pnmax": False
               }
    return(params_p) 

def compute_params_p(args):
    ''' 
        return dimensionality, params_p, and var_k, for the experiment
    '''
    ## P and kernel parameters ####
    if args['P'] == "unif":
        d = args['d']
        var_p = args['var_P'] # Variance of P
        var_k = float(2*d)
        params_p = {"name": "unif", "var": var_p, "d": int(d)}

    if args['P'] == "unif-[0,1]":
        d = args['d']
        var_p = args['var_P']
        var_k = float(2*d)
        params_p = {"name": "unif-[0,1]", "var": var_p, "d": int(d)}
        
    if args['P'] == "gauss":       
        d = args['d']
        var_p = args['var_P'] # Variance of P
        var_k = float(2*d) # Variance of k
        params_p = {"name": "gauss", "var": var_p, "d": int(d), "saved_samples": False,
                   "flip_Pnmax": False}
        
    if args['P'] == "mog":
        d = args['d']
        var_p = args['var_P'] # Variance of P

        if d==1:
            params_p = {
                'name'      : 'diag_mog', 
                'weights'   : [0.7, 0.3], # Raaz (10/12): use unbalanced MoG
                'means'     : np.array([[-2.], [2.]]), 
                'covs'      : [var_p, var_p],
                'd'         : int(d)
            }
            var_k = float(2*d)

        elif d==2:
            # d will be set to 2
            assert(args['M'] in [3, 4, 6, 8])
            params_p = compute_diag_mog_params(args['M'])
            d = params_p["d"]
            var_k = float(2*d)

        else:
            raise ValueError(f'cannot use mog for d>2')
        
    # if args.P == "mcmc":
    #     # d will be set automatically; nmax needs to be changed
    #     assert(args.filename is not None)
    #     params_p = compute_mcmc_params_p(args.filename)
    #     d = params_p["d"]
    #     var_k = (params_p["med_dist"])**2
    return(d, params_p, var_k)


######## functions related to creating the regression dataset ########


def get_noise(n, noise, seed=None):
    return rand.default_rng(seed).normal(0,noise, size=(n,))

def sin(X, noise, seed=None):
    # Add 1 to sin so that it is always positive (plus or minus noise)
    y = np.sin(npl.norm(X, axis=-1) * 2 * np.pi) + get_noise(len(X), noise, seed)
    return y

def stair(X, noise, seed=None):
    return 2 * np.floor(npl.norm(X, axis=-1)) + get_noise(len(X), noise, seed)

def quadratic(X):
    return npl.norm(X, axis=-1)**2

def sum_gauss(X, noise, k=10, seed=None):
    """
    Sum of m (= k^d) Gaussians
        y = \sum_{i=1}^m \exp(-||x - a_i||^2 / (2\sigma^2)) + noise
    \sigma = 0.25
    a_i are evenly spaced between -1 and 1
    noise is N(0, noise) i.i.d. for each x

    Args:
        X: input data (n,d), d>=1
        noise: std of guassian noise
        k: number of Gaussians
        seed: random seed
    
    Returns:
        y: sum of k^d Gaussians evaluated at X
    """
    n, d = X.shape
    anchor_points = grid(k, d, mag=1.)
    y = gaussian(X, anchor_points, 0.25).sum(axis=-1) + get_noise(n, noise, seed)
    return y

def logistic(X, k=10, seed=None):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # construct discriminant function
    anchor_points = np.linspace(-1, 1, k)[:, np.newaxis]
    f = 4* (gaussian(X, anchor_points, 0.125).sum(axis=-1) - 0.5)# / len(anchor_points)
    # f = gauss(X, anchor_points, 0.125).sum(axis=-1)

    # apply sigmoid to get probability
    prob = sigmoid(f)
    # plt.scatter(X.squeeze(), f.squeeze(), label='f')
    # plt.scatter(X.squeeze(), prob.squeeze(), label='prob')
    # # show legend
    # plt.legend()
    # plt.show()
    
    # sample from Bernoulli distribution and rescale to {-1,1}
    # labels = 2*( np.random.binomial(1, prob) - 0.5 )
    labels = rand.default_rng(seed).binomial(1, prob)

    return labels, f, prob

def sum_laplace(X, noise, k=10, seed=None):
    d = X.shape[-1]
    # anchor_points = np.array([-np.ones(d), np.ones(d)])
    anchor_points = np.linspace(-1, 1, k)[:, np.newaxis]
    # y = laplacian(X, anchor_points, 0.125).sum(axis=-1) / len(anchor_points) + get_noise(len(X), noise)
    y = laplace(X, anchor_points, 0.25).sum(axis=-1) + get_noise(len(X), noise, seed)
    return y

# def step(X, noise):
#     # y = np.sign(X[:,0])
#     prob = 1 / (1 + np.exp(-(X / (noise + 1e-10))[:,0]))
#     y = np.random.binomial(1, prob)

#     # if noise:
#     #     assert 0 <= noise <= 1
#     #     flip = np.random.binomial(1, noise, size=(X.shape[0],))
#     #     y[flip==1] *= -1

#     # Re-normalize to [0,1]
#     # y = (y + 1) / 2

#     return y

def get_Xy(X, y, 
        #    normalize_y=True
           ):
    """
    Combine X with shape (n,d) and y with shape (n,) into a single ndarray of shape (n,d+1)
    If y is a one-hot vector for multiclass prediction, then y should have shape (n,k) and the output will have shape (n,d+k)
    """
    if len(y.shape) == 1:
        Xy_train = np.concatenate([X, y[...,np.newaxis]], axis=-1)
    else:
        assert len(y.shape) == 2
        Xy_train = np.concatenate([X, y], axis=-1)
    return Xy_train

class ToyData(object):
    def __init__(self, X_name, f_name, X_var=1, d=1, noise=1, M=4, k=None) -> None:
        self.X_name = X_name
        self.f_name = f_name
        self.X_var = X_var
        self.d = d
        self.noise = noise
        self.M = M
        self.k = k

        self.d, self.params_p, self.var_k = compute_params_p(args={
            'P'     : X_name,
            'var_P' : X_var,
            'd'     : d,
            'M'     : M
        })
        assert self.d == d

        self.params_f = {
            'f_name' : f_name,
            'noise' : noise,
            'k' : k
        }

    def get_params(self):
        return self.d, self.params_p, self.params_f, self.var_k
    
    def __repr__(self) -> str:
        return f"ToyData(X_name={self.X_name}, f_name={self.f_name}, X_var={self.X_var}, d={self.d}, noise={self.noise}, M={self.M}, k={self.k})"

    def sample(self, n, seed=None, shuffle=True):
        # Create X data

        X = sample(n, params_p=self.params_p, seed=seed)
        
        # Create y data

        if self.f_name == 'sin':
            y = sin(X, self.noise, seed)
        elif self.f_name == 'stair':
            y = stair(X, self.noise, seed)
        # elif self.f_name == 'quad':
        #     y = quadratic(X)
        elif self.f_name == 'sum_gauss':
            y = sum_gauss(X, self.noise, self.k, seed)
        elif self.f_name == 'logistic':
            y = logistic(X, self.k, seed)[0]
        elif self.f_name == 'sum_laplace':
            y = sum_laplace(X, self.noise, self.k, seed)
        # elif self.f_name == 'step':
        #     y = step(X, self.noise)
        elif self.f_name == 'dnc_paper':
            y = np.minimum(X, 1-X).sum(axis=-1) + get_noise(len(X), self.noise, seed)
        else:
            raise ValueError(f'f_name={self.f_name} not supported')
        
        # shuffle data
        if shuffle:
            shuffle_idx = np.arange(n)
            rand.shuffle(shuffle_idx)
            X = X[shuffle_idx]
            y = y[shuffle_idx]

        return X, y
    
    def pdf(self, X):
        """
        Pdf of data distribution evaluated at X
        """

        return pdf(X, self.params_p)
    
def get_toy_dataset(X_name, f_name, n, X_var=1, d=1, noise=1, M=4, k=None):
    """
    Sample from toy dataset
    Use cached result if possible    
    """

    toy_data = ToyData(
        X_name, 
        f_name, 
        X_var=X_var, 
        d=d, 
        noise=noise, 
        M=M,
        k=k,
    )
    print('sampling dataset with params', str(toy_data))
    dataset = toy_data.sample(n)

    return dataset

def get_sigma_heuristic(X, sample_size=100, return_dist=False):
    """
    Median heuristic for choosing bandwidth param
    """

    X_sample = X[np.random.choice(len(X), size=sample_size, replace=False)]
    Y_sample = X[np.random.choice(len(X), size=sample_size, replace=False)]

    dist = euclidean_distances(X_sample, Y_sample, squared=False).flatten()
    sigma = np.median(dist)

    if return_dist:
        return sigma, dist
    
    return sigma
