from ..util_k import kernel_eval, to_regression_kernel
from ...util_sample import get_Xy
from ..util_thin import kt_thin2, get_coreset_size, log4
from .util_falkon import Falkon

import numpy as np
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_X_y
from sklearn.base import RegressorMixin, ClassifierMixin
from functools import partial

import torch
import falkon
from falkon.center_selection import CenterSelector, UniformSelector

def get_cpu_options(debug=False):
    return falkon.FalkonOptions(keops_active="no", use_cpu=True, debug=debug)

def get_falkon_kernel(kernel, sigma):
    # Translate parameters from sklearn to falkon
    if kernel == 'gauss':
        falkon_kernel = falkon.kernels.GaussianKernel(sigma=sigma)
    elif kernel == 'laplace':
        falkon_kernel = falkon.kernels.LaplacianKernel(sigma=sigma)
    else:
        raise ValueError(f'kernel={kernel} not implemented')
    
    return falkon_kernel

def numpy_to_torch(a):
    return torch.from_numpy(a).to(dtype=torch.float32)

def torch_to_numpy(a):
    return a.detach().cpu().numpy()

class KernelRidgeFalkon(falkon.Falkon):
    """
    Wrapper for falkon.Falkon estimator
    """

    def __init__(self, alpha=1, sigma=1, m=None, debug=False, postprocess=None):
        self.alpha = alpha
        self.sigma = sigma
        self.debug = debug
        self.m = m
        self.postprocess = postprocess

        super().__init__(
            M=None,
            kernel=get_falkon_kernel(self.kernel_name, sigma),
            penalty=alpha,
            center_selection=UniformSelector(random_gen=None, num_centers=0),
            options=get_cpu_options(debug=debug)
        )

    def fit(self, X, y):
        if self.postprocess:
            y = self._validate_targets(y)
            
        # Reconfigure params
        self.kernel = get_falkon_kernel(self.kernel_name, self.sigma)
        self.penalty = self.alpha

        n = len(X)
        if self.m is None:
            m = int(log4(n))
        else:
            m = self.m

        self.center_selection.num_centers = get_coreset_size(n, m=m)
        # print(f'KernelRidgeFalkon: original size: {len(X)}, coreset size: {self.center_selection.num_centers}')

        super().fit(numpy_to_torch(X), numpy_to_torch(y))

    def predict(self, X):
        pred = super().predict(numpy_to_torch(X)).numpy()

        if self.postprocess == 'round':
            return np.round(pred)
        
        elif self.postprocess == 'argmax':
            max_index = np.argmax(pred, axis=1)
            one_hot = np.zeros_like(pred)
            one_hot[np.arange(len(pred)), max_index] = 1
            return one_hot

        elif self.postprocess == 'threshold':
            # print('thresholding')
            # only for binary classification
            pred = pred.squeeze()
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
            # cast pred to int
        else:
            assert self.postprocess is None

        return pred
    
    def _validate_targets(self, y):
        # y = column_or_1d(y, warn=True)
        cls = np.unique(y)
        if len(cls) < 2:
            print(y)
            # raise ValueError(
            #     "The number of classes has to be greater than one; got %d class"
            #     % len(cls)
            # )

        self.classes_ = cls

        return y

class KTCenterSelector(CenterSelector):
    def __init__(self, random_gen, kernel, sigma, m=None, ydim=1, store_K=False, normalize_y=True):
        self.kernel = kernel
        self.sigma = sigma
        self.m = m
        self.ydim = ydim
        self.store_K = store_K
        self.normalize_y = normalize_y

        super().__init__(random_gen)

    def select_indices(self, X, Y):
        # KERNEL THINNING
        
        # Perform conversion to torch data structure
        X, Y = torch_to_numpy(X), torch_to_numpy(Y)
        # Check that X and y have correct shape        
        X, y = check_X_y(X, Y)

        d = X.shape[-1]
        var_k = self.sigma**2

        params_k_swap = {"name": self.kernel, "var": var_k, "d": int(d)}
        params_k_split = {"name": self.kernel, "var": var_k, "d": int(d)}
        
        split_kernel = partial(kernel_eval, params_k=params_k_split)
        swap_kernel = partial(kernel_eval, params_k=params_k_swap)
        
        # if self.use_regression_kernel:
        split_kernel = to_regression_kernel(split_kernel, ydim=self.ydim)
        swap_kernel = to_regression_kernel(swap_kernel, ydim=self.ydim)

        # if self.use_regression_kernel:
        Xy = get_Xy(X, y, normalize_y=self.normalize_y)
        kt_coreset = kt_thin2(
            Xy, 
            split_kernel, 
            swap_kernel, 
            seed=self.random_gen, 
            m=self.m,
            store_K=self.store_K
        )

        # else:
        #     kt_coreset = kt_thin2(X, split_kernel, swap_kernel, seed=self.seed, 
        #                           m=self.m)
            
        # print(f'original size: {len(X)}, kt coreset size: {len(kt_coreset)}')

        return numpy_to_torch(X[kt_coreset]), kt_coreset
    
    def select(self, X, Y):
        """Select M observations from 2D tensor `X`, preserving device and memory order.

        The selection strategy is uniformly at random. To control the randomness,
        pass an appropriate numpy random generator to this class's constructor.

        Parameters
        ----------
        X
            N x D tensor containing the whole input dataset. If N is lower than the number of
            centers this class is programmed to pick, a warning will be raised and only N centers
            will be returned.
        Y
            Optional N x T tensor containing the input targets. If `Y` is provided,
            the same observations selected for `X` will also be selected from `Y`.
            Certain models (such as :class:`falkon.models.LogisticFalkon`) require centers to be
            extracted from both predictors and targets, while others (such as
            :class:`falkon.models.Falkon`) only require the centers from the predictors.

        Returns
        -------
        X_M
            The randomly selected centers. They will be in a new, memory-contiguous tensor.
            All characteristics of the input tensor will be preserved.
        (X_M, Y_M)
            If `Y` was different than `None` then the entries of `Y` corresponding to the
            selected centers of `X` will also be returned.
        """
        out = self.select_indices(X, Y)
        if len(out) == 2:
            return out[0]
        return out[0], out[1]
    
class KernelRidgeFalkonKT(Falkon):
    """
    Wrapper for falkon.Falkon estimator with centers chosen using KT (regression variant)
    """
    def __init__(self, alpha=1, sigma=1, debug=False, m=None, postprocess=None, 
                ydim=1, store_K=True, normalize_y=False):
        self.alpha = alpha
        self.sigma = sigma
        self.debug = debug
        self.m = m
        self.postprocess = postprocess

        self.ydim = ydim
        self.store_K = store_K
        self.normalize_y = normalize_y

        super().__init__(
            M=None,
            kernel=get_falkon_kernel(self.kernel_name, sigma),
            penalty=alpha,
            center_selection=KTCenterSelector(
                123, 
                self.kernel_name, 
                sigma, 
                m=m, 
                ydim=ydim, store_K=store_K, normalize_y=normalize_y
            ),
            options=get_cpu_options(debug=debug)
        )

    def fit(self, X, y):
        if self.postprocess:
            y = self._validate_targets(y)

        # Reconfigure params given (possibly updated) sigma
        self.kernel = get_falkon_kernel(self.kernel_name, self.sigma)
        self.penalty = self.alpha

        super().fit(numpy_to_torch(X), numpy_to_torch(y))

    def predict(self, X):
        pred = super().predict(numpy_to_torch(X)).numpy(force=True)

        if self.postprocess == 'round':
            return np.round(pred)
        elif self.postprocess == 'argmax':
            max_index = np.argmax(pred, axis=1)
            one_hot = np.zeros_like(pred)
            one_hot[np.arange(len(pred)), max_index] = 1
            return one_hot
        elif self.postprocess == 'threshold':
            # print('thresholding')
            # only for binary classification
            pred = pred.squeeze()
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0

        return pred
    
    def _validate_targets(self, y):
        # y = column_or_1d(y, warn=True)
        cls = np.unique(y)
        if len(cls) < 2:
            print(y)
            # raise ValueError(
            #     "The number of classes has to be greater than one; got %d class"
            #     % len(cls)
            # )

        self.classes_ = cls

        return y

class KernelRidgeFalkonGauss(KernelRidgeFalkon):
    kernel_name = 'gauss'
class KernelRidgeFalkonLaplace(KernelRidgeFalkon):
    kernel_name = 'laplace'

class KernelRidgeFalkonKTGauss(KernelRidgeFalkonKT):
    kernel_name = 'gauss'
class KernelRidgeFalkonKTLaplace(KernelRidgeFalkonKT):
    kernel_name = 'laplace'    

class KernelRidgeFalkonGaussRegressor(KernelRidgeFalkonGauss, RegressorMixin):
    pass
class KernelRidgeFalkonLaplaceRegressor(KernelRidgeFalkonLaplace, RegressorMixin):
    pass

class KernelRidgeFalkonKTGaussRegressor(KernelRidgeFalkonKTGauss, RegressorMixin):
    pass
class KernelRidgeFalkonKTLaplaceRegressor(KernelRidgeFalkonKTLaplace, RegressorMixin):
    pass

class KernelRidgeFalkonGaussClassifier(KernelRidgeFalkonGauss, ClassifierMixin):
    pass
class KernelRidgeFalkonLaplaceClassifier(KernelRidgeFalkonLaplace, ClassifierMixin):
    pass

class KernelRidgeFalkonKTGaussClassifier(KernelRidgeFalkonKTGauss, ClassifierMixin):
    pass
class KernelRidgeFalkonKTLaplaceClassifier(KernelRidgeFalkonKTLaplace, ClassifierMixin):
    pass

def get_regressor(kernel, use_kt_center_selection=False):
    if kernel == 'gauss':
        if use_kt_center_selection:
            return KernelRidgeFalkonKTGaussRegressor
        else:
            return KernelRidgeFalkonGaussRegressor
    elif kernel == 'laplace':
        if use_kt_center_selection:
            return KernelRidgeFalkonKTLaplaceRegressor
        else:
            return KernelRidgeFalkonLaplaceRegressor
    else:
        raise ValueError(f'kernel={kernel} not implemented')
    
def get_classifier(kernel, use_kt_center_selection=False):
    if kernel == 'gauss':
        if use_kt_center_selection:
            return KernelRidgeFalkonKTGaussClassifier
        else:
            return KernelRidgeFalkonGaussClassifier
    elif kernel == 'laplace':
        if use_kt_center_selection:
            return KernelRidgeFalkonKTLaplaceClassifier
        else:
            return KernelRidgeFalkonLaplaceClassifier
    else:
        raise ValueError(f'kernel={kernel} not implemented')