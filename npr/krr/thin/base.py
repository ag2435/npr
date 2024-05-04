from .util_k_mmd import gauss, laplacian, gaussian_M, laplacian_M, sobolev
from ..rfm2.util_rfm_estimators import get_rfm_regressor

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import torch
import numpy as np
from joblib import Memory

location = './cachedir'
memory = Memory(location, verbose=0)

@memory.cache
def get_feature_matrix(X, y, kernel, alpha=1, sigma=1, rfm_iters=1, val_data=None):
    """
    We put this as a separate function so that we can call it from child classes
    """
    assert kernel in ['gauss_M', 'laplace_M']
    print('learning feature matrix...')
    rfm = get_rfm_regressor(kernel[:-2], alpha=alpha, sigma=sigma, iters=rfm_iters)
    rfm.fit(X, y, val_data=val_data)
    M = rfm._model.M.numpy(force=True) # learned feature matrix
    return M

class KernelRidgeBase(BaseEstimator):

    def __init__(self, kernel='laplace', alpha=1, sigma=1, postprocess=None, M=None):
        assert kernel in ['gauss', 'laplace', 'sobolev', 'gauss_M', 'laplace_M']

        self.kernel = kernel
        self.alpha = alpha
        self.sigma = sigma
        self.postprocess = postprocess

        self.M = M

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, multi_output=True)
        if self.postprocess:
            y = self._validate_targets(y)

        self.X_fit_, self.y_fit_ = X, y

        if self.kernel == 'gauss_M':
            self.M_ = get_feature_matrix(X, y, self.kernel, self.alpha, self.sigma) if self.M is None else self.M
            # print('M_gauss', M, self.M)
            K = gaussian_M(X, X, self.M_, self.sigma)
        elif self.kernel == 'laplace_M':
            self.M_ = get_feature_matrix(X, y, self.kernel, self.alpha, self.sigma) if self.M is None else self.M
            K = laplacian_M(X, X, self.M_, self.sigma)
        elif self.kernel == 'gauss':
            K = gauss(X, X, self.sigma)
        elif self.kernel == 'laplace':
            K = laplacian(X, X, self.sigma)
        elif self.kernel == 'sobolev':
            K = sobolev(X, X)
        else:
            raise ValueError(f'kernel={self.kernel} is not supported')
            
        # pytorch version (roughly 6x faster than numpy version)
        K_torch = torch.from_numpy(K).double()
        y_torch = torch.from_numpy(y).double()
        self.sol_ = torch.linalg.solve(
            K_torch + self.alpha * torch.eye(len(K), device=K_torch.device, dtype=K_torch.dtype), 
            y_torch
        ).numpy(force=True).T
        return K

    def predict(self, X):
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        if self.kernel == 'gauss_M':
            # print('76> M_gauss')
            # K = gauss(
            #     self.X_fit_, # (n, rank)
            #     X @ self.W.T, # (?, rank)
            #     self.sigma
            # )
            # print('M_', self.M_)
            K = gaussian_M(self.X_fit_, X, self.M_, self.sigma)
        elif self.kernel == 'laplace_M':
            # K = laplacian(
            #     self.X_fit_, # (n, rank)
            #     X @ self.W.T, # (?, rank)
            #     self.sigma
            # )
            K = laplacian_M(self.X_fit_, X, self.M_, self.sigma)
        elif self.kernel == 'gauss':
            K = gauss(self.X_fit_, X, self.sigma)
        elif self.kernel == 'laplace':
            K = laplacian(self.X_fit_, X, self.sigma)
        elif self.kernel == 'sobolev':
            K = sobolev(self.X_fit_, X)
        else:
            raise ValueError(f'kernel={self.kernel} is not supported')
        
        pred = (self.sol_ @ K).T
        
        if self.postprocess == 'round':
            pred = np.maximum(pred, 0)
            pred = np.minimum(pred, 1)
            return np.round(pred)
        
        elif self.postprocess == 'argmax':
            max_index = np.argmax(pred, axis=1)
            one_hot = np.zeros_like(pred)
            one_hot[np.arange(len(pred)), max_index] = 1
            return one_hot
        
        elif self.postprocess == 'threshold':
            # print('thresholding')
            # only for binary classification
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
    

class KernelRidgeRegressor(KernelRidgeBase, RegressorMixin):
    pass
class KernelRidgeClassifier(KernelRidgeBase, ClassifierMixin):
    pass


# class KernelRidgeLowRank(KernelRidgeBase):
#     """
#     Args:
#     - kernel, alpha, sigma: standard KernelRidge parameters
#         NOTE: use sigma=10 to avoid RFM numerical stability issues
#     - m (int | None): number of times to half the dataset
#         if None, then m = log4(n) (i.e., sqrt(n) coreset size)
#     """
    
#     def __init__(self, 
#                  kernel='laplace', 
#                  alpha=1, 
#                  sigma=1, 
#                  postprocess=None,
#                 ydim=1, store_K=True,
#                     # use_compress_gsn=False
#                     rfm_iters=1,
#                 ):
#         super().__init__(kernel=kernel, alpha=alpha, sigma=sigma, postprocess=postprocess)
        
#         # self.m = m
#         self.ydim = ydim
#         # self.use_compress_gsn = use_compress_gsn
#         self.store_K = store_K
#         self.rfm_iters = rfm_iters
#         self.M = None # learned feature matrix
        
#     def fit(self, X, y, val_data=None, rank=None):
#         # FEATURE LEARNING
#         if self.M is None:
#             print('learning feature matrix...')
#             self.rfm = get_rfm_regressor(self.kernel, alpha=self.alpha, sigma=self.sigma, iters=self.rfm_iters)
#             self.rfm.fit(X, y, val_data=val_data)
#             self.M = self.rfm._model.M.numpy(force=True) # learned feature matrix

#         # KERNEL THINNING
#         # var_k = self.sigma**2

#         # use the regular Gaussian and Laplacian kernels but with d=rank
#         # params_k_swap = {"name": self.kernel[2:], "var": var_k, "d": rank}
#         # params_k_split = {"name": self.kernel[2:], "var": var_k, "d": rank}
#         # convert M to low-rank
#         U, S, V = np.linalg.svd(self.M)
#         # compute dimensionality reduction matrix
#         self.W = np.diag(np.sqrt(S[:rank])) @ V[:rank] # (rank, d)

#         # first do dimensionality reduction
#         X_low = X @ self.W.T # (n, rank)
#         print(X_low.shape)
#         # X_ = get_Xy(X_low, y)

#         # split_kernel = get_kernel(params_k_split)
#         # swap_kernel = get_kernel(params_k_swap)
        
#         # assert (len(y.shape)==1 and self.ydim==1) or \
#         #     y.shape[1] == self.ydim, f"last dimension of y.shape={y.shape} doesn't match self.ydim={self.ydim}"
        
#         # split_kernel = to_regression_kernel(split_kernel, ydim=self.ydim)
#         # swap_kernel = to_regression_kernel(swap_kernel, ydim=self.ydim)

#         # kt_coreset = kt_thin2(X_, split_kernel, swap_kernel, 
#         #                         m=self.m, store_K=self.store_K)
   
#         # print(f'original size: {len(X)}, kt coreset size: {len(kt_coreset)}')

#         return super().fit(X_low, y)
#         # return super().fit(X_low[kt_coreset], y[kt_coreset])
    
#     def predict(self, X):
#         check_is_fitted(self)

#         # Input validation
#         X = check_array(X)

#         # project X into low rank subspace
#         if self.kernel == 'gauss':
#             K = gauss(self.X_fit_, X @ self.W.T, self.sigma)
#         elif self.kernel == 'laplace':
#             K = laplacian(self.X_fit_, X @ self.W.T, self.sigma)
#         else:
#             raise ValueError(f'kernel={self.kernel} is not supported')
        
#         pred = (self.sol_ @ K).T
        
#         if self.postprocess == 'round':
#             pred = np.maximum(pred, 0)
#             pred = np.minimum(pred, 1)
#             return np.round(pred)
        
#         elif self.postprocess == 'argmax':
#             max_index = np.argmax(pred, axis=1)
#             one_hot = np.zeros_like(pred)
#             one_hot[np.arange(len(pred)), max_index] = 1
#             return one_hot
        
#         elif self.postprocess == 'threshold':
#             # print('thresholding')
#             # only for binary classification
#             pred[pred > 0.5] = 1
#             pred[pred <= 0.5] = 0
#             # cast pred to int
#         else:
#             assert self.postprocess is None

#         return pred

# class KernelRidgeLowRankRegressor(KernelRidgeLowRank, RegressorMixin):
#     pass
# class KernelRidgeLowRankClasifier(KernelRidgeLowRank, ClassifierMixin):
#     pass
