from .base import KernelRidgeBase, get_feature_matrix
# from .util_thin import sd_thin, kt_thin2
from .util_thin_dnc import sd_thin_dnc, kt_thin2_dnc, kt_thin1_dnc
from .util_k_mmd import kernel_eval, to_regression_kernel, get_kernel, gauss, laplacian, euclidean_distances
from ..util_sample import get_Xy
from ..rfm2.util_rfm_estimators import get_rfm_regressor

from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np

class KernelRidgeThin(KernelRidgeBase):
    """
    Args:
    - kernel, alpha, sigma: standard KernelRidge parameters
    - m (int | None): number of times to half the dataset
        if None, then m = log4(n) (i.e., sqrt(n) coreset size)
    """
    def __init__(self, kernel='laplace', alpha=1, sigma=1, postprocess=None,
                 m=None,
                 use_dnc=False,
                 M=None,
                 verbose=0,):
        super().__init__(kernel=kernel, alpha=alpha, sigma=sigma, postprocess=postprocess, M=M)
        self.m = m
        self.use_dnc = use_dnc
        self.verbose = verbose
        
    def fit(self, X, y, **kwargs):
        if self.kernel in ['gauss_M', 'laplace_M']:
            self.M = get_feature_matrix(X, y, kernel=self.kernel, alpha=self.alpha, sigma=self.sigma) if self.M is None else self.M

        self.coresets_ = self.thin(X, y)
        if self.use_dnc:
            # use all coresets (i.e., divide-and-conquer estimator)
            self.estimators_ = []
            for i, coreset in enumerate(self.coresets_):
                if self.verbose:
                    print(f'fitting estimator {i} on coreset of size {len(coreset)}...')
                estimator = KernelRidgeBase(
                    kernel=self.kernel, 
                    alpha=self.alpha, 
                    sigma=self.sigma, 
                    postprocess=None,
                    M=self.M,
                )
                estimator.fit(X[coreset], y[coreset])
                self.estimators_.append(estimator)
        else:
            # use one coreset
            coreset = self.coresets_[0]
            super().fit(X[coreset], y[coreset])

    def predict(self, X, return_all=False):
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        if self.use_dnc:
            # use all coresets (i.e., divide-and-conquer estimator)
            preds = [estimator.predict(X) for estimator in self.estimators_] # list of predictions            
            pred = np.mean(preds, axis=0)
            
        else:
            pred = super().predict(X)
            preds = [pred,]

        if return_all:
            return pred, preds
        
        return pred

    def thin(self, X, y):
        """
        Args:
        - X, y: dataset

        Returns:
        - list of coresets
        """
        raise NotImplementedError

class KernelRidgeST(KernelRidgeThin):
    """
    Args:
    - kernel, alpha, sigma: standard KernelRidge parameters
    - m (int | None): number of times to half the dataset
        if None, then m = log4(n) (i.e., sqrt(n) coreset size)
    """
    # def __init__(self, m=None, kernel='laplace', alpha=1, sigma=1, postprocess=None,
    #              use_dnc=False):
    #     super().__init__(kernel=kernel, alpha=alpha, sigma=sigma, postprocess=postprocess)
    #     self.m = m
    #     self.use_dnc = use_dnc
        
    # def fit(self, X, y, **kwargs):
    #     # STANDARD THINNING

    #     # Check that X and y have correct shape
    #     # X, y = check_X_y(X, y,)

    #     # with TicToc('sd coreset', print_toc=PRINT_TOC):
    #     sd_coreset = sd_thin(X, m=self.m)
    #     # print(f'original size: {len(X)}, sd coreset size: {len(sd_coreset)}')

    #     # with TicToc('fit', print_toc=PRINT_TOC):
    #     super().fit(X[sd_coreset], y[sd_coreset])
        
    def thin(self, X, y):
        """
        Args:
        - X, y: dataset

        Returns:
        - list of coresets
        """
        if self.m == 0:
            # Zero halving rounds requested
            # Return coreset containing all indices
            return [ np.arange(X.shape[0], dtype=int), ]
        
        return sd_thin_dnc(X, m=self.m, verbose=self.verbose)

class KernelRidgeKT(KernelRidgeThin):
    """
    Args:
    - kernel, alpha, sigma: standard KernelRidge parameters
    - seed (int; default 123) seed for experiment
    - d (int; default 1): dimension of the points
    - m (int | None): number of times to half the dataset
        if None, then m = log4(n) (i.e., sqrt(n) coreset size)
    """
    
    def __init__(self, 
                 kernel='laplace', 
                 alpha=1, 
                 sigma=1, 
                 postprocess=None,
                # normalize_y=False, 
                    # use_compress_gsn=False
                m=None,
                use_dnc=False,
                ydim=1, store_K=True,
                use_compresspp=True,
                verbose=0, 
                ):
        super().__init__(kernel=kernel, alpha=alpha, sigma=sigma, 
                         postprocess=postprocess, 
                         m=m, use_dnc=use_dnc, verbose=verbose)
        
        # self.m = m
        self.ydim = ydim
        # self.use_compress_gsn = use_compress_gsn
        self.store_K = store_K
        # self.use_dnc = use_dnc
        self.use_compresspp = use_compresspp

    # def fit(self, X, y, **kwargs):
    #     # KERNEL THINNING
        
    #     # Check that X and y have correct shape
    #     # X, y = check_X_y(X, y)

    #     d = X.shape[-1]
    #     var_k = self.sigma**2

    #     params_k_swap = {"name": self.kernel, "var": var_k, "d": int(d)}
    #     params_k_split = {"name": self.kernel, "var": var_k, "d": int(d)}
        
    #     # split_kernel = partial(kernel_eval, params_k=params_k_split)
    #     # swap_kernel = partial(kernel_eval, params_k=params_k_swap)
    #     split_kernel = get_kernel(params_k_split)
    #     swap_kernel = get_kernel(params_k_swap)
        
    #     assert (len(y.shape)==1 and self.ydim==1) or \
    #         y.shape[1] == self.ydim, f"last dimension of y.shape={y.shape} doesn't match self.ydim={self.ydim}"
        
    #     split_kernel = to_regression_kernel(split_kernel, ydim=self.ydim)
    #     swap_kernel = to_regression_kernel(swap_kernel, ydim=self.ydim)

    #     X_ = get_Xy(X, y)

    #     # if self.use_compress_gsn:
    #     #     assert self.kernel == 'gauss', 'compress_gsn_kt not compatible with non-gaussian kernels'
    #     #     kt_coreset = kt_thin3(X, split_kernel, swap_kernel, 
    #     #                           seed=self.seed, m=self.m, store_K=self.store_K, 
    #     #                           var_k=var_k)
    #     # else:
    #     kt_coreset = kt_thin2(X_, split_kernel, swap_kernel, 
    #                             m=self.m, store_K=self.store_K)
   
    #     # print(f'original size: {len(X)}, kt coreset size: {len(kt_coreset)}')

    #     super().fit(X[kt_coreset], y[kt_coreset])
        
    def thin(self, X, y):
        if self.m == 0:
            # Zero halving rounds requested
            # Return coreset containing all indices
            return [ np.arange(X.shape[0], dtype=int), ]
        
        d = X.shape[-1]
        var_k = self.sigma**2

        params_k_swap = {"name": self.kernel, "var": var_k, "d": int(d), "M":self.M}
        params_k_split = {"name": self.kernel, "var": var_k, "d": int(d), "M":self.M}        
        split_kernel = get_kernel(params_k_split)
        swap_kernel = get_kernel(params_k_swap)
        
        assert (len(y.shape)==1 and self.ydim==1) or \
            y.shape[1] == self.ydim, f"last dimension of y.shape={y.shape} doesn't match self.ydim={self.ydim}"
        
        split_kernel = to_regression_kernel(split_kernel, ydim=self.ydim)
        swap_kernel = to_regression_kernel(swap_kernel, ydim=self.ydim)

        X_ = get_Xy(X, y)

        if self.use_compresspp:
            return kt_thin2_dnc(X_, split_kernel, m=self.m, store_K=self.store_K)
        else:
            return kt_thin1_dnc(X_, split_kernel, m=self.m, store_K=self.store_K)


# class KernelRidgeKTFeature(KernelRidgeThin):
#     """
#     Args:
#     - kernel, alpha, sigma: standard KernelRidge parameters
#         NOTE: use sigma=10 to avoid RFM numerical stability issues
#     - m (int | None): number of times to half the dataset
#         if None, then m = log4(n) (i.e., sqrt(n) coreset size)
#     """
    
#     def __init__(self, 
#                  kernel='M_laplace', 
#                  alpha=1, 
#                  sigma=10, 
#                  postprocess=None,
#                 ydim=1, store_K=True,
#                     # use_compress_gsn=False
#                     rfm_iters=1,
#                  m=None,
#                 use_dnc=False,
#                 ):
#         super().__init__(kernel=kernel, alpha=alpha, sigma=sigma, postprocess=postprocess,
#                          m=m, use_dnc=use_dnc)
        
#         # self.m = m
#         self.ydim = ydim
#         # self.use_compress_gsn = use_compress_gsn
#         self.store_K = store_K
#         self.rfm_iters = rfm_iters
#         self.M = None # learned feature matrix
#         # self.use_dnc = use_dnc
        
#     def fit(self, X, y, val_data=None, rank=None):
#         # FEATURE LEARNING
#         if self.M is None:
#             print('learning feature matrix...')
#             self.rfm = get_rfm_regressor(self.kernel[2:], alpha=self.alpha, sigma=self.sigma, iters=self.rfm_iters)
#             self.rfm.fit(X, y, val_data=val_data)
#             self.M = self.rfm._model.M.numpy(force=True) # learned feature matrix

#         # KERNEL THINNING
#         var_k = self.sigma**2

#         # if rank is None:
#         d = X.shape[-1]
#         params_k_swap = {"name": self.kernel, "var": var_k, "d": int(d), "M":self.M}
#         params_k_split = {"name": self.kernel, "var": var_k, "d": int(d), "M":self.M}
#         X_ = get_Xy(X, y)

#         # else:
#         #     # use the regular Gaussian and Laplacian kernels but with d=rank
#         #     params_k_swap = {"name": self.kernel[2:], "var": var_k, "d": rank}
#         #     params_k_split = {"name": self.kernel[2:], "var": var_k, "d": rank}
#         #     # convert M to low-rank
#         #     U, S, V = np.linalg.svd(self.M)
#         #     # compute dimensionality reduction matrix
#         #     self.W = np.diag(np.sqrt(S[:rank])) @ V[:rank] # (rank, d)

#         #     # first do dimensionality reduction
#         #     X_low = X @ self.W.T # (n, rank)
#         #     print(X_low.shape)
#         #     X_ = get_Xy(X_low, y)

#         split_kernel = get_kernel(params_k_split)
#         swap_kernel = get_kernel(params_k_swap)
        
#         assert (len(y.shape)==1 and self.ydim==1) or \
#             y.shape[1] == self.ydim, f"last dimension of y.shape={y.shape} doesn't match self.ydim={self.ydim}"
        
#         split_kernel = to_regression_kernel(split_kernel, ydim=self.ydim)
#         swap_kernel = to_regression_kernel(swap_kernel, ydim=self.ydim)

#         kt_coreset = kt_thin2(X_, split_kernel, swap_kernel, 
#                                 m=self.m, store_K=self.store_K)
   
#         # print(f'original size: {len(X)}, kt coreset size: {len(kt_coreset)}')

#         return super().fit(X[kt_coreset], y[kt_coreset])
#         # return super().fit(X_low[kt_coreset], y[kt_coreset])
    


class KernelRidgeSTRegressor(KernelRidgeST, RegressorMixin):
    pass

class KernelRidgeKTRegressor(KernelRidgeKT, RegressorMixin):
    pass

# class KernelRidgeKTFeatureRegressor(KernelRidgeKTFeature, RegressorMixin):
#     pass

class KernelRidgeSTClassifier(KernelRidgeST, ClassifierMixin):
    pass

class KernelRidgeKTClassifier(KernelRidgeKT, ClassifierMixin):
    pass

# class KernelRidgeKTFeatureClassifier(KernelRidgeKTFeature, ClassifierMixin):
#     pass
