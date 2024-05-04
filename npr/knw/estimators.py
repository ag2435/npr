from .base import KernelNadarayaWatsonBase
from ..krr.thin.util_thin import sd_thin
# from .util_thin_dnc import sd_thin_dnc, kt_thin2_dnc, kt_thin1_dnc
from ..krr.thin.util_k_mmd import (
    # kernel_eval, 
    to_product_kernel, 
    get_kernel, 
    # gauss, laplacian, euclidean_distances
)
from ..krr.util_sample import get_Xy
# from ..rfm2.util_rfm_estimators import get_rfm_regressor

# kernel thinning functionality
from ..compress import compresspp_kt

from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np

class KernelNadarayaWatsonThin(KernelNadarayaWatsonBase):
    """
    Args:
    - kernel, alpha, sigma: standard KernelRidge parameters
    - m (int | None): number of times to half the dataset
        if None, then m = log4(n) (i.e., sqrt(n) coreset size)
    """
    def __init__(self, kernel='epanechnikov', sigma=1, postprocess=None,
                 m=None,
                 use_dnc=False,
                 M=None,
                 verbose=0,):
        super().__init__(kernel=kernel, sigma=sigma, postprocess=postprocess, M=M)
        self.m = m
        self.use_dnc = use_dnc
        self.verbose = verbose
        
    def fit(self, X, y, **kwargs):
        # if self.kernel in ['gauss_M', 'laplace_M']:
        #     self.M = get_feature_matrix(X, y, kernel=self.kernel, alpha=self.alpha, sigma=self.sigma) if self.M is None else self.M

        # self.coresets_ = self.thin(X, y)
        # if self.use_dnc:
        #     # use all coresets (i.e., divide-and-conquer estimator)
        #     self.estimators_ = []
        #     for i, coreset in enumerate(self.coresets_):
        #         if self.verbose:
        #             print(f'fitting estimator {i} on coreset of size {len(coreset)}...')
        #         estimator = KernelRidgeBase(
        #             kernel=self.kernel, 
        #             alpha=self.alpha, 
        #             sigma=self.sigma, 
        #             postprocess=None,
        #             M=self.M,
        #         )
        #         estimator.fit(X[coreset], y[coreset])
        #         self.estimators_.append(estimator)
        # else:
        # use one coreset
        coreset, coreset2 = self.thin(X,y)
        super().fit(X[coreset], y[coreset], X2=X[coreset2])

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

class KernelNadarayaWatsonST(KernelNadarayaWatsonThin):
    """
    Args:
    - kernel, alpha, sigma: standard KernelRidge parameters
    - m (int | None): number of times to half the dataset
        if None, then m = log4(n) (i.e., sqrt(n) coreset size)
    """

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
            return np.arange(X.shape[0], dtype=int)
        
        # return sd_thin_dnc(X, m=self.m, verbose=self.verbose)
        coreset = sd_thin(X, m=self.m,)
        return coreset, coreset

class KernelNadarayaWatsonKT(KernelNadarayaWatsonThin):
    """
    Args:
    - kernel, alpha, sigma: standard KernelRidge parameters
    - seed (int; default 123) seed for experiment
    - d (int; default 1): dimension of the points
    - m (int | None): number of times to half the dataset
        if None, then m = log4(n) (i.e., sqrt(n) coreset size)
    """
    
    def __init__(self, 
                 kernel='epanechnikov', 
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
        super().__init__(kernel=kernel, sigma=sigma, 
                         postprocess=postprocess, 
                         m=m, use_dnc=use_dnc, verbose=verbose)
        
        # self.m = m
        self.ydim = ydim
        # self.use_compress_gsn = use_compress_gsn
        self.store_K = store_K
        # self.use_dnc = use_dnc
        self.use_compresspp = use_compresspp

    def thin(self, X, y):
        if self.m == 0:
            # Zero halving rounds requested
            # Return coreset containing all indices
            return np.arange(X.shape[0], dtype=int)
        
        # d = X.shape[-1]
        # var_k = self.sigma**2

        # params_k_swap = {"name": self.kernel, "var": var_k, "d": int(d), "M":self.M}
        # params_k_split = {"name": self.kernel, "var": var_k, "d": int(d), "M":self.M}        
        # split_kernel = get_kernel(params_k_split)
        # swap_kernel = get_kernel(params_k_swap)

        # coreset2 = kt_thin2(X, split_kernel, swap_kernel, m=self.m, store_K=self.store_K)
        
        # assert (len(y.shape)==1 and self.ydim==1) or \
        #     y.shape[1] == self.ydim, f"last dimension of y.shape={y.shape} doesn't match self.ydim={self.ydim}"
        
        # coreset = kt_thin2(
        #     X=get_Xy(X, y), 
        #     split_kernel=to_product_kernel(split_kernel, ydim=self.ydim),
        #     swap_kernel=to_product_kernel(swap_kernel, ydim=self.ydim),
        #     m=self.m, 
        #     store_K=self.store_K
        # )
        
        # return coreset, coreset2

        # 
        # Using new compresspp_kt function
        # 
        kernel_type = self.kernel.encode() # e.g., "gaussian" -> b"gaussian"
        coreset1 = compresspp_kt(
            X=get_Xy(X, y),
            kernel_type=kernel_type, # e.g., "gaussian" -> b"gaussian"
            k_params=np.array([self.sigma**2, 1]), # use product kernel
        )
        coreset2 = compresspp_kt(
            X=X,
            kernel_type=kernel_type, # e.g., "gaussian" -> b"gaussian"
            k_params=np.array([self.sigma**2, 0]), # use base kernel
        )
        return coreset1, coreset2

class KernelNadarayaWatsonSTRegressor(KernelNadarayaWatsonST, RegressorMixin):
    pass

class KernelNadarayaWatsonKTRegressor(KernelNadarayaWatsonKT, RegressorMixin):
    pass

# class KernelRidgeKTFeatureRegressor(KernelRidgeKTFeature, RegressorMixin):
#     pass

class KernelNadarayaWatsonSTClassifier(KernelNadarayaWatsonST, ClassifierMixin):
    pass

class KernelNadarayaWatsonKTClassifier(KernelNadarayaWatsonKT, ClassifierMixin):
    pass

# class KernelRidgeKTFeatureClassifier(KernelRidgeKTFeature, ClassifierMixin):
#     pass
