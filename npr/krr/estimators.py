from .base import KernelRidgeBase, get_feature_matrix
from ..util_thin import sd_thin
from ..util_sample import get_Xy
from .utils import get_feature_matrix

from goodpoints.compress import compresspp_kt

from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np
from time import time

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
                 verbose=0,
                 **kwargs):
        super().__init__(kernel=kernel, alpha=alpha, sigma=sigma, postprocess=postprocess, M=M)
        self.m = m
        self.use_dnc = use_dnc
        self.verbose = verbose
        
    def fit(self, X, y, **kwargs):
        if self.kernel in ['gaussian_M', 'laplace_M']:
            self.M = get_feature_matrix(X, y, kernel=self.kernel, alpha=self.alpha, sigma=self.sigma) if self.M is None else self.M

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
        start = time()
        coreset = self.thin(X,y)
        print(f'thin time: {time()-start:.4f} s')
        # print('coreset size', len(coreset))
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
        return coreset

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
                m=None,
                use_dnc=False,
                ydim=1,
                verbose=0, 
                ablation=0,
                **kwargs
                ):
        super().__init__(kernel=kernel, alpha=alpha, sigma=sigma, 
                         postprocess=postprocess, 
                         m=m, use_dnc=use_dnc, verbose=verbose)
        
        self.ydim = ydim
        self.ablation = ablation
        
    def thin(self, X, y):
        if self.m == 0:
            # Zero halving rounds requested
            # Return coreset containing all indices
            return np.arange(X.shape[0], dtype=int)
        if self.kernel == 'gaussian_M':
            assert self.ablation == 0, "ablation not supported for gaussian_M"
        if self.ablation == 0:
            # use special kernel:
            #   k(x1,x2)^2 + y1*y2 * k(x1,x2)
            if self.kernel == 'gaussian_M':
                coreset = compresspp_kt(
                    X=get_Xy(X, y),
                    kernel_type=f"loss_{self.kernel}".encode(),
                    k_params=np.concatenate([
                        np.array([self.sigma**2, 1], dtype=float),
                        self.M.flatten(),
                    ]),
                )
            else:
                coreset = compresspp_kt(
                    X=get_Xy(X, y),
                    kernel_type=f"loss_{self.kernel}".encode(),
                    k_params=np.array([self.sigma**2, 1], dtype=float),
                )
        elif self.ablation == 1:
            # use base kernel on concatenated vector:
            #   k((x1,x2), (y1,y2))
            coreset = compresspp_kt(
                X=get_Xy(X, y),
                kernel_type=self.kernel.encode(),
                k_params=np.array([2*self.sigma**2], dtype=float), # use 
            )
        elif self.ablation == 2:
            # use base kernel on X only:
            #   k(x1,x2)
            coreset = compresspp_kt(
                X=X,
                kernel_type=self.kernel.encode(),
                k_params=np.array([2*self.sigma**2], dtype=float),
            )
        elif self.ablation == 3:
            # use product kernel (from nadaraya-watson estimator):
            #   k(x1,x2) * (1 + y1*y2)
            assert self.kernel == 'gaussian', \
                f"only gaussian kernel supported for ablation 3, got {self.kernel}"
            coreset = compresspp_kt(
                X=get_Xy(X, y),
                kernel_type=b"prod_gaussian",
                k_params=np.array([self.sigma**2, 1], dtype=float),
            )
        return coreset

class KernelRidgeSTRegressor(KernelRidgeST, RegressorMixin):
    pass

class KernelRidgeKTRegressor(KernelRidgeKT, RegressorMixin):
    pass

class KernelRidgeSTClassifier(KernelRidgeST, ClassifierMixin):
    pass

class KernelRidgeKTClassifier(KernelRidgeKT, ClassifierMixin):
    pass
