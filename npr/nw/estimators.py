from .base import NadarayaWatsonBase
from ..util_thin import sd_thin
from ..util_sample import get_Xy
# from ..rfm2.util_rfm_estimators import get_rfm_regressor

from goodpoints.compress import compresspp_kt

from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np

class NadarayaWatsonThin(NadarayaWatsonBase):
    """
    Args:
        kernel, alpha, sigma: standard KernelRidge parameters
        m (int | None): number of times to half the dataset
            if None, then m = log4(n) (i.e., sqrt(n) coreset size)
    """
    def __init__(self, kernel='epanechnikov', sigma=1, postprocess=None,
                 m=None,
                 use_dnc=False,
                 M=None,
                 verbose=0, 
                 **kwargs):
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
        coreset = self.thin(X,y)
        # print('coreset size:', len(coreset))
        super().fit(X[coreset], y[coreset], X2=X[coreset])

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

class NadarayaWatsonST(NadarayaWatsonThin):
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

class NadarayaWatsonKT(NadarayaWatsonThin):
    """
    Args:
        kernel: kernel type
        sigma: bandwidth
        m (int | None): number of times to half the dataset
            if None, then m = log4(n) (i.e., sqrt(n) coreset size)
        ablation (int): if 1, then use concatenated vectors, 
            if 2, then use only the x vector
    """
    
    def __init__(self, 
                 kernel='epanechnikov', 
                 sigma=1, 
                 postprocess=None,
                 m=None,
                use_dnc=False,
                ydim=1,
                verbose=0, 
                ablation=0,
                **kwargs
                ):
        super().__init__(kernel=kernel, sigma=sigma, 
                         postprocess=postprocess, 
                         m=m, use_dnc=use_dnc, verbose=verbose)
        
        self.ydim = ydim
        self.ablation = ablation

    def thin(self, X, y):
        if self.m == 0:
            # Zero halving rounds requested
            # Return coreset containing all indices
            return np.arange(X.shape[0], dtype=int)
        
        kernel_type = self.kernel.encode() # e.g., "gaussian" -> b"gaussian"
        if self.ablation == 0:
            # use special kernel:
            #   k(x1,x2) * (1+ y1*y2)
            coreset = compresspp_kt(
                X=get_Xy(X, y),
                kernel_type=kernel_type,
                k_params=np.array([self.sigma**2, 1], dtype=float), # use product kernel
            )
        elif self.ablation == 1:
            # use base kernel on concatenated vector:
            #   k((x1,x2), (y1,y2))
            coreset = compresspp_kt(
                X=get_Xy(X, y),
                kernel_type=kernel_type,
                k_params=np.array([self.sigma**2, 0], dtype=float), # use product kernel
            )
        elif self.ablation == 2:
            # use base kernel on X only:
            #   k(x1,x2)
            coreset = compresspp_kt(
                X=X,
                kernel_type=kernel_type,
                k_params=np.array([self.sigma**2, 0], dtype=float), # use product kernel
            )
        else:
            raise ValueError(f"ablation {self.ablation} is not supported")
        
        return coreset

class NadarayaWatsonSTRegressor(NadarayaWatsonST, RegressorMixin):
    pass

class NadarayaWatsonKTRegressor(NadarayaWatsonKT, RegressorMixin):
    pass

class NadarayaWatsonSTClassifier(NadarayaWatsonST, ClassifierMixin):
    pass

class NadarayaWatsonKTClassifier(NadarayaWatsonKT, ClassifierMixin):
    pass