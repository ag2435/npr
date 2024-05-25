from .rpcholesky import rpcholesky  
from .KRR_Nystrom import KRR_Nystrom
from ..util_thin import get_coreset_size, log4

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np

# 
# %%%%%%%%%%%%%%%%%%%%%%%%%%% Implementation of RPCholesky KRR %%%%%%%%%%%%%%%%%%%%%%%%%%%
# 

class KernelRidgeRPCholesky(BaseEstimator):
    def __init__(self, kernel='gaussian', alpha=1, sigma=1, m=None, **kwargs):
        assert kernel in ['gaussian', 'laplace'], f"kernel {kernel} is not supported"
        self.kernel = kernel
        self.alpha = alpha
        self.sigma = sigma
        self.m = m

    def fit(self, X, y):
        # print(f'fitting rpcholesky with kernel={self.kernel}, sigma={self.sigma}, alpha={self.alpha}')
        model = KRR_Nystrom(kernel=self.kernel, bandwidth = self.sigma)

        n = len(X)
        if self.m is None:
            m = int(log4(n))
        else:
            m = self.m

        k = get_coreset_size(n, m=m)
        # print('coreset size:', coreset_size)

        model.fit_Nystrom(X, y, 
                          lamb = self.alpha, 
                          sample_num = k, 
                          sample_method = rpcholesky, 
                          solve_method = 'Direct')
        # print(len(model.sol))
        self.model_ = model
        # for plotting purposes
        self.X_fit_ = X[model.sample_idx]
        self.y_fit_ = y[model.sample_idx]
        
    def predict(self, X):
        return self.model_.predict_Nystrom(X)    
        # preds = model.predict_Nystrom(test_sample)
    
class KernelRidgeRPCholeskyRegressor(KernelRidgeRPCholesky, RegressorMixin):
    pass
class KernelRidgeRPCholeskyClassifier(KernelRidgeRPCholesky, ClassifierMixin):
    pass

# 
# %%%%%%%%%%%%%%%% Implementation of RPCholesky for Nadaraya-Watson %%%%%%%%%%%%%%%%
# 
from ..nw.estimators import NadarayaWatsonThin
from .matrix import KernelMatrix

class NadarayaWatsonRPCholesky(NadarayaWatsonThin):
    def thin(self, X, y):
        if self.m == 0:
            # Zero halving rounds requested
            # Return coreset containing all indices
            return np.arange(X.shape[0], dtype=int)
        
        n = len(X)
        if self.m is None:
            m = int(log4(n))
        else:
            m = self.m

        sample_num = get_coreset_size(n, m=m)
        # print('coreset size:', coreset_size)

        K = KernelMatrix(X, kernel = self.kernel, bandwidth = self.sigma)
        lra = rpcholesky(K, sample_num)
        arr_idx = lra.get_indices()
        return arr_idx

class NadarayaWatsonRPCholeskyRegressor(NadarayaWatsonRPCholesky, RegressorMixin):
    pass
class NadarayaWatsonRPCholeskyClassifier(NadarayaWatsonRPCholesky, ClassifierMixin):
    pass