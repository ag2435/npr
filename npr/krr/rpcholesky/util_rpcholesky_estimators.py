from .rpcholesky import rpcholesky  
from .KRR_Nystrom import KRR_Nystrom
from ..thin.util_thin import get_coreset_size, log4

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import torch
import numpy as np

class KernelRidgeRPCholesky(BaseEstimator):
    def __init__(self, kernel, alpha=1, sigma=1, m=None):
        self.kernel = kernel
        self.alpha = alpha
        self.sigma = sigma
        self.m = m

    def fit(self, X, y):
        kernel_names = {
            'gauss': 'gaussian',
            'laplace': 'laplace'
        }
        model = KRR_Nystrom(kernel =kernel_names[self.kernel], bandwidth = self.sigma)

        n = len(X)
        if self.m is None:
            m = int(log4(n))
        else:
            m = self.m

        k = get_coreset_size(n, m=m)

        model.fit_Nystrom(X, y, lamb = self.alpha, sample_num = k, sample_method = rpcholesky, solve_method = 'Direct')

        self.model_ = model
        
    def predict(self, X):
        return self.model_.predict_Nystrom(X)    
        # preds = model.predict_Nystrom(test_sample)
    
class KernelRidgeRPCholeskyRegressor(KernelRidgeRPCholesky, RegressorMixin):
    pass
class KernelRidgeRPCholeskyClassifier(KernelRidgeRPCholesky, ClassifierMixin):
    pass

def get_rpcholesky_regressor(kernel, alpha=1e-3, sigma=1, m=None):
    assert kernel in ['gauss', 'laplace']
    return KernelRidgeRPCholeskyRegressor(kernel, alpha=alpha, sigma=sigma, m=m)

def get_rpcholesky_classifier(kernel, alpha=1e-3, sigma=1, m=None):
    assert kernel in ['gauss', 'laplace']
    return KernelRidgeRPCholeskyClassifier(kernel, alpha=alpha, sigma=sigma, m=m)