from .rpcholesky import rpcholesky  
from .KRR_Nystrom import KRR_Nystrom
from ..util_thin import get_coreset_size, log4

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class KernelRidgeRPCholesky(BaseEstimator):
    def __init__(self, kernel='gaussian', alpha=1, sigma=1, m=None):
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
        
    def predict(self, X):
        return self.model_.predict_Nystrom(X)    
        # preds = model.predict_Nystrom(test_sample)
    
class KernelRidgeRPCholeskyRegressor(KernelRidgeRPCholesky, RegressorMixin):
    pass
class KernelRidgeRPCholeskyClassifier(KernelRidgeRPCholesky, ClassifierMixin):
    pass
