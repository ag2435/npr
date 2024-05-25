from ..util_k import gaussian, laplace, gaussian_M, laplace_M, sobolev

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import torch
import numpy as np
# persistent cache
# from .. import memory

# @memory.cache
def get_feature_matrix(X, y, kernel, alpha=1, sigma=1, rfm_iters=1, val_data=None):
    """
    We put this as a separate function so that we can call it from child classes
    """
    from ..rfm import get_rfm_regressor

    assert kernel in ['gaussian_M', 'laplace_M']
    print('learning feature matrix...')
    rfm = get_rfm_regressor(kernel[:-2], alpha=alpha, sigma=sigma, iters=rfm_iters)
    rfm.fit(X, y, val_data=val_data)
    M = rfm._model.M.numpy(force=True) # learned feature matrix
    return M

class KernelRidgeBase(BaseEstimator):

    def __init__(self, kernel='laplace', alpha=1, sigma=1, postprocess=None, M=None, **kwargs):
        assert kernel in [
            'gaussian', 
            'laplace', 
            # 'sobolev', 
            # 'gaussian_M', 
            # 'laplace_M'
        ], f'kernel={kernel} is not supported'

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

        if self.kernel == 'gaussian_M':
            self.M_ = get_feature_matrix(X, y, self.kernel, self.alpha, self.sigma) if self.M is None else self.M
            # print('M_gauss', M, self.M)
            K = gaussian_M(X, X, self.M_, self.sigma)
        elif self.kernel == 'laplace_M':
            self.M_ = get_feature_matrix(X, y, self.kernel, self.alpha, self.sigma) if self.M is None else self.M
            K = laplace_M(X, X, self.M_, self.sigma)
        elif self.kernel == 'gaussian':
            K = gaussian(X, X, self.sigma)
        elif self.kernel == 'laplace':
            K = laplace(X, X, self.sigma)
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

        if self.kernel == 'gaussian_M':
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
            K = laplace_M(self.X_fit_, X, self.M_, self.sigma)
        elif self.kernel == 'gaussian':
            K = gaussian(self.X_fit_, X, self.sigma)
        elif self.kernel == 'laplace':
            K = laplace(self.X_fit_, X, self.sigma)
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
