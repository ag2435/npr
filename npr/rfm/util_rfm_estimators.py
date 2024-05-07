"""
sklearn wrapper for RFM (defined in goodpoints/krr/rfm2/recursive_feature_machine.py)
"""

from .recursive_feature_machine import GaussRFM, LaplaceRFM

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import torch
import numpy as np

class RFM(BaseEstimator):
    def __init__(self, kernel, alpha=1, sigma=1, 
                 iters=1, postprocess=None, ydim=1, 
                 use_kt=False, diag=False, centering=False,):
        self.kernel = kernel
        self.alpha = alpha
        self.sigma = sigma
        self.iters = iters
        self.postprocess = postprocess
        self.ydim = ydim
        self.use_kt = use_kt
        self.diag = diag
        self.centering = centering

    def fit(self, X, y, val_data=None):
        if self.postprocess:
            y = self._validate_targets(y)

        # NOTE: it's important to use double precision
        X_torch = torch.from_numpy(X).double()
        y_torch = torch.from_numpy(y.reshape(-1, self.ydim))

        if val_data is None:
            X_val_torch = X_torch
            y_val_torch = y_torch
        else:
            X_val, y_val = val_data
            X_val_torch = torch.from_numpy(X_val).double()
            y_val_torch = torch.from_numpy(y_val.reshape(-1, self.ydim)).double()

        if self.kernel == 'gauss':
            model = GaussRFM(
                reg=self.alpha, 
                bandwidth=self.sigma, 
                iters=self.iters, 
                use_kt=self.use_kt,
                diag=self.diag,
                centering=self.centering,
            )
        elif self.kernel == 'laplace':
            model = LaplaceRFM(
                reg=self.alpha, 
                bandwidth=self.sigma, 
                iters=self.iters, 
                use_kt=self.use_kt,
                diag=self.diag,
                centering=self.centering,
            )
        else:
            raise ValueError(f'kernel={self.kernel} not implemented')
        
        rfm_result = model.fit(
            (X_torch, y_torch), 
            (X_val_torch, y_val_torch), 
            loader=False, 
            method='lstsq', 
            print_every=5,
            return_mse=True
        )

        self._model = model

        return rfm_result
    
    def predict(self, X):
        X_torch = torch.from_numpy(X).double()
        pred = self._model.predict(X_torch).numpy(force=True)

        if self.postprocess == 'threshold':
            if self.ydim == 1:
                pred[pred > 0.5] = 1
                pred[pred <= 0.5] = 0
            else:
                pred = pred.argmax(axis=-1)

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
    
class RFMRegressor(RFM, RegressorMixin):
    pass
class RFMClassifier(RFM, ClassifierMixin):
    pass
