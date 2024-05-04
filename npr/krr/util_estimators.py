'''
File containing helper functions for details about kernel ridge regression models
(following the sklearn estimator API)
'''

from .thin.util_k_mmd import euclidean_distances

from .thin.base import KernelRidgeRegressor, KernelRidgeClassifier
from .thin.estimators import KernelRidgeSTRegressor, KernelRidgeKTRegressor #, KernelRidgeKTFeatureRegressor
from .thin.estimators import KernelRidgeSTClassifier, KernelRidgeKTClassifier #, KernelRidgeKTFeatureClassifier
from .rpcholesky.util_rpcholesky_estimators import get_rpcholesky_regressor, get_rpcholesky_classifier
from .rfm2.util_rfm_estimators import get_rfm_regressor, get_rfm_classifier

import numpy as np

def get_regressor(name, kernel, **kwargs):
    """
    Returns the estimator with the given name
    """
    if 'postprocess' in kwargs:
        assert kwargs['postprocess'] is None

    if 'full' in name:
        return KernelRidgeRegressor(kernel=kernel, **kwargs)
    
    elif 'st' in name:
        return KernelRidgeSTRegressor(kernel=kernel, **kwargs)
    
    elif 'kt' in name:
        return KernelRidgeKTRegressor(kernel=kernel, **kwargs)
    
    # elif 'kf' in name:
    #     return KernelRidgeKTFeatureRegressor(kernel='M_' + kernel, **kwargs)
    
    elif 'falkon' in name:
        try:
            from goodpoints.krr.falkon.util_falkon_estimators import get_regressor as get_falkon_regressor
            return get_falkon_regressor(kernel, 'kt' in name)(**kwargs)
        except (ImportError, AttributeError) as e:
            print(e)
            return None
        
    elif 'rfm' in name:
        return get_rfm_regressor(kernel, **kwargs)
    
    # elif 'lowrank' in name:
    #     return KernelRidgeLowRankRegressor(kernel=kernel, **kwargs)
    
    elif 'rpcholesky' in name:
        return get_rpcholesky_regressor(kernel, **kwargs)
        
    else:
        raise ValueError(f'{name} is not supported')
    
def get_classifier(name, kernel, **kwargs):
    """
    Returns the estimator with the given name
    """
    if 'full' in name:
        return KernelRidgeClassifier(kernel=kernel, **kwargs)
    
    elif 'st' in name:
        return KernelRidgeSTClassifier(kernel=kernel, **kwargs)
    
    elif 'kt' in name:
        return KernelRidgeKTClassifier(kernel=kernel, **kwargs)
    
    # elif 'kf' in name:
    #     return KernelRidgeKTFeatureClassifier(kernel='M_' + kernel, **kwargs)
    
    elif 'falkon' in name:
        try:
            from goodpoints.krr.falkon.util_falkon_estimators import get_classifier as get_falkon_classifier
            return get_falkon_classifier(kernel, 'kt' in name)(**kwargs)
        except (ImportError, AttributeError) as e:
            print(e)
            return None
        
    elif 'rfm' in name:
        return get_rfm_classifier(kernel, **kwargs)
    
    # elif 'lowrank' in name:
    #     return KernelRidgeLowRankClasifier(kernel=kernel, **kwargs)
    
    elif 'rpcholesky' in name:
        return get_rpcholesky_classifier(kernel, **kwargs)
        
    else:
        raise ValueError(f'{name} is not supported')
    
def get_estimator(task, name, kernel, **kwargs):
    """
    Args:
    - task [regression | classification]: determines score() method
    - name: name of estimator type
    - kernel: kernel type
    - use_dnc: whether to use divide-and-conquer estimator
    - kwargs: additional keyword arguments for the estimator
    """

    if task == 'regression':
        return get_regressor(name, kernel, **kwargs)
    elif task == 'classification':
        return get_classifier(name, kernel, **kwargs)
    else:
        raise ValueError(f'{task} is not supported')
    
def get_sigma_heuristic(X, sample_size=100, return_dist=False):
    """
    Median heuristic for choosing bandwidth param
    """

    X_sample = X[np.random.choice(len(X), size=sample_size, replace=False)]
    Y_sample = X[np.random.choice(len(X), size=sample_size, replace=False)]

    dist = euclidean_distances(X_sample, Y_sample, squared=False).flatten()
    sigma = np.median(dist)

    if return_dist:
        return sigma, dist
    
    return sigma

# ############ DnC Estimator ############

# class DnC(BaseEstimator):
#     def __init__(self, ) -> None:
#         super().__init__()

#     def fit(self, X, y):
#         pass

#     def predict(self, X):
#         pass

# def get_dnc_estimator(partitions, task, name, kernel, **kwargs):
#     """
#     Returns divide-and-conquer estimator with `partitions` number of partitions
#     """
