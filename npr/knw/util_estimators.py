from .base import KernelNadarayaWatsonRegressor, KernelNadarayaWatsonClassifier
from .estimators import (
    KernelNadarayaWatsonSTRegressor, KernelNadarayaWatsonSTClassifier,
    KernelNadarayaWatsonKTRegressor, KernelNadarayaWatsonKTClassifier,
)

def get_regressor(name, kernel, **kwargs):
    """
    Returns the estimator with the given name
    """
    if 'postprocess' in kwargs:
        assert kwargs['postprocess'] is None

    if 'full' in name:
        return KernelNadarayaWatsonRegressor(kernel=kernel, **kwargs)
    
    elif 'st' in name:
        return KernelNadarayaWatsonSTRegressor(kernel=kernel, **kwargs)
    
    elif 'kt' in name:
        return KernelNadarayaWatsonKTRegressor(kernel=kernel, **kwargs)
    
    # elif 'falkon' in name:
    #     try:
    #         from goodpoints.krr.falkon.util_falkon_estimators import get_regressor as get_falkon_regressor
    #         return get_falkon_regressor(kernel, 'kt' in name)(**kwargs)
    #     except (ImportError, AttributeError) as e:
    #         print(e)
    #         return None
        
    # elif 'rfm' in name:
    #     return get_rfm_regressor(kernel, **kwargs)
    
    # elif 'rpcholesky' in name:
    #     return get_rpcholesky_regressor(kernel, **kwargs)
        
    else:
        raise ValueError(f'{name} is not supported')
    
def get_classifier(name, kernel, **kwargs):
    """
    Returns the estimator with the given name
    """
    if 'full' in name:
        return KernelNadarayaWatsonClassifier(kernel=kernel, **kwargs)
    
    elif 'st' in name:
        return KernelNadarayaWatsonSTClassifier(kernel=kernel, **kwargs)
    
    elif 'kt' in name:
        return KernelNadarayaWatsonKTClassifier(kernel=kernel, **kwargs)
    
    # elif 'falkon' in name:
    #     try:
    #         from goodpoints.krr.falkon.util_falkon_estimators import get_classifier as get_falkon_classifier
    #         return get_falkon_classifier(kernel, 'kt' in name)(**kwargs)
    #     except (ImportError, AttributeError) as e:
    #         print(e)
    #         return None
        
    # elif 'rfm' in name:
    #     return get_rfm_classifier(kernel, **kwargs)
    
    # elif 'rpcholesky' in name:
    #     return get_rpcholesky_classifier(kernel, **kwargs)
        
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