from .base import NadarayaWatsonRegressor, NadarayaWatsonClassifier
from .estimators import (NadarayaWatsonSTRegressor, 
                         NadarayaWatsonSTClassifier,
                         NadarayaWatsonKTRegressor, 
                         NadarayaWatsonKTClassifier)

def get_regressor(name, kernel, **kwargs):
    """
    Returns the estimator with the given name
    """
    if 'postprocess' in kwargs:
        assert kwargs['postprocess'] is None

    if name=='full':
        return NadarayaWatsonRegressor(kernel=kernel, **kwargs)
    
    elif name=='st':
        return NadarayaWatsonSTRegressor(kernel=kernel, **kwargs)
    
    elif name=='kt':
        return NadarayaWatsonKTRegressor(kernel=kernel, **kwargs)
    
    elif name=='rpcholesky':
        from .estimators import NadarayaWatsonRPCholeskyRegressor
        return NadarayaWatsonRPCholeskyRegressor(kernel=kernel, **kwargs)
    
    else:
        raise ValueError(f'{name} is not supported')
    
def get_classifier(name, kernel, **kwargs):
    """
    Returns the estimator with the given name
    """
    if name=='full':
        return NadarayaWatsonClassifier(kernel=kernel, **kwargs)
    
    elif name=='st':
        return NadarayaWatsonSTClassifier(kernel=kernel, **kwargs)
    
    elif name=='kt':
        return NadarayaWatsonKTClassifier(kernel=kernel, **kwargs)
    
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