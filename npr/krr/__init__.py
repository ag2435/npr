from .base import KernelRidgeRegressor, KernelRidgeClassifier
from .estimators import (KernelRidgeSTRegressor, KernelRidgeKTRegressor,
                         KernelRidgeSTClassifier, KernelRidgeKTClassifier)

def get_regressor(name, **kwargs):
    """
    Returns the estimator with the given name
    """
    if 'postprocess' in kwargs:
        assert kwargs['postprocess'] is None

    if name=='full':
        return KernelRidgeRegressor(**kwargs)
    
    elif name=='st':
        return KernelRidgeSTRegressor(**kwargs)
    
    elif name=='kt':
        return KernelRidgeKTRegressor(**kwargs)
    
    else:
        raise ValueError(f'{name} is not supported')
    
def get_classifier(name, **kwargs):
    """
    Returns the estimator with the given name
    """
    if name=='full':
        return KernelRidgeClassifier(**kwargs)
    
    elif name=='st':
        return KernelRidgeSTClassifier(**kwargs)
    
    elif name=='kt':
        return KernelRidgeKTClassifier(**kwargs)
    
    else:
        raise ValueError(f'{name} is not supported')
    
def get_estimator(task, name, **kwargs):
    """
    Args:
    - task [regression | classification]: determines score() method
    - name: name of estimator type
    - kernel: kernel type
    - use_dnc: whether to use divide-and-conquer estimator
    - kwargs: additional keyword arguments for the estimator
    """
    if task == 'regression':
        return get_regressor(name, **kwargs)
    elif task == 'classification':
        return get_classifier(name, **kwargs)
    else:
        raise ValueError(f'{task} is not supported')
