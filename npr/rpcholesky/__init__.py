"""
Source: https://github.com/eepperly/Randomly-Pivoted-Cholesky
"""

from .estimators import (KernelRidgeRPCholeskyRegressor,
                         KernelRidgeRPCholeskyClassifier)

def get_estimator(task, name, **kwargs):
    """
    Args:
    - task [regression | classification]: determines score() method
    - name: name of thinning method (note: unused in this implementation)
    - kernel: kernel type
    - use_dnc: whether to use divide-and-conquer estimator
    - kwargs: additional keyword arguments for the estimator
    """
    if task == 'regression':
        return KernelRidgeRPCholeskyRegressor(**kwargs)
    elif task == 'classification':
        return KernelRidgeRPCholeskyClassifier(**kwargs)
    else:
        raise ValueError(f'{task} is not supported')
