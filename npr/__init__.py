def estimator_factory(task, method, thin, **kwargs):
    """
    Get estimator given method and thin

    Args:
        task (str): 'regression' or 'classification'
        method (str): 'nw' or 'krr'
        thin (str): thinning method (e.g., 'full', 'st', 'kt')
        **kwargs: additional keyword arguments that will be passed to the estimator
            e.g., kernel, sigma, and alpha (in the case of krr)
    """
    assert task in ['regression', 'classification']
    if method == 'nw':
        from .nw import get_estimator
        return get_estimator(task, thin, **kwargs)
    elif method == 'krr':
        if thin in ['full', 'st', 'kt']:
            from .krr import get_estimator
            return get_estimator(task, thin, **kwargs)
        elif thin == 'rpcholesky':
            from .rpcholesky import get_estimator
            return get_estimator(task, thin, **kwargs)
        else:
            raise ValueError(f'{thin} is not supported for method {method}')
    else:
        raise ValueError(f'{method} is not supported')