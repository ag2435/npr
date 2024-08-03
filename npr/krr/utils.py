# persistent cache
from joblib import Memory

location = './cachedir'
memory = Memory(location, verbose=0)

# @memory.cache
def get_feature_matrix(X, y, kernel, alpha=1, sigma=1, rfm_iters=3, val_data=None):
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