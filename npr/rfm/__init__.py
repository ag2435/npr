from .util_rfm_estimators import (RFMRegressor, 
                                  RFMClassifier)

def get_rfm_regressor(kernel, alpha=1e-3, sigma=10, iters=1, ydim=1, use_kt=False, postprocess=None,
                      diag=False, centering=False,):
    assert kernel in ['gauss', 'laplace']
    return RFMRegressor(
        kernel, 
        sigma=sigma, 
        alpha=alpha, 
        iters=iters, 
        ydim=ydim,
        use_kt=use_kt,
        # postprocess=postprocess
        diag=diag, centering=centering,
    )

def get_rfm_classifier(kernel, alpha=1e-3, sigma=10, iters=1, ydim=1, use_kt=False, postprocess='threshold',
                       diag=False, centering=False,):
    assert kernel in ['gauss', 'laplace']
    return RFMClassifier(
        kernel, 
        sigma=sigma, 
        alpha=alpha, 
        iters=iters, 
        ydim=ydim,
        use_kt=use_kt,
        postprocess=postprocess,
        diag=diag, centering=centering,
    )
