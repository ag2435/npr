# RPCholesky Bugs

In the implementation of `kernels.py::GaussianKernel` (https://github.com/eepperly/Randomly-Pivoted-Cholesky/blob/794eba4a569124572e0b092bc15fcc96443ee52e/kernels.py#L29), should it be `np.linalg.norm(...)` instead of `np.linalg.norm(...)**2`?

Errors:
```
logn=14: model=KernelRidgeRPCholeskyRegressor()
/home/ag2435/kt_regression/src/npr/npr/rpcholesky/KRR_Nystrom.py:54: LinAlgWarning: Ill-conditioned matrix (rcond=1.0968e-16): result may not be accurate.
  self.sol = scipy.linalg.solve(KMn @ KnM + KnM.shape[0]*lamb*KMM + 100*KMM.max()*np.finfo(float).eps*np.identity(sample_num), KMn @ Ytr, assume_a='pos')
```
```
83 fits failed out of a total of 1000.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
83 fits failed with the following error:
Traceback (most recent call last):
  File "/home/ag2435/.conda/envs/npr/lib/python3.10/site-packages/sklearn/model_selection/_validation.py", line 895, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "/home/ag2435/kt_regression/src/npr/npr/rpcholesky/estimators.py", line 33, in fit
    model.fit_Nystrom(X, y,
  File "/home/ag2435/kt_regression/src/npr/npr/rpcholesky/KRR_Nystrom.py", line 54, in fit_Nystrom
    self.sol = scipy.linalg.solve(KMn @ KnM + KnM.shape[0]*lamb*KMM + 100*KMM.max()*np.finfo(float).eps*np.identity(sample_num), KMn @ Ytr, assume_a='pos')
  File "/home/ag2435/.conda/envs/npr/lib/python3.10/site-packages/scipy/linalg/_basic.py", line 253, in solve
    _solve_check(n, info)
  File "/home/ag2435/.conda/envs/npr/lib/python3.10/site-packages/scipy/linalg/_basic.py", line 41, in _solve_check
    raise LinAlgError('Matrix is singular.')
numpy.linalg.LinAlgError: Matrix is singular.
```

- [ ] Of the 1000 fits, how many of them ended in singular error and how many just had ill-conditioned matrix?
- [ ] Is it dependent on $\sigma$?