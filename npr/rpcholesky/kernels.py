#!/usr/bin/env python3

import numpy as np
from scipy.special import gamma, kv
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

def LaplaceKernel(x,y,bandwidth=1.0):
    # for single x,y in R^d
    return np.exp( -np.linalg.norm(x-y, ord=1) / bandwidth )

def LaplaceKernel_vec(vec_x,vec_y,bandwidth=1.0):
    # for vec_x, vec_y in R^{n*d}, return n values
    dsts = np.linalg.norm(vec_x-vec_y, ord = 1, axis = -1)
    return np.exp(-dsts/bandwidth)

def LaplaceKernel_mtx(xx,yy,bandwidth=1.0):
    # xx in R^{nx*d} and yy in R^{ny*d} are both collection of points (two axis)
    # return nx*ny values
    # dsts = np.linalg.norm(xx[:, None, :] - yy[None, :, :], axis=-1)
    dsts = manhattan_distances(xx,yy) # faster
    return np.exp(-dsts/bandwidth)

def GaussianKernel(x,y,bandwidth=1.0):
# for single x,y in R^d
    return np.exp(-0.5*np.linalg.norm(x-y)**2/bandwidth**2)

def GaussianKernel_vec(vec_x,vec_y, bandwidth=1.0):
    # for vec_x, vec_y in R^{n*d}, return n values
    dsts = np.linalg.norm(vec_x-vec_y, axis = -1) #**2
    return np.exp(-0.5*dsts**2/bandwidth**2)

def GaussianKernel_mtx(xx,yy,bandwidth=1.0):
    # xx in R^{nx*d} and yy in R^{ny*d} are both collection of points (two axis)
    # return nx*ny values
    # dsts = np.linalg.norm(xx[:, None, :] - yy[None, :, :], axis=-1)
    dsts = euclidean_distances(xx,yy) # faster
    return np.exp(-0.5*dsts**2/bandwidth**2)
    
def MaternKernel(x,y,bandwidth=1.0, nu=0.5):
    # for single x,y in R^d
    d = np.linalg.norm(x-y) / bandwidth
    if nu == 0.5:
        return np.exp(-d)
    elif nu == 1.5:
        sqrt3 = np.sqrt(3.0)
        return (1 + sqrt3*d) * np.exp(-sqrt3*d)
    elif nu == 2.5:
        sqrt5 = np.sqrt(5.0)
        return (1 + sqrt5*d + 5.0/3.0*d*d ) * np.exp(-sqrt5*d)
    else:
        sqrt2nu = np.sqrt(2.0*nu)
        return 1.0/(gamma(nu) * 2.0**(nu-1)) * (sqrt2nu * d) ** nu * kv(nu, sqrt2nu * d)

def MaternKernel_vec(vec_x,vec_y,bandwidth=1.0, nu=0.5):
    # for vec_x, vec_y in R^{n*d}, return n values
    d = np.linalg.norm(vec_x-vec_y, axis = -1)/bandwidth
    if nu == 0.5:
        return np.exp(-d)
    elif nu == 1.5:
        sqrt3 = np.sqrt(3.0)
        return (1 + sqrt3*d) * np.exp(-sqrt3*d)
    elif nu == 2.5:
        sqrt5 = np.sqrt(5.0)
        return (1 + sqrt5*d + 5.0/3.0*d*d ) * np.exp(-sqrt5*d)
    else:
        sqrt2nu = np.sqrt(2.0*nu)
        return 1.0/(gamma(nu) * 2.0**(nu-1)) * (sqrt2nu * d) ** nu * kv(nu, sqrt2nu * d)

def MaternKernel_mtx(xx,yy,bandwidth,nu):
        # xx in R^{nx*d} and yy in R^{ny*d} are both collection of points (two axis)
        # return nx*ny values
        # dsts = np.linalg.norm(xx[:, None, :] - yy[None, :, :], axis=-1)
    d = np.sqrt(euclidean_distances(xx,yy))/bandwidth # faster
    if nu == 0.5:
        return np.exp(-d)
    elif nu == 1.5:
        sqrt3 = np.sqrt(3.0)
        return (1 + sqrt3*d) * np.exp(-sqrt3*d)
    elif nu == 2.5:
        sqrt5 = np.sqrt(5.0)
        return (1 + sqrt5*d + 5.0/3.0*d*d ) * np.exp(-sqrt5*d)
    else:
        sqrt2nu = np.sqrt(2.0*nu)
        return 1.0/(gamma(nu) * 2.0**(nu-1)) * (sqrt2nu * d) ** nu * kv(nu, sqrt2nu * d)

# 
# Epanechnikov kernel
# 
def EpanechnikovKernel(x,y,bandwidth=1.0):
    # for single x,y in R^d
    dst = np.linalg.norm(x-y)
    dst /= bandwidth
    return 0 if dst <= 1 else 0.75 * (1 - dst**2)

def EpanechnikovKernel_vec(vec_x,vec_y, bandwidth=1.0):
    # for vec_x, vec_y in R^{n*d}, return n values
    dsts = np.linalg.norm(vec_x-vec_y, axis = -1) #**2
    dsts /= bandwidth
    return np.where(dsts <= 1, 0.75 * (1 - dsts**2), 0)

def EpanechnikovKernel_mtx(xx,yy,bandwidth=1.0):
    # xx in R^{nx*d} and yy in R^{ny*d} are both collection of points (two axis)
    # return nx*ny values
    # dsts = np.linalg.norm(xx[:, None, :] - yy[None, :, :], axis=-1)
    dsts = euclidean_distances(xx,yy) # faster
    dsts /= bandwidth
    # return np.exp(-0.5*dsts**2/bandwidth**2)
    return np.where(dsts <= 1, 0.75 * (1 - dsts**2), 0)
# 
# Wendland kernel
# 
def WendlandKernel(x,y,bandwidth=1.0):
    # for single x,y in R^d
    dst = np.linalg.norm(x-y)
    dst /= bandwidth
    return 0 if dst <= 1 else (1 - dst)

def WendlandKernel_vec(vec_x,vec_y, bandwidth=1.0):
    # for vec_x, vec_y in R^{n*d}, return n values
    dsts = np.linalg.norm(vec_x-vec_y, axis = -1) #**2
    dsts /= bandwidth
    return np.where(dsts <= 1, 1 - dsts, 0)

def WendlandKernel_mtx(xx,yy,bandwidth=1.0):
    # xx in R^{nx*d} and yy in R^{ny*d} are both collection of points (two axis)
    # return nx*ny values
    # dsts = np.linalg.norm(xx[:, None, :] - yy[None, :, :], axis=-1)
    dsts = euclidean_distances(xx,yy) # faster
    dsts /= bandwidth
    # return np.exp(-0.5*dsts**2/bandwidth**2)
    return np.where(dsts <= 1, 1 - dsts, 0)
    