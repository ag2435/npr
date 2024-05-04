'''File containing helper functions for details about dataset thinning
'''

from ... import kt, compress

import numpy as np
from functools import partial

# STANDARD THINNING

def log4(n):
    return np.log2(n) / 2

def get_g(n):
    return int( np.ceil( log4(log4(n)) ) ) # Use default value

def get_coreset_size(n, m=1):
    if get_g(n) <= m:
        # with TicToc('compresspp', print_toc=PRINT_TOC):
        # Compress with g'=g+inflation (compressing returns set of size 2^(g+inflation) sqrt(n) )
        # Thin with g'=g (thinning returns set of size 2^inflation sqrt(n) )
        largest_pow_four = compress.largest_power_of_four(n)
        log2n = n.bit_length() - 1
        scale = n // largest_pow_four
        return 2**( 2*(log2n//2) - m ) * scale
    else:
        return int(n / 2**m)

def sd_thin(X, m=1):
    '''
    Args:
    - X: dataset of size n
    - m: number of times to halve
    '''

    n = len(X)
    indeces = np.arange(0, n)
    np.random.shuffle(indeces) # Randomly shuffle indices

    if m is None:
        m = int(log4(n))
    coreset_size = get_coreset_size(n, m=m)

    return indeces[:coreset_size]

# KERNEL THINNING

def kt_thin1(X, split_kernel, swap_kernel, seed=123):
    # 
    # Construct coreset using kt.thin function
    # 
    m = int(np.log2(len(X)) // 2)
    result = kt.thin(X, m, split_kernel, swap_kernel, delta=0.5, seed=seed)
    return result

def kt_thin2(X, split_kernel, swap_kernel, seed=123, m=1, store_K=True):
    '''
    Construct coreset using compress.compresspp function
    Note: Should be much faster than kt_thin1
    
    Args:
    - X: dataset of size n
    - split_kernel: kernel used for splitting
    - swap_kernel: kernel used for swapping
    - seed: random seed
    - m: number of times to halve
    '''

    n = len(X)
    l = n.bit_length() - 1
    # print('n:', n)

    size = int(log4(n))
    # print('size:', size)
    g = get_g(n)
    assert g <= size
    
    if m is None:
        m = int(log4(n))

    # Specify base failure probability for kernel thinning
    delta = 0.5
    # Each Compress Halve call applied to an input of length l uses KT( l^2 * halve_prob ) 
    halve_prob = delta / ( 4*(4**size)*(2**g)*( g + (2**g) * (size  - g) ) )
    # print('halve prop:', halve_prob)
    ###halve_prob = 0 if size == g else delta * .5 / (4 * (4**size) * (4 ** g) * (size - g) ) ###
    # Each Compress++ Thin call uses KT( thin_prob )
    thin_prob = delta * g / (g + ( (2**g)*(size - g) ))
    # print('thin prop:', thin_prob)
    
    # Use kt.thin for compress algorithm
    # with TicToc('declare halve thin'):
    halve = compress.symmetrize(lambda x: kt.thin(X = x, m=1, split_kernel = split_kernel, swap_kernel = swap_kernel, 
                                                    seed = seed, unique=True, delta = halve_prob*(len(x)**2), store_K=store_K))
    thin = partial(kt.thin, m=g, split_kernel = split_kernel, swap_kernel = swap_kernel, 
                            seed = seed, delta = thin_prob, store_K=store_K)
    
    if g <= m:
        # with TicToc('compresspp', print_toc=PRINT_TOC):
        # Compress with g'=g+inflation (compressing returns set of size 2^(g+inflation) sqrt(n) )
        # Thin with g'=g (thinning returns set of size 2^inflation sqrt(n) )
        k = l - l//2 - m
        result = compress.compresspp(X, halve, thin, g + k)
    else:
        result = thin(X, m=m)

    return result

def kt_thin3(X, split_kernel, swap_kernel, seed=123, m=1, var_k=1.):
    '''
    Construct coreset using compress.compresspp function
    Note: Should be much faster than kt_thin1
    
    Args:
    - X: dataset of size n
    - split_kernel: kernel used for splitting
    - swap_kernel: kernel used for swapping
    - seed: random seed
    - m: number of times to halve

    NOTE: only for gaussian
    '''

    n = len(X)
    l = n.bit_length() - 1
    print('n:', n)
    print('l:', l)

    size = int(log4(n))
    print('size:', size)
    g = get_g(n)
    assert g <= size
    
    if m is None:
        m = size

    # Specify base failure probability for kernel thinning
    delta = 0.5
    # Each Compress Halve call applied to an input of length l uses KT( l^2 * halve_prob ) 
    halve_prob = delta / ( 4*(4**size)*(2**g)*( g + (2**g) * (size  - g) ) )
    # print('halve prop:', halve_prob)
    ###halve_prob = 0 if size == g else delta * .5 / (4 * (4**size) * (4 ** g) * (size - g) ) ###
    # Each Compress++ Thin call uses KT( thin_prob )
    thin_prob = delta * g / (g + ( (2**g)*(size - g) ))
    # print('thin prop:', thin_prob)
    
    thin = partial(kt.thin, m=g+1, split_kernel = split_kernel, swap_kernel = swap_kernel, 
                            seed = seed, delta = thin_prob)
    
    if g <= m:
        k = l - l//2 - m
        X_intermediate = compress.compress_gsn_kt(X, g+ k, lam_sqd=np.array([var_k,]), delta=delta, seed=seed)
        # size_intermediate = 2^(g+1) sqrt(n)
        assert len(X_intermediate) == 2**(g+k+1) * 2**(l//2), len(X_intermediate)
        result = thin(X_intermediate)
    else:
        print('g > m, where g: {g}, m: {m}...using thin (no compress)')
        result = thin(X, m=m)

    return result
