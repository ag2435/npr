'''File containing helper functions for details about dataset thinning
'''

# use divide-and-conquer versions of KT and compress
from ... import kt_dnc, compress_dnc

import numpy as np
from functools import partial

# STANDARD THINNING

def log4(n):
    return np.log2(n) / 2

def get_g(n):
    return int( np.ceil( log4(log4(n)) ) ) # Use default value

def get_coreset_size(n, m=1):
    # if get_g(n) <= m:
    #     # with TicToc('compresspp', print_toc=PRINT_TOC):
    #     # Compress with g'=g+inflation (compressing returns set of size 2^(g+inflation) sqrt(n) )
    #     # Thin with g'=g (thinning returns set of size 2^inflation sqrt(n) )
    #     largest_pow_four = compress_dnc.largest_power_of_four(n)
    #     log2n = n.bit_length() - 1
    #     scale = n // largest_pow_four
    #     return 2**( 2*(log2n//2) - m ) * scale
        
    # else:
    #     return int(n / 2**m)
    
    return int(n/2**m)

def sd_thin_dnc(X, m=1, verbose=0):
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
    if verbose:
        print(f'coreset size: {coreset_size}, m: {m}')

    # return indeces[:coreset_size]
    # n' = 2^m * coreset_size
    coresets = np.array_split(indeces[:2**m * coreset_size], 2**m)
    return coresets

# KERNEL THINNING

def kt_thin2_dnc(X, split_kernel, seed=None, m=1, store_K=True):
    '''
    Construct coreset using compress.compresspp function
    
    Args:
    - X: dataset of size n
    - split_kernel: kernel used for splitting
    - swap_kernel: kernel used for swapping
    - seed: random seed
    - m: number of times to halve
    - store_K: whether to store the kernel matrix 
        (storing usually leads to faster runtime but uses more memory)

    Returns:
    - List of coresets
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
    halve = lambda x: kt_dnc.thin(X = x, m=1, split_kernel = split_kernel, 
                                #   swap_kernel = swap_kernel, 
                                                    seed = seed, 
                                                    # unique=True, 
                                                    delta = halve_prob*(len(x)**2), 
                                                    store_K=store_K)
    thin = partial(kt_dnc.thin, m=g, split_kernel = split_kernel, 
                #    swap_kernel = swap_kernel, 
                            seed = seed, delta = thin_prob, store_K=store_K)
    
    if g <= m:
        # with TicToc('compresspp', print_toc=PRINT_TOC):
        # Compress with g'=g+inflation (compressing returns set of size 2^(g+inflation) sqrt(n) )
        # Thin with g'=g (thinning returns set of size 2^inflation sqrt(n) )
        k = l - l//2 - m
        result = compress_dnc.compresspp(X, halve, thin, g + k)
    else:
        result = thin(X, m=m)

    return result

def kt_thin1_dnc(X, split_kernel, seed=None, m=1, store_K=True):
    """
    Construct coreset using kt.thin function
    NOTE: this function is slow for large datasets and takes O(n^2) time,
        but since DnC takes O(n^2) time, it only increases the total runtime by a constant factor
    """
    n = len(X)

    if m is None:
        m = int(log4(n))

    return kt_dnc.thin(X, m=m, split_kernel=split_kernel, seed=seed, store_K=store_K)