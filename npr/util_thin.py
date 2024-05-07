'''File containing helper functions for details about dataset thinning
'''

from goodpoints.compress import largest_power_of_four
import numpy as np

# STANDARD THINNING

def log4(n):
    return np.log2(n) / 2

def get_g(n):
    return int( np.ceil( log4(log4(n)) ) ) # Use default value

# def get_coreset_size(n, m=1):
#     if get_g(n) <= m:
#         # with TicToc('compresspp', print_toc=PRINT_TOC):
#         # Compress with g'=g+inflation (compressing returns set of size 2^(g+inflation) sqrt(n) )
#         # Thin with g'=g (thinning returns set of size 2^inflation sqrt(n) )
#         largest_pow_four = largest_power_of_four(n)
#         log2n = n.bit_length() - 1
#         scale = n // largest_pow_four
#         return 2**( 2*(log2n//2) - m ) * scale
#     else:
#         return int(n / 2**m)

def get_coreset_size(n, m=1):
    nearest_pow_four = largest_power_of_four(n)
    return nearest_pow_four // 2**m

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
