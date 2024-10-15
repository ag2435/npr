#!/bin/bash
# # sum-of-2-gaussian
# python run_synthetic.py -m nw -thin full -cv -j 4 -t 1 -op $1
# python run_synthetic.py -m nw -thin st -cv -j 4 -op $1
# python run_synthetic.py -m nw -thin kt -cv -j 4 -op $1
# python run_synthetic.py -m nw -thin kt -cv -j 4 --ablation 1 -op $1
# python run_synthetic.py -m nw -thin kt -cv -j 4 --ablation 2 -op $1
# python run_synthetic.py -m nw -thin kt -cv -j 4 --ablation 3 -op $1

# # sin
# python run_synthetic.py -gt sin -m nw -thin full -cv -j 4 -t 1 -op $1
# python run_synthetic.py -gt sin -m nw -thin st -cv -j 4 -op $1
# python run_synthetic.py -gt sin -m nw -thin kt -cv -j 4 -op $1
# python run_synthetic.py -gt sin -m nw -thin kt -cv -j 4 --ablation 1 -op $1
# python run_synthetic.py -gt sin -m nw -thin kt -cv -j 4 --ablation 2 -op $1
# python run_synthetic.py -gt sin -m nw -thin kt -cv -j 4 --ablation 3 -op $1

# sin * exp
KERNEL=wendland
# python run_synthetic.py -gt sinexp -m nw -k $KERNEL -thin full -sig [0.01,0.1,0.01] -j 4 -t 1 -op $1
# python run_synthetic.py -gt sinexp -m nw -k $KERNEL -thin st -sig [0.01,0.1,0.01] -j 4 -op $1
# python run_synthetic.py -gt sinexp -m nw -k $KERNEL -thin kt -sig [0.01,0.1,0.01] -j 4 -op $1
# python run_synthetic.py -gt sinexp -m nw -k $KERNEL -thin kt -sig [0.01,0.1,0.01] -j 4 --ablation 1 -op $1
# python run_synthetic.py -gt sinexp -m nw -k $KERNEL -thin kt -sig [0.01,0.1,0.01] -j 4 --ablation 2 -op $1
# python run_synthetic.py -gt sinexp -m nw -k $KERNEL -thin kt -sig [0.01,0.1,0.01] -j 4 --ablation 3 -op $1
python run_synthetic.py -gt sinexp -m nw -k $KERNEL -thin rpcholesky -sig [0.01,0.1,0.01] -j 4 -op $1