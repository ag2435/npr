# # sum-of-2-gaussian
# python run_synthetic.py -m krr -thin full -cv -alpha 0.1 -t 1 -j 4 -k gaussian -op $1
# python run_synthetic.py -m krr -thin st -cv -alpha 0.1 -j 4 -k gaussian -op $1
# python run_synthetic.py -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian -op $1
# python run_synthetic.py -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian --ablation 1 -op $1
# python run_synthetic.py -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian --ablation 2 -op $1
# python run_synthetic.py -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian --ablation 3 -op $1
# python run_synthetic.py -m krr -thin rpcholesky -cv -alpha 1e-5 -j 4 -k gaussian -op $1

# # sin
# python run_synthetic.py -gt sin -m krr -thin full -cv -alpha 0.1 -t 1 -j 4 -k gaussian -op $1
# python run_synthetic.py -gt sin -m krr -thin st -cv -alpha 0.1 -j 4 -k gaussian -op $1
# python run_synthetic.py -gt sin -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian -op $1
# python run_synthetic.py -gt sin -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian --ablation 1 -op $1
# python run_synthetic.py -gt sin -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian --ablation 2 -op $1
# python run_synthetic.py -gt sin -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian --ablation 3 -op $1
# python run_synthetic.py -gt sin -m krr -thin rpcholesky -cv -alpha 1e-5 -j 4 -k gaussian -op $1

# sinexp
# python run_synthetic.py -gt sinexp -m krr -thin full -sig [0.01,0.1,0.01] -alpha 0.1 -t 1 -j 4 -k gaussian -op $1
# python run_synthetic.py -gt sinexp -m krr -thin st -sig [0.01,0.1,0.01] -alpha 0.1 -j 4 -k gaussian -op $1
# python run_synthetic.py -gt sinexp -m krr -thin kt -sig [0.01,0.1,0.01] -alpha 0.1 -j 4 -k gaussian -op $1
python run_synthetic.py -gt sinexp -m krr -thin kt -sig [0.01,0.1,0.01] -alpha 0.1 -j 4 -k gaussian --ablation 1 -op $1
python run_synthetic.py -gt sinexp -m krr -thin kt -sig [0.01,0.1,0.01] -alpha 0.1 -j 4 -k gaussian --ablation 2 -op $1
python run_synthetic.py -gt sinexp -m krr -thin kt -sig [0.01,0.1,0.01] -alpha 0.1 -j 4 -k gaussian --ablation 3 -op $1
# python run_synthetic.py -gt sinexp -m krr -thin rpcholesky -sig [0.01,0.1,0.01] -alpha 1e-5 -j 4 -k gaussian -op $1