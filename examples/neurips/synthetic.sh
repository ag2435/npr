# 
# %%%%%%%%%%%%%%%%%%%%%%%% Nadaraya-Watson Experiments %%%%%%%%%%%%%%%%%%%%%%%%
# 
# default hyperparameters:
#   sigma: 0.1
# 
# WITHOUT Cross Validation 
# (d=1)
python run_synthetic.py -m nw -thin full -k epanechnikov
python run_synthetic.py -m nw -thin st -k epanechnikov
python run_synthetic.py -m nw -thin kt -k epanechnikov
# (d=4)
python run_synthetic.py -d 4 -m nw -thin full -k epanechnikov
python run_synthetic.py -d 4 -m nw -thin st -k epanechnikov
python run_synthetic.py -d 4 -m nw -thin kt -k epanechnikov
# (d=8)
python run_synthetic.py -d 8 -m nw -thin full -k epanechnikov
python run_synthetic.py -d 8 -m nw -thin st -k epanechnikov
python run_synthetic.py -d 8 -m nw -thin kt -k epanechnikov

# 
# WITH Cross Validation
# \sigma range: [0.1, 2.0] in 0.1 increments
# (d=1)
python run_synthetic.py -m nw -thin full -cv -j 4 -t 1
python run_synthetic.py -m nw -thin st -cv -j 4
python run_synthetic.py -m nw -thin kt -cv -j 4
python run_synthetic.py -m nw -thin kt -cv -j 4 --ablation 1
python run_synthetic.py -m nw -thin kt -cv -j 4 --ablation 2
python run_synthetic.py -m nw -thin kt -cv -j 4 --ablation 3
# (d=4)
python run_synthetic.py -d 4 -m nw -thin full -cv -j 4 -t 1
# optimal \sigma ~ 1.8-2.0
python run_synthetic.py -d 4 -m nw -thin st -cv -j 4
# optimal \sigma ~ 1.8-2.0
python run_synthetic.py -d 4 -m nw -thin kt -cv -j 4
# (d=8)
python run_synthetic.py -d 8 -m nw -thin full -cv -j 4 -t 1
python run_synthetic.py -d 8 -m nw -thin st -cv -j 4
python run_synthetic.py -d 8 -m nw -thin kt -cv -j 4

# 
# To run ablations for kernel thinning, add --ablation <ABLATION_STUDY> to the above commands
# 

# 
# %%%%%%%%%%%%%%%%%%%%%%%% Kernel Ridge Regression Experiments %%%%%%%%%%%%%%%%%%%%%%%%
# 
# default hyperparameters:
#   sigma: 0.1
#   alpha: 1e-3
# NOTE: 
#   optimal alpha for rpcholesky is lower than for krr
#   full-krr solution is deterministic (to make things faster, can do 1 trial only)
#   run >1 trials to get timings
#   we use ground truth sigma=0.25
# 
# WITHOUT Cross Validation
# 
# (d=1)
python run_synthetic.py -m krr -thin full -k gaussian -sig 0.25 -alpha 0.1 -t 1
python run_synthetic.py -m krr -thin st -k gaussian -sig 0.25 -alpha 0.1
python run_synthetic.py -m krr -thin kt -k gaussian -sig 0.25 -alpha 0.1 
# NOTE: rpcholesky gives LinAlgError for large n
python run_synthetic.py -m krr -thin rpcholesky -k gaussian -sig 0.25 -alpha 1e-4 -lo 8 -hi 10
# (d=4)
python run_synthetic.py -d 4 -m krr -thin full -k gaussian -sig 0.25 -alpha 0.1 -t 1
python run_synthetic.py -d 4 -m krr -thin st -k gaussian -sig 0.25 -alpha 0.1
python run_synthetic.py -d 4 -m krr -thin kt -k gaussian -sig 0.25 -alpha 0.1 
# NOTE: rpcholesky gives LinAlgError for large n
python run_synthetic.py -d 4 -m krr -thin rpcholesky -k gaussian -sig 0.25 -alpha 1e-4 -lo 8 -hi 10
# (d=8)
python run_synthetic.py -d 8 -m krr -thin full -k gaussian -sig 0.25 -alpha 0.1 -t 1
python run_synthetic.py -d 8 -m krr -thin st -k gaussian -sig 0.25 -alpha 0.1
python run_synthetic.py -d 8 -m krr -thin kt -k gaussian -sig 0.25 -alpha 0.1 
# NOTE: rpcholesky gives LinAlgError for large n
python run_synthetic.py -d 8 -m krr -thin rpcholesky -k gaussian -sig 0.25 -alpha 1e-4 -lo 8 -hi 10

# 
# WITH Cross Validation
# \sigma range: [0.1, 2.0] in 0.1 increments
# 
# (d=1)
python run_synthetic.py -m krr -thin full -cv -alpha 0.1 -t 1 -j 4 -k gaussian
python run_synthetic.py -m krr -thin st -cv -alpha 0.1 -j 4 -k gaussian
python run_synthetic.py -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian
python run_synthetic.py -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian --ablation 1
python run_synthetic.py -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian --ablation 2
python run_synthetic.py -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian --ablation 3
python run_synthetic.py -m krr -thin rpcholesky -cv -alpha 1e-5 -j 4 -k gaussian
# (d=4)
# optimal \sigma ~ 0.9-1.0
python run_synthetic.py -d 4 -m krr -thin full -cv -alpha 0.1 -t 1 -j 4 -k gaussian
# optimal \sigma ~ 1.3-1.5
python run_synthetic.py -d 4 -m krr -thin st -cv -alpha 0.1 -j 4 -k gaussian
# optimal \sigma ~ 1.5
python run_synthetic.py -d 4 -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian
# optimal \sigma ~ >2
python run_synthetic.py -d 4 -m krr -thin full -cv -alpha 1e-5 -j 4 -k gaussian
# (d=8)
python run_synthetic.py -d 8 -m krr -thin full -cv -alpha 0.1 -t 1 -j 4 -k gaussian
python run_synthetic.py -d 8 -m krr -thin st -cv -alpha 0.1 -j 4 -k gaussian
python run_synthetic.py -d 8 -m krr -thin kt -cv -alpha 0.1 -j 4 -k gaussian
python run_synthetic.py -d 8 -m krr -thin full -cv -alpha 1e-5 -j 4 -k gaussian
# 
# To run ablations for kernel thinning, add --ablation <ABLATION_STUDY> to the above commands
# 