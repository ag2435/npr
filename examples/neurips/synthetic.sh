# nadaraya-watson
# default hyperparameters:
#   sigma: 0.1
# python run_synthetic.py -m nw -thin full -k epanechnikov
# python run_synthetic.py -m nw -thin st -k epanechnikov
# python run_synthetic.py -m nw -thin kt -k epanechnikov

# kernel ridge regression
# default hyperparameters:
#   sigma: 0.1
#   alpha: 1e-3
# NOTE: 
#   optimal alpha for rpcholesky is lower than for krr
#   full-krr solution is deterministic (to make things faster, can do 1 trial only)
#   run >1 trials to get timings
#   we use ground truth sigma=0.25
python run_synthetic.py -m krr -thin full -k gaussian -sig 0.25 -alpha 0.1 -t 1
python run_synthetic.py -m krr -thin st -k gaussian -sig 0.25 -alpha 0.1
python run_synthetic.py -m krr -thin kt -k gaussian -sig 0.25 -alpha 0.1 
# NOTE: rpcholesky gives LinAlgError for large n
python run_synthetic.py -m krr -thin rpcholesky -k gaussian -sig 0.25 -alpha 1e-4 -lo 8 -hi 10