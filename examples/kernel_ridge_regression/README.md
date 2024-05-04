# KRR Experiments

To install goodpoints:
```
pip install -e .
```

*Recommended:* install dependencies using conda
```
conda create -n goodpoints python=3.10 jupyter pandas line_profiler plotly nbformat papermill
conda install pytorch=2.0.0 torchvision torchaudio torchmetrics cpuonly -c pytorch
# install falkon
pip install falkon -f https://falkon.dibris.unige.it/torch-2.0.0_cpu.html
```
Note: make sure to use python==3.10. For some reason, I had issues with sklearn GridSearchCV multiprocessing using python==3.11.

## Line profiler results

`%lprun -f kt.split_X kt_coreset = kt_thin2(X_train, split_kernel, swap_kernel, m=m)`:
```
Timer unit: 1e-09 s

Total time: 22.3625 s
File: /Users/ag2435/repos/goodpoints/goodpoints/kt.py
Function: split_X at line 237

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   237                                           def split_X(X, m, kernel, delta=0.5, seed=None, verbose=False):
   238                                               """Returns 2^m KT-SPLIT coresets of size floor(n/2^m) as a 2D array.
   239                                               Uses O(nd) memory as kernel matrix is not stored.
   240                                               
   241                                               Args:
   242                                                 X: Input sequence of sample points with shape (n, d)
   243                                                 m: Number of halving rounds
   244                                                 kernel: Kernel function (typically a square-root kernel, krt);
   245                                                   kernel(y,X) returns array of kernel evaluations between y and each row of X
   246                                                 delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
   247                                                 seed: Random seed to set prior to generation; if None, no seed will be set
   248                                                 verbose: If False do not print intermediate time taken, if True print that info
   249                                               """
   250        86      49000.0    569.8      0.0      if m == 0:
   251                                                   # Zero halving rounds requested
   252                                                   # Return 2D coreset array containing a single coreset (one row) with all indices
   253                                                   return(np.arange(X.shape[0], dtype=int)[newaxis,:])
   254                                               
   255        86      28000.0    325.6      0.0      verbose = verbose and (m>=7)
   256                                               # Function which returns kernel value for two arrays of row indices of X
   257        86      32000.0    372.1      0.0      def k(ii, jj):
   258                                                   return(kernel(X[ii], X[jj]))
   259                                               
   260                                               # Initialize random number generator
   261        86    2363000.0  27476.7      0.0      rng = default_rng(seed)
   262        86      31000.0    360.5      0.0      n, _ = X.shape
   263                                               
   264                                               # Initialize coresets, each a vector of integers indexing the rows of X
   265        86      38000.0    441.9      0.0      coresets = dict()
   266                                               # Store sum of kernel evaluations between each point eventually added to a coreset
   267                                               # and all of the points previously added to that coreset
   268        86       9000.0    104.7      0.0      KC = dict()
   269                                           
   270                                               # Initialize subGaussian parameters
   271                                               # sig_sqd[j][j2] determines the threshold for halving coresets[j][j2]
   272        86      10000.0    116.3      0.0      sig_sqd = dict()
   273                                               # Store multiplier to avoid recomputing
   274        86     290000.0   3372.1      0.0      log_multiplier = 2*np.log(2*n*m/delta)
   275                                           
   276       173      42000.0    242.8      0.0      for j in range(m+1):
   277                                                   # Initialize coresets[j][j2] for each j2 < 2^j to array of size n/2^j 
   278                                                   # with invalid -1 values
   279       173      66000.0    381.5      0.0          num_coresets = int(2**j)
   280                                                   
   281       173      19000.0    109.8      0.0          num_points_in_coreset = n//num_coresets
   282       173     748000.0   4323.7      0.0          coresets[j] = np.full((num_coresets, num_points_in_coreset), -1, dtype=int)
   283                                                   # Initialize associated coreset kernel sums arbitrarily
   284       173     110000.0    635.8      0.0          KC[j] = np.empty((num_coresets, num_points_in_coreset))
   285                                           
   286                                                   # Initialize subGaussian parameters to 0 
   287       173      72000.0    416.2      0.0          sig_sqd[j] = np.zeros(num_coresets)
   288                                                                         
   289                                               # Store kernel(xi, xi) for each point i
   290        86      60000.0    697.7      0.0      diagK = np.empty(n)
   291                                                   
   292                                               # If verbose---Track the time taken if m is large
   293                                               # Output timing when sample size doubles til n/2, and then every n/8 sample points
   294        86      12000.0    139.5      0.0      nidx = 1
   295        86      92000.0   1069.8      0.0      tic()
   296     15872    1929000.0    121.5      0.0      for i in range(n):
   297                                                   # Track progress
   298     14983    2065000.0    137.8      0.0          if i==nidx:
   299       889    1188000.0   1336.3      0.0              fprint(f"Tracking update: Finished processing sample number {nidx}/{n}", verbose=verbose)
   300       889    1688000.0   1898.8      0.0              toc(print_elapsed=verbose)
   301       889     990000.0   1113.6      0.0              tic()
   302                                                       
   303       545     157000.0    288.1      0.0              if nidx<int(n/2):
   304       545      89000.0    163.3      0.0                  nidx *= 2
   305                                                       else:
   306       344     104000.0    302.3      0.0                  nidx += int(n/2**3)
   307                                           
   308                                                   # Add each datapoint to coreset[0][0]
   309     15872    5232000.0    329.6      0.0          coreset = coresets[0][0]
   310     15872    2825000.0    178.0      0.0          coreset[i] = i
   311                                                   # Capture index i as 1D array to ensure X[i_array] is a 2D array
   312     15872    3781000.0    238.2      0.0          i_array = coreset[i, newaxis]
   313                                                   # Store kernel evaluation with all points <= i
   314     15872 14873303000.0 937078.1     66.5          ki = k(i_array, coreset[:(i+1)]) 
   315                                                   # Store summed kernel inner product with all points < i
   316     15872   59354000.0   3739.5      0.3          KC[0][0,i] = np.sum(ki[:i]) 
   317                                                   # Store diagonal element, kernel(xi, xi)
   318     15872    3755000.0    236.6      0.0          diagK[i] = ki[i] 
   319                                           
   320                                                   # If 2^(j+1) divides (i+1), add a point from coreset[j][j2] to each of
   321                                                   # coreset[j+1][2*j2] and coreset[j+1][2*j2+1]
   322     15872   13586000.0    856.0      0.1          for j in range(min(m, largest_power_of_two(i+1))):
   323      8064    1220000.0    151.3      0.0              parent_coresets = coresets[j]
   324      8064    1268000.0    157.2      0.0              child_coresets = coresets[j+1]
   325      8064     854000.0    105.9      0.0              parent_KC = KC[j]
   326      8064     998000.0    123.8      0.0              child_KC = KC[j+1]
   327      8064    2217000.0    274.9      0.0              num_parent_coresets = parent_coresets.shape[0]
   328                                                       # j_log_multiplier = 2*np.log(2*n*m/delta/2^j) 
   329                                                       #                  = 2*np.log(2*n*m/delta) - j * 2 log(2)
   330                                                       #                  = log_multiplier - j * TWO_LOG_2
   331                                                       # the term is 2^{j-1} in the paper because j starts at 1; here j starts at 0
   332      8064    2374000.0    294.4      0.0              j_log_multiplier = log_multiplier - j * TWO_LOG_2
   333                                                       # Consider each parent coreset in turn
   334      8192    2142000.0    261.5      0.0              for j2 in range(num_parent_coresets):
   335      8192    1966000.0    240.0      0.0                  parent_coreset = parent_coresets[j2]
   336                                                           #tic()
   337                                                           # Number of points in parent_coreset
   338      8192    1263000.0    154.2      0.0                  parent_idx = (i+1) // num_parent_coresets
   339                                                           # Get last two points from the parent coreset
   340                                                           # newaxis ensures array dimensions are appropriate for kernel function
   341      8192    3642000.0    444.6      0.0                  point1, point2 = parent_coreset[parent_idx-2, newaxis], parent_coreset[parent_idx-1, newaxis]
   342                                                           # Compute kernel(x1, x2)
   343      8192  122920000.0  15004.9      0.5                  K12 = k(point1,point2)
   344                                           
   345                                                           # Use adaptive failure threshold
   346                                                           # Compute b^2 = ||f||^2 = ||k(x1,.) - k(x2,.)||_k^2
   347      8192   18760000.0   2290.0      0.1                  b_sqd = diagK[point2] + diagK[point1] - 2*K12
   348                                                           # Update threshold for halving parent coreset
   349                                                           # a = max(b sig sqrt(j_log_multiplier), b^2)
   350      8192   20439000.0   2495.0      0.1                  thresh = max(np.sqrt(sig_sqd[j][j2]*b_sqd*j_log_multiplier), b_sqd)
   351      8104    2740000.0    338.1      0.0                  if sig_sqd[j][j2] == 0:
   352        88     319000.0   3625.0      0.0                      sig_sqd[j][j2] = b_sqd
   353      8104    9057000.0   1117.6      0.0                  elif thresh != 0:
   354                                                               # Note: If threshold is zero, b_sqd is zero so sigma does not change
   355                                                               # If thresh != 0, update subGaussian parameter
   356                                                               # s^2 += 2*b^2*(.5 + (b^2/(2 a) - 1)*s^2/a)_+
   357      8104   28803000.0   3554.2      0.1                      sig_sqd_update = .5 + (b_sqd/(2*thresh) - 1)*sig_sqd[j][j2]/thresh
   358      4346    3919000.0    901.7      0.0                      if sig_sqd_update > 0:
   359      3758   14589000.0   3882.1      0.1                          sig_sqd[j][j2] += 2*b_sqd*sig_sqd_update
   360                                                           # To avoid division by zero, set zero threshold to arbitrary positive value
   361                                                           # (Does not impact algorithm correctness as b_sqd = 0 as well)
   362      8192    8639000.0   1054.6      0.0                  if thresh == 0: thresh = 1.
   363                                           
   364                                                           # Compute inner product with other points in parent coreset:
   365                                                           #  sum_{l < parent_idx-2} <k(coreset[j][l],.), k(x1, .) - k(x2, .)>
   366                                                           # Note that KC[j, point1] = <k(coreset[j][l],.), k(x1, .)> and
   367                                                           # KC[j, point2] = <k(coreset[j][l],.), k(x2, .)> + k(x1,x2)         
   368      8104    1256000.0    155.0      0.0                  if parent_idx > 2:
   369      8104    8542000.0   1054.0      0.0                      alpha = parent_KC[j2, parent_idx-2] - parent_KC[j2, parent_idx-1] + K12 
   370                                                           else:
   371        88      14000.0    159.1      0.0                      alpha = 0.
   372                                                           # Identify the two new child coresets
   373      8192    2498000.0    304.9      0.0                  left_child_coreset = child_coresets[2*j2]
   374      8192    2127000.0    259.6      0.0                  right_child_coreset = child_coresets[2*j2+1]
   375                                                           # Number of points in each new coreset
   376      8192    1406000.0    171.6      0.0                  child_idx = (parent_idx//2)-1
   377      8104     939000.0    115.9      0.0                  if child_idx > 0:
   378                                                               # Subtract 2 * inner product with all points in left child coreset:
   379                                                               # - 2 * sum_{l < child_idx} <k(coreset[j][l],.), k(x1, .) - k(x2, .)> 
   380      8104    1845000.0    227.7      0.0                      child_points = left_child_coreset[:child_idx]
   381      8104 3919858000.0 483694.2     17.5                      point1_kernel_sum = np.sum(k(point1,child_points))
   382      8104 3115611000.0 384453.5     13.9                      point2_kernel_sum = np.sum(k(point2,child_points))
   383      8104   14185000.0   1750.4      0.1                      alpha -= 2*(point1_kernel_sum - point2_kernel_sum)
   384                                                           else:
   385        88      13000.0    147.7      0.0                      point1_kernel_sum = 0
   386        88       8000.0     90.9      0.0                      point2_kernel_sum = 0
   387                                                           # Add point2 to coreset[j] with probability prob_point2; add point1 otherwise
   388      8192   18157000.0   2216.4      0.1                  prob_point2 = 0.5*(1-alpha/thresh)
   389      4200    9780000.0   2328.6      0.0                  if rng.random() <= prob_point2:
   390      4200   14118000.0   3361.4      0.1                      left_child_coreset[child_idx] = point2
   391      4200    5823000.0   1386.4      0.0                      right_child_coreset[child_idx] = point1
   392      4200    1354000.0    322.4      0.0                      child_KC[2*j2, child_idx] = point2_kernel_sum
   393      4200    1575000.0    375.0      0.0                      child_KC[2*j2+1, child_idx] = point1_kernel_sum
   394                                                           else:
   395      3992   12881000.0   3226.7      0.1                      left_child_coreset[child_idx] = point1
   396      3992    5385000.0   1348.9      0.0                      right_child_coreset[child_idx] = point2
   397      3992    1272000.0    318.6      0.0                      child_KC[2*j2, child_idx] = point1_kernel_sum
   398      3992    1568000.0    392.8      0.0                      child_KC[2*j2+1, child_idx] = point2_kernel_sum
   399                                           
   400                                               # Return coresets of size floor(n/2^m)
   401        86      16000.0    186.0      0.0      return(coresets[m])
```

`%lprun -f kt.swap_X kt_coreset = kt_thin2(X_train, split_kernel, swap_kernel, m=m)`:
```
Timer unit: 1e-09 s

Total time: 69.5919 s
File: /Users/ag2435/repos/goodpoints/goodpoints/kt.py
Function: swap_X at line 606

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   606                                           def swap_X(X, coresets, kernel, meanK=None, unique=False):
   607                                               """Selects the candidate coreset with smallest MMD to all points in X (after comparing with
   608                                               a baseline standard thinning coreset) and iteratively refine that coreset.
   609                                               
   610                                               Args:
   611                                                 X: Input sequence of sample points with shape (n, d)
   612                                                 coresets: 2D array with each row specifying the row indices of X belonging to a coreset
   613                                                 kernel: Kernel function (typically the target kernel, k);
   614                                                   kernel(y,X) returns array of kernel evaluations between y and each row of X
   615                                                 meanK: None or array of length n with meanK[ii] = mean of kernel(X[ii], X);
   616                                                   used to speed up computation when not None
   617                                                 unique: If True, constrains the output to never contain the same row index more than once
   618                                               """
   619                                               # Compute meanK if appropriate
   620        86 16281373000.0 189318290.7     23.4      if meanK is None: meanK = kernel_matrix_row_mean(X, kernel)
   621                                               # Return refined version of best coreset
   622        86 53310490000.0 619889418.6     76.6      return(refine_X(X, best_X(X, coresets, kernel, meanK=meanK), kernel, meanK=meanK, unique=unique))
```

`%lprun -f kt.refine_X kt_coreset = kt_thin2(X_train, split_kernel, swap_kernel, m=m)`
```
Timer unit: 1e-09 s

Total time: 32.4726 s
File: /Users/ag2435/repos/goodpoints/goodpoints/kt.py
Function: refine_X at line 751

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   751                                           def refine_X(X, coreset, kernel, meanK=None, unique=False):
   752                                               """
   753                                               Replaces each element of a coreset in turn by the point in X that yields the minimum 
   754                                               MMD between all points in X and the resulting coreset.
   755                                               
   756                                               Args:
   757                                                 X: Input sequence of sample points with shape (n, d)
   758                                                 coreset: Row indices of X representing coreset
   759                                                 kernel: Kernel function (typically the target kernel, k);
   760                                                   kernel(y,X) returns array of kernel evaluations between y and each row of X
   761                                                 meanK: None or array of length n with meanK[ii] = mean of kernel(X[ii], X);
   762                                                   used to speed up computation when not None
   763                                                 unique: If True, constrains the output to never contain the same row index more than once
   764                                                         (logic of point-by-point swapping is altered to ensure MMD improvement
   765                                                         as well as that the coreset does not contain any repeated points at any iteration)
   766                                               """
   767        86      52000.0    604.7      0.0      n = X.shape[0]
   768                                           
   769                                               # Initialize new KT coreset to original coreset
   770        86     494000.0   5744.2      0.0      coreset = np.copy(coreset)
   771        86      16000.0    186.0      0.0      coreset_size = len(coreset)
   772                                               
   773                                               # Initialize sufficient kernel matrix statistics
   774                                               # sufficient_stat = twoncoresumK + ndiagK - twomeanK where
   775                                               #   ndiagK[ii] = diagonal element of kernel matrix, kernel(X[ii], X[ii]) / coreset_size
   776                                               #   twomeanK[ii] = 2 * the mean of kernel(X[ii], X)
   777                                               #   twoncoresumK[ii] = 2 * the sum of kernel(X[ii], X[coreset]) / coreset_size
   778        86      33000.0    383.7      0.0      two_over_coreset_size = 2/coreset_size
   779                                               
   780                                               # Initialize coreset indicator of size n, takes value True at coreset indices
   781        86     159000.0   1848.8      0.0      coreset_indicator = np.zeros(n, dtype=bool)
   782        86     171000.0   1988.4      0.0      coreset_indicator[coreset] = True
   783                                               
   784        86      13000.0    151.2      0.0      if meanK is None:
   785                                                   sufficient_stat = np.empty(n)
   786                                                   for ii in range(n):
   787                                                       # if unique then set sufficient_stat for coreset indices to infinity
   788                                                       if unique and coreset_indicator[ii]:
   789                                                           sufficient_stat[ii] = np.inf
   790                                                       else:
   791                                                           # Kernel evaluation between xi and every row in X
   792                                                           kii =  kernel(X[ii, newaxis], X)
   793                                                           sufficient_stat[ii] = 2*(np.mean(kii[coreset])-np.mean(kii)) + kii[ii]/coreset_size               
   794                                               else:
   795                                                   # Initialize to kernel diagonal normalized by coreset_size - 2 * meanK
   796        86   73804000.0 858186.0      0.2          sufficient_stat = kernel(X, X)/coreset_size - 2 * meanK
   797                                                   # Add in contribution of coreset
   798     15872    2128000.0    134.1      0.0          for ii in range(n):
   799                                                       # if unique then set sufficient_stat for coreset indices to infinity
   800      8192    1153000.0    140.7      0.0              if unique and coreset_indicator[ii]:
   801      7680    1749000.0    227.7      0.0                  sufficient_stat[ii] = np.inf
   802                                                       else:
   803                                                           # Kernel evaluation between xi and every coreset row in X
   804      8192 6885223000.0 840481.3     21.2                  kiicore =  kernel(X[ii, newaxis], X[coreset])
   805      8192   45320000.0   5532.2      0.1                  sufficient_stat[ii] += 2*np.mean(kiicore)
   806                                               
   807                                               # Consider each coreset point in turn 
   808      7808    1323000.0    169.4      0.0      for coreset_idx in range(coreset_size):
   809                                                   # if unique have to compute sufficient stats for the current coreset point
   810      7680     712000.0     92.7      0.0          if unique:
   811                                                       # initially all coreset indices have sufficient_stat set to infinity; 
   812                                                       # before altering any coreset point, we compute the sufficient_stat for it, and
   813                                                       # compare replacing it with every other *non coreset* point; the best point (bp)
   814                                                       # then takes the spot of the point in consideration with bp's sufficient_stat set to infty.
   815                                                       # thus at each iteration of the for loop, all the current coreset elements except the one 
   816                                                       # in consideration have sufficient_stats set to infty
   817      7680    1460000.0    190.1      0.0              cidx = coreset[coreset_idx] # the index of coreset_idx in X
   818      7680    1024000.0    133.3      0.0              if meanK is None:
   819                                                           # Kernel evaluation between x at cidx and every row in X
   820                                                           kcidx = kernel(X[cidx, newaxis], X)
   821                                                           sufficient_stat[cidx] = 2*(np.mean(kcidx[coreset])-np.mean(kcidx)) + kcidx[cidx]/coreset_size    
   822                                                       else:
   823      7680 8501147000.0 1106920.2     26.2                  kcidxcore =  kernel(X[cidx, newaxis], X[coreset])
   824      7680  146018000.0  19012.8      0.4                  sufficient_stat[cidx] = kernel(X[cidx, newaxis], X[cidx, newaxis])/coreset_size - 2 * meanK[cidx] 
   825      7680   60051000.0   7819.1      0.2                  sufficient_stat[cidx] += 2*np.mean(kcidxcore)
   826                                                       
   827                                                   # Remove the contribution of coreset_idx point from the normalized coreset sum in sufficient stat
   828      7808 9048044000.0 1158817.1     27.9          sufficient_stat -= kernel(X[coreset[coreset_idx], newaxis], X)*two_over_coreset_size
   829                                                   # Find the input point that would reduce MMD the most
   830      7808   22502000.0   2881.9      0.1          best_point = np.argmin(sufficient_stat)
   831                                                   # Add best point to coreset and its contribution to sufficient stat
   832      7808    2127000.0    272.4      0.0          coreset[coreset_idx] = best_point
   833      7808 7672627000.0 982662.3     23.6          sufficient_stat += kernel(X[best_point, newaxis], X)*two_over_coreset_size
   834      7680    1118000.0    145.6      0.0          if unique:
   835      7680    4092000.0    532.8      0.0              sufficient_stat[best_point] = np.inf
   836        86       8000.0     93.0      0.0      return(coreset)
```