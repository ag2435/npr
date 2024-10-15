from sklearn import datasets
import numpy as np

def normalize(func):
    def wrapper(normalize):
        if normalize:
            X, y = func(normalize)

            X_mean = X.mean(0, keepdims=True)
            X_std = X.std(0, keepdims=True)
            X -= X_mean
            X /= X_std

            y_mean = y.mean(0, keepdims=True)
            y_std = y.std(0, keepdims=True)
            y = y.astype(float)
            y -= y_mean
            y /= y_std

            return X, y
        else:
            return func(normalize)
    return wrapper

def add_noise(func):
    """
    Add random noise to y so KRR is less sensitive to regularization parameter
    """
    def wrapper(normalize, noisey):
        X, y = func(normalize, noisey)
        y += noisey * np.random.randn(*y.shape)
        return X, y
    return wrapper

# @add_noise
# @normalize
def get_housing_dataset():
    X,y= datasets.fetch_california_housing(return_X_y=True)
    print(X.shape, y.shape)
    # normalize X by dividing by norm
    # print("normalizing X")
    # norm = np.linalg.norm(X, axis=1, keepdims=True)
    # print(norm.shape)
    # X = X/norm

    # normalize X by subtracting mean and dividing by std
    print("normalizing X")
    X_mean = X.mean(0, keepdims=True)
    X_std = X.std(0, keepdims=True)
    X -= X_mean
    X /= X_std
    
    return X, y

def get_svhn_dataset(n_samples, normalize=True):
    """
    Return dataset as np.arrays since we need to pass it into kernel thinning, 
    which only takes numpy arrays

    Ref:
    - https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html#svhn
    - http://ufldl.stanford.edu/housenumbers/
    
    10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
    73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat less difficult samples, to use as extra training data
    Comes in two formats:
    1. Original images with character level bounding boxes.
    2. MNIST-like 32-by-32 images centered around a single character (many of the images do contain some distractors at the sides).
    """
    # import torch
    import torchvision
    import torchvision.transforms as transforms

    def set_data_path():
        return "datasets/svhn"
    
    def pre_process(torchset,n_samples,num_classes=10):
        # indices = list(np.random.choice(len(torchset),n_samples))
        indices = range(len(torchset))

        trainset = []
        for ix in indices:
            x,y = torchset[ix]
            ohe_y = np.zeros(num_classes)
            ohe_y[y] = 1

            x_ = x/np.linalg.norm(x) if normalize else x
            trainset.append((np.reshape(x_,-1),ohe_y))
        return trainset

    data_path = set_data_path() ## set this data path

    trainset0 = torchvision.datasets.SVHN(root=data_path,
                                        split = "train",
                                        download=True)
    testset0 = torchvision.datasets.SVHN(root=data_path,
                                        split = "test",
                                        download=True)
    print(f"trainset0: {len(trainset0)}, testset0: {len(testset0)}")
    trainset = pre_process(trainset0,n_samples=n_samples, num_classes=10)
    X_train, y_train = zip(*trainset)

    testset = pre_process(testset0,n_samples=n_samples, num_classes=10)
    X_test, y_test = zip(*testset)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def get_svhn67_dataset(n_samples, normalize=True):
    """
    Binary classification version of SVHN dataset (using only digits 6 and 7)
    """
    # import torch
    import torchvision
    import torchvision.transforms as transforms

    def set_data_path():
        return "datasets/svhn"

    def pre_process(torchset,n_samples):
        a, b = 6, 7
        six_sevens = [ix for ix,y in enumerate(torchset) if y[1]==a or y[1]==b]
        indices = list(np.random.choice(six_sevens,n_samples))

        trainset = []
        for ix in indices:
            x,y = torchset[ix]
            x_ = x/np.linalg.norm(x) if normalize else x
            trainset.append( (np.reshape(x_, -1), 0 if y==a else 1) )
        return trainset

    data_path = set_data_path() ## set this data path

    trainset0 = torchvision.datasets.SVHN(root=data_path,
                                        split = "train",
                                        download=True)
    testset0 = torchvision.datasets.SVHN(root=data_path,
                                        split = "test",
                                        download=True)
    print(f"trainset0: {len(trainset0)}, testset0: {len(testset0)}")
    trainset = pre_process(trainset0,n_samples=n_samples)
    X_train, y_train = zip(*trainset)

    testset = pre_process(testset0,n_samples=n_samples)
    X_test, y_test = zip(*testset)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def get_susy_dataset():
    """
    Dataset information:
    The data has been produced using Monte Carlo simulations. The first 8 features 
    are kinematic properties measured by the particle detectors in the accelerator. 
    The last ten features are functions of the first 8 features; these are high-level 
    features derived by physicists to help discriminate between the two classes. 
    There is an interest in using deep learning methods to obviate the need for 
    physicists to manually develop such features. Benchmark results using Bayesian 
    Decision Trees from a standard physics package and 5-layer neural networks and 
    the dropout algorithm are presented in the original paper. The last 500,000 
    examples are used as a test set.n about your data set.
    """
    import pandas as pd
    # hardcode path to data for now
    DATA_PATH = "/share/nikola/ag2435/kt_regression/src/npr/examples/neurips/data/SUSY.csv"
    # load as csv
    df = pd.read_csv(DATA_PATH, header=None)
    # get first column of labels as integer
    y = df.iloc[:,0].values.astype(int)
    # get the rest of the columns as features
    X = df.iloc[:,1:].values
    return X, y

def get_real_dataset(name): #, normalize=True, n_samples=20000):
    if name == 'housing':
        return get_housing_dataset()
    # elif name == 'msd':
    #     return get_msd_dataset(normalize=normalize, noisey=noisey)
    # elif name == 'taxi':
    #     # TODO
    #     raise NotImplementedError('taxi dataset not implemented')
    # elif name == 'svhn':
    #     return get_svhn_dataset(n_samples, normalize)
    # elif name == 'svhn67':
    #     return get_svhn67_dataset(n_samples, normalize)
    elif name == 'susy':
        return get_susy_dataset()
    else:
        raise ValueError(f'dataset={name} not implememented')
    


# def get_data(url, filename='data.txt'):
#     r = requests.get(url, allow_redirects=True)
#     with open(filename, 'wb') as fh:
#         fh.write(r.content)
# Eventually, try something like:
# https://adamkoscielniak.wordpress.com/2017/04/27/loading-a-dataframe-from-box-api-in-python/

# @add_noise
# @normalize
# def get_msd_dataset(normalize=True, noisey=0):
#     """
#     Million Song Dataset:
#     https://archive.ics.uci.edu/dataset/203/yearpredictionmsd

#     NOTE:
#     You should respect the following train / test split:
#     train: first 463,715 examples
#     test: last 51,630 examples
#     It avoids the 'producer effect' by making sure no song
#     from a given artist ends up in both the train and test set.
#     """

#     filename = 'datasets/YearPredictionMSD.txt'
#     df = pd.read_csv(filename, header=None)
#     X = df.iloc[:, 1:].values
#     y = df.iloc[:, 0].values
#     return X, y

# @add_noise
# @normalize
# def get_airlines_dataset(normalize=True):
#     # NOTE: this is the wrong dataset because it is categorical
#     return datasets.fetch_openml('airlines', return_X_y=True, version=1)
