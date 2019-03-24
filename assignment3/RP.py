import numpy as np
import matplotlib.pyplot as plt
from readData import *
from collections import defaultdict
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import StandardScaler
from itertools import product
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse as sps
from scipy.linalg import pinv

def wineQuality():
    scaler = StandardScaler()
    dat = load_wine_quality_data()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    performRP(X, 'wine_rp')


def performRP(X, name):
    dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13]
    dist_corr = defaultdict(dict)
    # rec_err = defaultdict(dict)
    for i, dim in product(range(10), dims):
        rp = SparseRandomProjection(random_state=i, n_components=dim)
        # rp.fit(X)
        # rec_err[dim][i] = reconstructionError(rp, X)
        dist_corr[dim][i] = pairwiseDistCorr(rp.fit_transform(X), X)
    # rec_err = pd.DataFrame(rec_err).T
    dist_corr = pd.DataFrame(dist_corr).T
    # rec_err.to_csv(name + '_rec_err.csv')
    dist_corr.to_csv(name + '_dist_corr.csv')
    # print(tmp)


def pairwiseDistCorr(X1, X2):
    assert X1.shape[0] == X2.shape[0]

    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)

    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]

def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)

def adultIncome():
    dat = load_adult_income_data()
    scaler = StandardScaler()
    # scaler = Normalizer()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    performRP(X, 'adult_rp')

def plotDistCorrFig(file):
    dat = pd.read_csv(file)
    dat = dat.iloc[:, 1:]
    dat.plot()
    plt.xlabel("Components")
    plt.ylabel("Distance Correlation")
    plt.title("")
    plt.show()

def plotRecErrFig(file, title):
    dat = pd.read_csv(file)
    dat = dat.iloc[:, 1:]
    dat.plot()
    plt.ylabel("Reconstruction Error")
    plt.xlabel("Components")
    plt.title("RP on " + title)
    plt.show()

def plotFig(name, title):
    # plotDistCorrFig(name + "_rp_dist_corr.csv")
    plotRecErrFig(name + "_rp_rec_err.csv", title)

if __name__=="__main__":
    # wineQuality()
    # adultIncome()
    plotFig("wine", "wine quality")
    plotFig("adult", "adult income")