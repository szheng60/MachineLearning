import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from readData import *
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
from collections import defaultdict
from prettytable import PrettyTable


def wineQuality():
    scaler = MinMaxScaler()
    # scaler = Normalizer()
    dat = load_wine_quality_data()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    # clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    clusters = np.arange(2, 10)
    printTable(clusters, X_norm, Y, 11)
    # analyze(X_norm, Y, clusters, 10, 6)
    # testing(X_norm, Y)

def adultIncome():
    dat = load_adult_income_data()
    scaler = MinMaxScaler()
    # scaler = Normalizer()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    printTable(clusters, X_norm, Y, 7)
    # testing(X_norm, Y)

def printTable(clusters, X_norm, Y, r):
    # models = [GaussianMixture(n, covariance_type='full', random_state=r).fit(X_norm)
    #           for n in clusters]
    # simplePlot(clusters, [m.bic(X_norm) for m in models], 'BIC', 'Bayesian information criterion')
    # # simplePlot(clusters, [m.aic(X_norm) for m in models], 'AIC', 'Akaike information criterion')
    # simplePlot(clusters, [m.score(X_norm) for m in models], 'LLH', 'Log Likelihood')

    # plt.plot(clusters, [m.bic(X_norm) for m in models], label='BIC')
    # plt.plot(clusters, [m.aic(X_norm) for m in models], label='AIC')
    # plt.plot(clusters, [m.score(X_norm) for m in models], label='LLH')
    # plt.legend(loc='best')
    # plt.xlabel('K')
    # plt.show()
    NMI = defaultdict(dict)
    # INL = defaultdict(dict)
    # SSE = defaultdict(dict)
    # ACC = defaultdict(dict)
    # AMI = defaultdict(dict)
    # ARI = defaultdict(dict)
    # CMS = defaultdict(dict)
    SIL = defaultdict(dict)
    # BIC = defaultdict(dict)
    # AIC = defaultdict(dict)
    # LLH = defaultdict(dict)
    for cluster in clusters:
        gm = GaussianMixture(n_components=cluster, random_state=r).fit(X_norm)
        cluster_labels = gm.predict(X_norm)
        NMI[cluster] = normalized_mutual_info_score(Y, cluster_labels)
        # # SSE[cluster] = km.score(X)
        # INL[cluster] = sum(np.min(cdist(X_norm, gm.means_, 'euclidean'), axis=1)) / X_norm.shape[0]
        # # SSE[cluster] = sum(np.min(cdist(X_norm, km.cluster_centers_, 'euclidean'), axis=1))
        # # ACC[cluster] = cluster_acc(Y, cluster_labels)
        # # AMI[cluster] = ami(Y, cluster_labels)
        # # ARI[cluster] = ari(Y, cluster_labels)
        # CMS[cluster] = completeness_score(Y, cluster_labels)
        SIL[cluster] = silhouette_score(X_norm, cluster_labels)
    # table = PrettyTable(['cluster', 'NMI', 'SSE', 'ACC', 'AMI', 'ARI', 'CMS', 'SIL'])
    table = PrettyTable(['cluster', 'NMI', 'SIL'])
    for c in clusters:
        table.add_row([c, NMI[c], SIL[c]])
    print(table)
    simplePlot(clusters, NMI.values(), 'NMI', 'Normalized Mutual Information')
    # simplePlot(clusters, AIC.values(), 'AIC', 'Akaike information criterion')
    # simplePlot(clusters, LLH.values(), 'LLH', 'Log Likelihood')
    simplePlot(clusters, SIL.values(), 'SIL', 'Silhouette')

def simplePlot(K, dat, name, title):
    plt.plot(K, dat, 'bx-')
    plt.xlabel('k')
    plt.ylabel(name)
    plt.title(title)
    plt.show()

if __name__=="__main__":
    # wineQuality()
    adultIncome()