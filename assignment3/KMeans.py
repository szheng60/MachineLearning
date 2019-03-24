import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from readData import *
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
from collections import defaultdict
from sklearn.metrics import accuracy_score as acc
from collections import Counter
from prettytable import PrettyTable
from sklearn.metrics.cluster import completeness_score


def wineQuality(dat):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    printTable(clusters, X_norm, Y)
    # analyze(X_norm, Y, clusters, 10, 6)
    # testing(X_norm, Y)

def adultIncome(dat):
    scaler = MinMaxScaler()
    # scaler = Normalizer()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    printTable(clusters, X_norm, Y)
    # testing(X_norm, Y)

def testing(X_norm, Y, clusters=3):
    km = KMeans(n_clusters=clusters, random_state=0).fit(X_norm)
    cluster_labels = km.labels_
    cmap = cm.get_cmap("Spectral")
    colors = cmap(cluster_labels.astype(float) / clusters)
    i=12

    for f1 in range(i, i+1):
        for f2 in range(0, 13):
            if f1 != f2:
                plt.scatter(X_norm.iloc[:, f1], X_norm.iloc[:, f2], marker='.', s=30, lw=0, alpha=0.7,
                            c=colors, edgecolor='k')

                # Labeling the clusters
                centers = km.cluster_centers_

                # Draw white circles at cluster centers
                plt.scatter(centers[:, f1], centers[:, f2], marker='o',
                            c="white", alpha=1, s=200, edgecolor='k')

                for i, c in enumerate(centers):
                    plt.scatter(c[f1], c[f2], marker='$%d$' % i, alpha=1,
                                s=50, edgecolor='k')

                # plt.set_title("The visualization of the clustered data.")
                # plt.set_xlabel("Feature space for the 1st feature")
                # plt.set_ylabel("Feature space for the 2nd feature")

                plt.title("Silhouette analysis for KMeans clustering on sample data "
                              "with n_clusters = {}-{}-{}".format(clusters, f1, f2),
                             fontsize=10, fontweight='bold')

                plt.show()

def analyze(X_norm, Y, clusters, f1, f2):
    # printTable(clusters, X_norm, Y)
    for cluster in clusters:
        km = KMeans(n_clusters=cluster, random_state=0).fit(X_norm)
        cluster_labels = km.labels_

        plotFig(X_norm, cluster, cluster_labels, km, f1, f2)


def plotFig(X_norm, cluster, cluster_labels, km, f1, f2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X_norm) + (cluster + 1) * 10])
    silhouette_avg = silhouette_score(X_norm, cluster_labels)
    sample_silhouette_values = silhouette_samples(X_norm, cluster_labels)
    y_lower = 10
    for i in range(cluster):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        cmap = cm.get_cmap("Spectral")
        color = cmap(float(i) / cluster)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    # 2nd Plot showing the actual clusters formed
    cmap = cm.get_cmap("Spectral")
    colors = cmap(cluster_labels.astype(float) / cluster)

    ax2.scatter(X_norm.iloc[:, f1], X_norm.iloc[:, f2], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    # Labeling the clusters
    centers = km.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, f1], centers[:, f2], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
    for i, c in enumerate(centers):
        ax2.scatter(c[f1], c[f2], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % cluster),
                 fontsize=14, fontweight='bold')
    plt.show()


def printTable(clusters, X_norm, Y):
    NMI = defaultdict(dict)
    INL = defaultdict(dict)
    # SSE = defaultdict(dict)
    # ACC = defaultdict(dict)
    # AMI = defaultdict(dict)
    # ARI = defaultdict(dict)
    CMS = defaultdict(dict)
    SIL = defaultdict(dict)
    for cluster in clusters:
        km = KMeans(n_clusters=cluster, random_state=0).fit(X_norm)
        cluster_labels = km.labels_
        NMI[cluster] = normalized_mutual_info_score(Y, cluster_labels)
        # SSE[cluster] = km.score(X)
        INL[cluster] = km.inertia_
        # SSE[cluster] = sum(np.min(cdist(X_norm, km.cluster_centers_, 'euclidean'), axis=1))
        # ACC[cluster] = cluster_acc(Y, cluster_labels)
        # AMI[cluster] = ami(Y, cluster_labels)
        # ARI[cluster] = ari(Y, cluster_labels)
        CMS[cluster] = completeness_score(Y, cluster_labels)
        SIL[cluster] = silhouette_score(X_norm, cluster_labels)
    # table = PrettyTable(['cluster', 'NMI', 'SSE', 'ACC', 'AMI', 'ARI', 'CMS', 'SIL'])
    table = PrettyTable(['cluster', 'NMI', 'INL', 'CMS', 'SIL'])
    for c in clusters:
        table.add_row([c, NMI[c], INL[c], CMS[c], SIL[c]])
    print(table)
    simplePlot(clusters, NMI.values(), 'NMI', 'Normalized Mutual Information')
    simplePlot(clusters, INL.values(), 'INL', 'Elbow Method')
    simplePlot(clusters, CMS.values(), 'CMS', 'Completeness ')
    simplePlot(clusters, SIL.values(), 'SIL', 'Silhouette')

def simplePlot(K, dat, name, title):
    plt.plot(K, dat, 'bx-')
    plt.xlabel('k')
    plt.ylabel(name)
    plt.title(title)
    plt.show()

def cluster_acc(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    return acc(Y,pred)


if __name__=="__main__":
    # wineQuality(load_wine_quality_data())
    adultIncome(load_adult_income_data())
