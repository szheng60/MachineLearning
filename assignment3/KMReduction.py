from sklearn.base import TransformerMixin,BaseEstimator
import matplotlib.pyplot as plt
from readData import *
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
from collections import defaultdict
from sklearn.decomposition import FastICA
from sklearn.metrics.cluster import completeness_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.random_projection import SparseRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def wineQuality():
    scaler = StandardScaler()
    dat = load_wine_quality_data()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    transformed = dataTransform(X, Y, 6, 8, 8, 4)
    km(clusters, transformed, Y)

def adultIncome():
    dat = load_adult_income_data()
    print(dat.columns)
    scaler = StandardScaler()
    # scaler = Normalizer()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    pca = PCA(n_components=9, random_state=0)
    transformed = dataTransform(X, Y, 9, 7, 8, 7)
    km(clusters, transformed, Y)

def dataTransform(X, Y, n_pca, n_ica, n_rp, n_rf):

    pca = PCA(n_components=n_pca, random_state=0)
    pca_after = pca.fit_transform(X)

    ica = FastICA(n_components=n_ica, random_state=11, whiten=True)
    ica_after = ica.fit_transform(X)

    rp = SparseRandomProjection(random_state=4, n_components=n_rp)
    rp_after = rp.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=5, n_jobs=7)
    filtr = ImportanceSelect(rf, n_rf)
    rf_after = filtr.fit_transform(X, Y)

    print(rf_after)

    return [pca_after, ica_after, rp_after, rf_after, X]


def km(clusters, dats, Y):
    NMI = defaultdict(dict)
    INL = defaultdict(dict)
    CMS = defaultdict(dict)
    SIL = defaultdict(dict)
    for i, dat in enumerate(dats):
        for cluster in clusters:
            km = KMeans(n_clusters=cluster, random_state=0).fit(dat)
            cluster_labels = km.labels_
            NMI[i][cluster] = normalized_mutual_info_score(Y, cluster_labels)
            INL[i][cluster] = km.inertia_
            CMS[i][cluster] = completeness_score(Y, cluster_labels)
            SIL[i][cluster] = silhouette_score(dat, cluster_labels)


    plt.plot(clusters, NMI[0].values(), 'bx-', color='C0')
    plt.plot(clusters, NMI[1].values(), 'bx-', color='C1')
    plt.plot(clusters, NMI[2].values(), 'bx-', color='C2')
    plt.plot(clusters, NMI[3].values(), 'bx-', color='C3')
    plt.plot(clusters, NMI[4].values(), 'bx-', color='C4')
    plt.legend(['PCA', 'ICA', 'RP', 'RF', 'Original'])
    plt.xlabel('k')
    plt.title('Normalized Mutual Information on K-Means')
    plt.show()

    plt.plot(clusters, INL[0].values(), 'bx-', color='C0')
    plt.plot(clusters, INL[1].values(), 'bx-', color='C1')
    plt.plot(clusters, INL[2].values(), 'bx-', color='C2')
    plt.plot(clusters, INL[3].values(), 'bx-', color='C3')
    plt.plot(clusters, INL[4].values(), 'bx-', color='C4')
    plt.legend(['PCA', 'ICA', 'RP', 'RF', 'Original'])
    plt.xlabel('k')
    plt.title('Elbow Method on K-Means')
    plt.show()

    plt.plot(clusters, CMS[0].values(), 'bx-', color='C0')
    plt.plot(clusters, CMS[1].values(), 'bx-', color='C1')
    plt.plot(clusters, CMS[2].values(), 'bx-', color='C2')
    plt.plot(clusters, CMS[3].values(), 'bx-', color='C3')
    plt.plot(clusters, CMS[4].values(), 'bx-', color='C4')
    plt.legend(['PCA', 'ICA', 'RP', 'RF', 'Original'])
    plt.xlabel('k')
    plt.title('Completeness on K-Means')
    plt.show()

    plt.plot(clusters, SIL[0].values(), 'bx-', color='C0')
    plt.plot(clusters, SIL[1].values(), 'bx-', color='C1')
    plt.plot(clusters, SIL[2].values(), 'bx-', color='C2')
    plt.plot(clusters, SIL[3].values(), 'bx-', color='C3')
    plt.plot(clusters, SIL[4].values(), 'bx-', color='C4')
    plt.legend(['PCA', 'ICA', 'RP', 'RF', 'Original'])
    plt.xlabel('k')
    plt.title('Silhouette on K-Means')
    plt.show()

class ImportanceSelect(BaseEstimator, TransformerMixin):
    def __init__(self, model, n=1):
         self.model = model
         self.n = n
    def fit(self, *args, **kwargs):
         self.model.fit(*args, **kwargs)
         return self
    def transform(self, X):
         return X[:,self.model.feature_importances_.argsort()[::-1][:self.n]]

if __name__=="__main__":
    # wineQuality()
    adultIncome()