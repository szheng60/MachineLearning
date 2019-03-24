from sklearn.base import TransformerMixin,BaseEstimator
from readData import *
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd

nn_arch= [(20,10),(20,),(10,),(10,10)]
nn_reg = [10**-x for x in range(2,5)]

def wineQuality():
    scaler = StandardScaler()
    dat = load_wine_quality_data()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    print("wine km")
    km(clusters, X, Y, "wine")
    print("wine em")
    em(clusters, X, Y, "wine")
    print("wine pca")
    pca(clusters, X, Y, "wine")
    print("wine ica")
    ica(clusters, X, Y, "wine")
    print("wine rp")
    rp(clusters, X, Y, "wine")
    print("wine rf")
    rf(clusters, X, Y, "wine")

def adultIncome():
    dat = load_adult_income_data()
    print(dat.columns)
    scaler = StandardScaler()
    # scaler = Normalizer()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # print("adult km")
    # km(clusters, X, Y, "adult")
    # print("adult em")
    # em(clusters, X, Y, "adult")
    # print("adult pca")
    # pca(clusters, X, Y, "adult")
    # print("adult ica")
    # ica(clusters, X, Y, "adult")
    # print("adult rp")
    # rp(clusters, X, Y, "adult")
    # print("adult rf")
    # rf(clusters, X, Y, "adult")
    # base(clusters, X, Y, "adult")

    RPAndKM(clusters, X, Y, "adult")

def RPAndKM(clusters, X, Y, name):
    rp = SparseRandomProjection(random_state=4, n_components=9)
    rp_after = rp.fit_transform(X)

    grid = {'km__n_clusters': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=500, early_stopping=True, random_state=5)
    km = kmeans(random_state=5)
    pipe = Pipeline([('km', km), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(rp_after, Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(name + '_nn_km_rp.csv')


def base(clusters, X, Y, name):
    grid = {'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=500, early_stopping=True, random_state=5)
    pipe = Pipeline([('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(X, Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(name + '_nn_base.csv')

def km(clusters, X, Y, name):
    grid = {'km__n_clusters': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=500, early_stopping=True, random_state=5)
    km = kmeans(random_state=5)
    pipe = Pipeline([('km', km), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10)

    gs.fit(X, Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(name + '_nn_km.csv')

def em(clusters, X, Y, name):
    grid = {'gmm__n_components': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=500, early_stopping=True, random_state=5)
    gmm = myGMM(random_state=5)
    pipe = Pipeline([('gmm', gmm), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(X, Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(name + '_nn_em.csv')

def pca(dims, X, Y, name):
    grid = {'pca__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    pca = PCA(random_state=5)
    mlp = MLPClassifier(activation='relu', max_iter=500, early_stopping=True, random_state=5)
    pipe = Pipeline([('pca', pca), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(X, Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(name + '_nn_pca.csv')

def ica(dims, X, Y, name):
    grid = {'ica__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    ica = FastICA(random_state=5)
    mlp = MLPClassifier(activation='relu', max_iter=500, early_stopping=True, random_state=5)
    pipe = Pipeline([('ica', ica), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(X, Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(name + '_nn_ica.csv')

def rp(dims, X, Y, name):
    grid = {'rp__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    rp = SparseRandomProjection(random_state=5)
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([('rp', rp), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(X, Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(name + '_nn_rp.csv')

def rf(dims, X, Y, name):
    rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=5, n_jobs=7)
    filtr = ImportanceSelect(rfc)
    grid = {'filter__n': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([('filter', filtr), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(X, Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(name + '_nn_rf.csv')

class myGMM(GMM):
    def transform(self,X):
        return self.predict_proba(X)

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