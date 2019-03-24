import matplotlib.pyplot as plt
from readData import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def wineQuality():
    scaler = StandardScaler()
    dat = load_wine_quality_data()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    performPCA(X, "wine quality")


def performPCA(X, title):
    pca = PCA()
    pca.fit(X)
    tmp = pd.Series(data=pca.explained_variance_)
    tmp1 = pd.Series(data=pca.explained_variance_ratio_)
    tmp2 = pd.Series(data=pca.explained_variance_ratio_.cumsum())
    df = pd.concat([tmp, tmp1, tmp2], axis=1)
    df.columns = ['variance', 'variance_ratio', 'variance_cum']
    print(df)
    df.plot()
    # plt.plot()
    plt.title("PCA on " + title)
    plt.xlabel('number of components')
    # plt.ylabel('cumulative explained variance')
    plt.show()


def adultIncome():
    dat = load_adult_income_data()
    print(dat.columns)
    scaler = StandardScaler()
    # scaler = Normalizer()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    performPCA(X, "adult income")

if __name__=="__main__":
    # wineQuality()
    adultIncome()