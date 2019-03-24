import matplotlib.pyplot as plt
from readData import *
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

def wineQuality():
    scaler = StandardScaler()
    dat = load_wine_quality_data()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    performICA(X, 'wine quality')


def performICA(X, title):
    ica = FastICA(random_state=11, whiten=True)
    kurt = {}
    dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13]
    for d in dims:
        ica.set_params(n_components=d)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt[d] = tmp.abs().mean()
    kurt = pd.Series(kurt)
    kurt.plot()
    plt.xlabel("K")
    plt.title("ICA on " + title)
    plt.show()


def adultIncome():
    dat = load_adult_income_data()
    scaler = StandardScaler()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    performICA(X, 'adult income')

if __name__=="__main__":
    wineQuality()
    # adultIncome()