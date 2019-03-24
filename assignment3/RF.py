import matplotlib.pyplot as plt
from readData import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def wineQuality():
    scaler = StandardScaler()
    dat = load_wine_quality_data()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    performRF(X, Y, 'wine_rf')


def performRF(X, Y, name):
    rf = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)

    sel = SelectFromModel(rf)
    sel.fit(X, Y)
    q = sel.get_support()
    print(q)

    fs = rf.fit(X, Y).feature_importances_

    print(fs)

    tmp = pd.Series(fs)

    print(tmp)
    # tmp = pd.Series(np.sort(fs)[::-1])
    tmp.to_csv(name + '.csv')


def adultIncome():
    dat = load_adult_income_data()
    scaler = StandardScaler()
    # scaler = Normalizer()
    X = scaler.fit_transform(dat.iloc[:, :-1])
    X_norm = pd.DataFrame(X)
    Y = dat.iloc[:, -1]

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    performRF(X, Y, 'adult_rf')

def plotFeatureImportances(file, title):
    dat = pd.read_csv(file, header=None)
    dat = dat.iloc[:, 1:]
    dat.plot()
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.title("RF on " + title)
    plt.show()

def plotFig(name, title):
    # plotDistCorrFig(name + "_rp_dist_corr.csv")
    plotFeatureImportances(name + "_rf.csv", title)

if __name__=="__main__":
    # wineQuality()
    adultIncome()
    # plotFig("wine", "wine quality")
    plotFig("adult", "adult income")