import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_wine_quality_data():
    csv_path = os.path.join("datasets", "winequality.csv")
    dat = pd.read_csv(csv_path)

    # X = dat.iloc[:, :-1]
    # Y = dat.iloc[:, -1]
    #
    # train, test = train_test_split(dat, random_state=0, test_size=0.25)
    # train.to_csv("winequality-train.csv", index=False)
    # test.to_csv("winequality-test.csv", index=False)
    #
    #
    # print(dat.columns)
    # return pd.read_csv(csv_path)
    return dat

def load_adult_income_data():
    csv_path=os.path.join("datasets", "adult.csv")
    dat = pd.read_csv(csv_path)

    # check unknown
    # checkUnknown(dat, "?")
    # remove data with '?'
    dat = dat[dat["workclass"] != "?"]
    dat = dat[dat["occupation"] != "?"]
    dat = dat[dat["native-country"] != "?"]

    dat.drop(['fnlwgt'], axis=1, inplace=True)

    dat['workclass'].replace('Federal-gov', 'Employed', inplace=True)
    dat['workclass'].replace('Local-gov', 'Employed', inplace=True)
    dat['workclass'].replace('Self-emp-inc', 'Employed', inplace=True)
    dat['workclass'].replace('Self-emp-not-inc', 'Employed', inplace=True)
    dat['workclass'].replace('State-gov', 'Employed', inplace=True)
    dat['workclass'].replace('Private', 'Employed', inplace=True)
    dat['workclass'].replace('Never-worked', 'Unemployed', inplace=True)
    dat['workclass'].replace('Without-pay', 'Unemployed', inplace=True)

    dat['marital-status'].replace('Never-married', 'NotMarried', inplace=True)
    dat['marital-status'].replace(['Married-AF-spouse'], 'Married', inplace=True)
    dat['marital-status'].replace(['Married-civ-spouse'], 'Married', inplace=True)
    dat['marital-status'].replace(['Married-spouse-absent'], 'NotMarried', inplace=True)
    dat['marital-status'].replace(['Separated'], 'NotMarried', inplace=True)
    dat['marital-status'].replace(['Divorced'], 'NotMarried', inplace=True)
    dat['marital-status'].replace(['Widowed'], 'NotMarried', inplace=True)

    dat['education'].replace('Preschool', 'dropout', inplace=True)
    dat['education'].replace('10th', 'dropout', inplace=True)
    dat['education'].replace('11th', 'dropout', inplace=True)
    dat['education'].replace('12th', 'dropout', inplace=True)
    dat['education'].replace('1st-4th', 'dropout', inplace=True)
    dat['education'].replace('5th-6th', 'dropout', inplace=True)
    dat['education'].replace('7th-8th', 'dropout', inplace=True)
    dat['education'].replace('9th', 'dropout', inplace=True)
    dat['education'].replace('HS-Grad', 'HighGrad', inplace=True)
    dat['education'].replace('HS-grad', 'HighGrad', inplace=True)
    dat['education'].replace('Some-college', 'Associates', inplace=True)
    dat['education'].replace('Assoc-acdm', 'Associates', inplace=True)
    dat['education'].replace('Assoc-voc', 'Associates', inplace=True)
    dat['education'].replace('Bachelors', 'Bachelors', inplace=True)
    dat['education'].replace('Masters', 'Masters', inplace=True)
    dat['education'].replace('Prof-school', 'Masters', inplace=True)
    dat['education'].replace('Doctorate', 'Doctorate', inplace=True)

    northAmerica = ["Canada", "Cuba", "Dominican-Republic", "El-Salvador", "Guatemala",
                   "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua",
                   "Outlying-US(Guam-USVI-etc)", "Puerto-Rico", "Trinadad&Tobago",
                   "United-States"]

    asia = ["Cambodia", "China", "Hong", "India", "Iran", "Japan", "Laos",
          "Philippines", "Taiwan", "Thailand", "Vietnam"]

    southAmerica = ["Columbia", "Ecuador", "Peru"]

    europe = ["England", "France", "Germany", "Greece", "Holand-Netherlands",
            "Hungary", "Ireland", "Italy", "Poland", "Portugal", "Scotland",
            "Yugoslavia"]

    other = ["South"]

    dat.loc[dat['native-country'].isin(northAmerica), 'native-country'] = "North America"
    dat.loc[dat['native-country'].isin(asia), 'native-country'] = "Asia"
    dat.loc[dat['native-country'].isin(southAmerica), 'native-country'] = "South America"
    dat.loc[dat['native-country'].isin(europe), 'native-country'] = "Europe"
    dat.loc[dat['native-country'].isin(other), 'native-country'] = "Other"

    dat['income'].replace('<=50K', 0, inplace=True)
    dat['income'].replace('>50K', 1, inplace=True)

    columns_to_encoding = ['workclass', 'marital-status', 'occupation',
                           'relationship', 'race', 'gender', 'education', 'native-country']


    le = LabelEncoder()
    for column in columns_to_encoding:
        dat[column] = le.fit_transform(dat[column])
    return dat
    # return dat.iloc[1:, :]

def checkUnknown(dat, unknown):
    for c in dat.columns:
        num_non = dat[c].isin([unknown]).sum()
        if num_non > 0:
            print (c)
            print (num_non)


if __name__=="__main__":
    load_wine_quality_data()