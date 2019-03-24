import pandas as pd
import numpy as np

if __name__=="__main__":
    file = "adult_nn_"
    base = pd.read_csv(file + "base.csv")
    em = pd.read_csv(file + "em.csv")
    km = pd.read_csv(file + "km.csv")
    pca = pd.read_csv(file + "pca.csv")
    ica = pd.read_csv(file + "ica.csv")
    rp = pd.read_csv(file + "rp.csv")
    rf = pd.read_csv(file + "rf.csv")
    km_rp = pd.read_csv(file + "km_rp.csv")