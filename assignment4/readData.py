import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load(fileName):
    csv_path = os.path.join("/home/song/Documents/temp/assignment4", fileName)
    dat = pd.read_csv(csv_path, names=['episode', 'steps', 'reward', 'time', 'convergence'], skiprows=1)

    return dat


