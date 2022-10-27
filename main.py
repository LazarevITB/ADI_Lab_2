import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sys

sys.path.append("..")

#Чтение данных
df = pd.read_csv('data/close_prices.csv')
df.head

X = df.loc[:, "AXP":]
