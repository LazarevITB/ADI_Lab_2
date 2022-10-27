import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sys

sys.path.append("..")

#Чтение данных
df = pd.read_csv('data/close_prices.csv')
df.head

X = df.loc[:, "AXP":]

#Обучение PCA
pca = PCA(n_components=10)
pca.fit(X)
sum_var = 0
for i, v in enumerate(pca.explained_variance_ratio_):
    sum_var += v
    if sum_var >= 0.9:
        break

print(1, str(i+1))

#Применение преобразований
X0 = pd.DataFrame(pca.transform(X))[0]
X0.head()

#Загружаем инфу
df2 = pd.read_csv("data/djia_index.csv")
df2.head()

corr = np.corrcoef(X0, df2["^DJI"])
print(2, f"{corr[1, 0]:.2f}")

#Определяем наибольший вес
print(3, X.columns[np.argmax(pca.components_[0])])