import json
import pandas as pd
from sklearn.decomposition import PCA as PCA
from sklearn.preprocessing import StandardScaler as prePCA
import numpy as np # linear algebra
import matplotlib.pyplot as plt

# Del conjunto de datos europe.csv ...
# - Calcular las componentes principales.
# - Interpretar la primera componente.
# - Realizar el gráfico biplot e interpretarlo.
with open("config.json") as file:
    config = json.load(file)

eu = pd.read_csv('europe.csv')
x = eu.iloc[:,1:8].values
y = eu.iloc[:,0].values

print(x)
print(y)

# standard = 
# pca = PCA().fit(standard)
