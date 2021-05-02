import json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as StdScal
import numpy as np # linear algebra
import matplotlib.pyplot as plt

import utils

# Del conjunto de datos europe.csv ...
# - Calcular las componentes principales.
# - Interpretar la primera componente.
# - Realizar el gráfico biplot e interpretarlo.

# Read configuration file
with open("config.json") as file:
    config = json.load(file)

plot_boolean = utils.read_config_param(
    config, "plot", lambda el : bool(el), lambda el : True)

# Read data_file csv
eu = pd.read_csv(config["data_file"])
headers = eu.columns.tolist()

# Take all rows, all cols but 0 as X
x = eu.iloc[:,1:].values
# Take all rows, first col as Y (Countries, dependant value)
y = eu.iloc[:,0].values

# Boxplot values from different variables to view each variance
plt.boxplot(x)
plt.xticks(range(1, x.shape[1] + 1), headers[1:])

# Scale data with media and normalize
x_scaled = StdScal().fit_transform(x)

# Boxplot values from different variables to view each variance
plt.boxplot(x_scaled)
plt.xticks(range(1, x_scaled.shape[1] + 1), headers[1:])

# Apply PCA fitting data and applying the dimensionality reduction
pca = PCA()
x_pca = pca.fit_transform(x_scaled)

# Print first component
first_cmp = x_pca[:, 0]
second_cmp = x_pca[:, 1]

print("First component")
for country, pc1 in zip(y, first_cmp):
    print(f'{country}: {pc1}')

fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(y, first_cmp)
plt.grid()
plt.show(block=False)

# Save variance ratio for each components
exp_variance = pca.explained_variance_ratio_

# Save PCA components
components = pca.components_
print("\nLoads 1 for each Xi")
# Print PC1 loads
for h, load in zip(headers[1:], components[0]):
    print(f'{h}: {load}')

print("\nVariance ratio", exp_variance)

print("\nLoads for all components")
for i, eigenvector in enumerate(components):
    print(f'Loads {i + 1}: \n\t{eigenvector}')


def myplot(fc, sc,coeff,var_labels,val_labels):
    fig, ax = plt.subplots(figsize=(12, 10))
    xs = fc
    ys = sc
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    ax.scatter(xs * scalex, ys * scaley)
    for i in range(len(xs)):
        ax.annotate(val_labels[i], (xs[i] * scalex, ys[i] * scaley), fontsize=10)

    for i in range(n):
        ax.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'purple', alpha=0.5)
        ax.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, var_labels[i], color = 'orange', ha = 'center', va = 'center', fontsize=15)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))
    plt.grid()
    plt.show(block=False)

if plot_boolean:
    # Plots
    utils.init_plotter()

    # Plot accumulated variance with n components
    utils.plot_values(range(len(exp_variance)), 'number of components', np.cumsum(exp_variance), 'cumulative variance', sci_y=False)

    myplot(first_cmp, second_cmp, np.transpose(components[0:2, :]), headers[1:], y)

    utils.hold_execution()
