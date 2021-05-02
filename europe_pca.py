import json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as StdScal
import numpy as np
import matplotlib.pyplot as plt

import utils

# Link with some code references
# https://www.aprendemachinelearning.com/comprende-principal-component-analysis/

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

# Scale data with media and normalize
x_scaled = StdScal().fit_transform(x)

# Apply PCA fitting data and applying the dimensionality reduction
pca = PCA()
x_pca = pca.fit_transform(x_scaled)

# Get first two principal components
pc1 = x_pca[:, 0]
pc2 = x_pca[:, 1]

# Save variance ratio for each components
exp_variance = pca.explained_variance_ratio_

# Save PCA components
components = pca.components_

################ PRINTING RESULTS ################

print("First principal component --> If > 0, positive loads are more important here")
for country, val in zip(y, pc1):
    print(f'{country}: {val}')

print("Second principal component")
for country, val in zip(y, pc2):
    print(f'{country}: {val}')

print("\nLoads 1 (component 1) for each Xi")
# Print PC1 loads
for h, load in zip(headers[1:], components[0]):
    print(f'{h}: {load}')

print("\nLoads 2 (component 2) for each Xi")
# Print PC1 loads
for h, load in zip(headers[1:], components[1]):
    print(f'{h}: {load}')

print("\nVariance ratio", exp_variance)

print("\nLoads for all components")
for i, eigenvector in enumerate(components):
    print(f'Loads {i + 1}: \n\t{eigenvector}')

################ PLOTTING RESULTS ################

if plot_boolean:
    # Plots
    utils.init_plotter()

    # Boxplot values from different variables to view each variance
    utils.plot_boxplot(x, headers[1:], 'x values')

    # Boxplot values from different scaled variables to view each variance
    utils.plot_boxplot(x_scaled, headers[1:], 'scaled x values')

    # Plot accumulated variance with n components
    utils.plot_values(range(len(exp_variance)), 'number of components', np.cumsum(exp_variance), 'cumulative variance', sci_y=False)

    # Plot horizontal bars with PC1 for each country
    utils.plot_horiz_bar(y, pc1, 'PC1')

    # Plot PC2 = f(PC1)
    utils.plot_two_components(pc1, pc2, components[0, :], components[1, :], headers[1:], y, scale=True)
    utils.plot_two_components(pc1, pc2, components[0, :], components[1, :], headers[1:], y, scale=False)

    utils.hold_execution()
