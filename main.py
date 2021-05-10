import json
import numpy as np
import perceptron as p
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# read config file
with open("config.json") as file:
    config = json.load(file)

data_filepath: str = config["data_file"]
eta: float = config["eta"]
epoch: int = config["epoch"]

# read data_file csv
full_data = pd.read_csv(data_filepath)

# retrieve headers
headers = full_data.columns.tolist()

# retrieve numeric values, scale data with media and normalize
data_scaled = StandardScaler().fit_transform(full_data.iloc[:, 1:].values)

# initialize neuron
neuron: p.SimplePerceptron = p.SimplePerceptron(eta=eta, w0=np.random.uniform(0.0, 0.2, len(data_scaled[0])))

# train neuron
for _ in range(0, epoch):
    for index in range(0, len(data_scaled)):
        neuron.train(data_scaled[index])

# get first component with PCA from neuron
neuron_pca = neuron.get_normalized_w()

# get first component with PCA from external library
pca = PCA()
pca.fit_transform(data_scaled)
library_pca = pca.components_[0]

print("Neuron PCA: " + str(neuron_pca))
print("Library PCA: " + str(library_pca))
