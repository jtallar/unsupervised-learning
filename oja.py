import json
import numpy as np
import perceptron as p
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# accumulation function for oja algorithm
def accum_function(data: np.ndarray, w: np.ndarray, e: float) -> np.ndarray:
    # out for this neuron
    y = np.dot(data, w)

    # calculate the delta w (oja algorithm)
    return e * y * (data - y * w)


# read config file
with open("config.json") as file:
    config = json.load(file)

data_filepath: str = config["data_file"]
eta: float = config["oja"]["eta"]
epoch: int = config["oja"]["epoch"]

# read data_file csv
full_data = pd.read_csv(data_filepath)

# retrieve numeric values, scale data with media and normalize
data_scaled = StandardScaler().fit_transform(full_data.iloc[:, 1:].values)

# w0 for neuron initialization
w0 = np.random.uniform(0.0, 0.2, len(data_scaled[0]))

# initialize neuron
neuron: p.SimplePerceptron = p.SimplePerceptron(w0=w0, accum_func=accum_function)

# train neuron
for _ in range(0, epoch):
    for _data in data_scaled:
        neuron.train(_data, eta)
    neuron.update_w()

# get first component with PCA from neuron
neuron_pca = neuron.get_normalized_w()

# get first component with PCA from external library
pca = PCA()
pca.fit_transform(data_scaled)
library_pca = pca.components_[0]

print("Neuron PCA: " + str(neuron_pca))
print("Library PCA: " + str(library_pca))

# calculate ecm between oja and library in both directions
ecm_sum = 0
ecm_sum_inv = 0
for i in range(len(neuron_pca)):
    ecm_sum += (neuron_pca[i] - library_pca[i]) ** 2
    ecm_sum_inv += (neuron_pca[i] + library_pca[i]) ** 2

ecm = ecm_sum / len(neuron_pca) if ecm_sum < ecm_sum_inv else ecm_sum_inv / len(neuron_pca)
print(f'ECM between neuron and library: {ecm:.2E}')