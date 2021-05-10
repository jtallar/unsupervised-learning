import json
import random
import numpy as np
import perceptron as p
import pandas as pd
from sklearn.preprocessing import StandardScaler


# scale data obtained
def scaled_data(filepath: str):
    # read data_file csv
    full_data = pd.read_csv(filepath)
    # retrieve numeric values, scale data with media and normalize
    return StandardScaler().fit_transform(full_data.iloc[:, 1:].values)


# w0 for neuron initialization with dataset
def w0_data_init(data: np.ndarray) -> np.ndarray:
    return data[random.randint(0, len(data[0]) - 1)]


# w0 for neuron initialization without dataset
def w0_random_init(data: np.ndarray) -> np.ndarray:
    return np.random.uniform(-1, 1, len(data[0]))


# accumulation function for oja algorithm
def accum_function(data: np.ndarray, w: np.ndarray, e: float) -> np.ndarray:
    return e * (data - w)


# returns the initialized the kohonen grid
def init_kohonen_grid(random_init: bool, data: np.ndarray) -> [[p.SimplePerceptron]]:
    # w0 function
    if not random_init:
        w0 = w0_data_init
    else:
        w0 = w0_random_init

    grid: [[p.SimplePerceptron]] = []
    for i in range(0, k):
        row: [p.SimplePerceptron] = []
        for j in range(0, k):
            row.append(p.SimplePerceptron(w0=w0(data), accum_func=accum_function))
        grid.append(row)
    return grid


# gets the indexes of the most similar neuron given the data
def get_best_neuron_indexes(grid: [[p.SimplePerceptron]], data: np.ndarray) -> (int, int):
    # get the best neuron position
    best_val: float = np.inf
    best_neuron: tuple = (0, 0)
    for i, row in enumerate(grid):
        for j, n in enumerate(row):
            curr_val: float = np.sqrt(np.sum((data - n.get_normalized_w()) ** 2))
            if curr_val < best_val:
                best_val = curr_val
                best_neuron = (i, j)
    return best_neuron


# gets a list of tuples with the indexes of the neighbour neurons
def get_neighbour_neuron_indexes(grid_side: int, best: (int, int), radius: float) -> [(int, int)]:
    neighbours: [(int, int)] = []
    for i in range(0, grid_side):
        for j in range(0, grid_side):
            if (best[0] - i) ** 2 + (best[1] - j) ** 2 < radius ** 2:
                neighbours.append((i, j))
    return neighbours


# updates the w of all the neurons in the grid
def update_all_w(grid: [[p.SimplePerceptron]]):
    for row in grid:
        for neuron in row:
            neuron.update_w()


# read config file
with open("config.json") as file:
    config = json.load(file)

data_filepath: str = config["data_file"]
k: int = config["kohonen"]["k"]
c: float = config["kohonen"]["c"]
is_w0_random_init: bool = config["kohonen"]["w0_random_init"]
iterations: int = config["kohonen"]["kxk_iterations"] * k * k

# retrieve scaled data
data_scaled = scaled_data(data_filepath)

# initialize grid
kohonen_grid: [[p.SimplePerceptron]] = init_kohonen_grid(is_w0_random_init, data_scaled)

# training iteration
eta: float
r: float
for iteration in range(0, iterations):
    # calculate eta and r
    eta = 1 / (iteration + 1)
    r = 1 + k * np.e ** (-c * iteration)

    for _data in data_scaled:
        # get the best neuron position
        pos_best: (int, int) = get_best_neuron_indexes(kohonen_grid, _data)

        # get the best neuron's neighbours indexes
        pos_neighbours: [(int, int)] = get_neighbour_neuron_indexes(k, pos_best, r)

        # train the corresponding neurons
        for (x, y) in pos_neighbours:
            kohonen_grid[x][y].train(_data, eta)

    # update all the w values as the epoch finished
    update_all_w(kohonen_grid)

# finished training


# matrix of matches count
registry_count_grid: [[int]] = np.zeros((k, k), dtype=int)
for _data in data_scaled:
    # get the best neuron position
    pos_best: (int, int) = get_best_neuron_indexes(kohonen_grid, _data)

    # update index in the corresponding position
    registry_count_grid[pos_best[0]][pos_best[1]] += 1

# TODO matrix u calculation
