import json
import random
import math
import numpy as np
import perceptron as p
import utils
import pandas as pd
from sklearn.preprocessing import StandardScaler


# scale data obtained
def scaled_data(filepath: str, normalize: bool = False):
    # read data_file csv
    full_data = pd.read_csv(filepath)
    # retrieve numeric values, scale data with media and stdev
    std_data = StandardScaler().fit_transform(full_data.iloc[:, 1:].values)
    # normalize if needed
    if normalize:
        std_data = np.divide(std_data, np.linalg.norm(std_data, 2))

    return full_data.iloc[:, 0].values, std_data


# w0 for neuron initialization with dataset
def w0_data_init(data: np.ndarray) -> np.ndarray:
    return data[random.randint(0, len(data) - 1)]


# w0 for neuron initialization without dataset
def w0_random_init(data: np.ndarray) -> np.ndarray:
    return np.random.uniform(0, 1, len(data[0]))


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

def eucledian_distance(vec_1: np.ndarray, vec_2: np.ndarray) -> float:
    return np.sqrt(np.sum((vec_1 - vec_2) ** 2))

# TODO: Do same thing with correlation distance
# gets the indexes of the most similar neuron given the data using eucledian distance
def get_best_neuron_indexes(grid: [[p.SimplePerceptron]], data: np.ndarray) -> (int, int):
    # get the best neuron position
    best_val: float = np.inf
    best_neuron: tuple = (0, 0)
    for i, row in enumerate(grid):
        for j, n in enumerate(row):
            curr_val: float = eucledian_distance(data, n.get_w())
            if curr_val < best_val:
                best_val = curr_val
                best_neuron = (i, j)
    return best_neuron


# gets a list of tuples with the indexes of the neighbour neurons and distance to best
def get_neighbour_neuron_indexes(grid_side: int, best: (int, int), radius: float) -> [(int, int, float)]:
    neighbours: [(int, int, float)] = []
    for i in range(0, grid_side):
        for j in range(0, grid_side):
            d = eucledian_distance(np.array(best), np.array((i, j)))
            if d <= radius:
                neighbours.append((i, j, d))
    return neighbours


# updates the w of all the neurons in the grid
def update_all_w(grid: [[p.SimplePerceptron]]):
    for row in grid:
        for neuron in row:
            neuron.update_w()


def get_corrected_eta(eta: float, distance: float, radius: float, adjust: bool = False):
    if not adjust:
        return eta
    return math.exp(-distance / radius) * eta


# read config file
with open("config.json") as file:
    config = json.load(file)

data_filepath: str = config["data_file"]
k: int = config["kohonen"]["k"]
c: float = config["kohonen"]["c"]
eta_init: float = config["kohonen"]["eta_init"]
is_w0_random_init: bool = config["kohonen"]["w0_random_init"]
normalize_data: bool = config["kohonen"]["normalize_data"]
neigh_dist_adj: bool = config["kohonen"]["neighbour_dist_adj"]
iterations: int = config["kohonen"]["kxk_iterations"] * k * k
plot_boolean: bool = config["plot"]

# retrieve scaled data
countries, data_scaled = scaled_data(data_filepath, normalize_data)

# initialize grid
kohonen_grid: [[p.SimplePerceptron]] = init_kohonen_grid(is_w0_random_init, data_scaled)

# training iteration
eta: float
r: float
for iteration in range(0, iterations):
    # calculate eta and r
    # eta = 1 / (iteration + 1)
    eta = (iterations - iteration) * eta_init / iterations
    # r = 1 + k * math.exp(-c * iteration)
    r = 1 + math.floor((iterations - iteration) * k / iterations) # initial radius is 1+k

    for _data in data_scaled:
        # get the best neuron position
        pos_best: (int, int) = get_best_neuron_indexes(kohonen_grid, _data)

        # get the best neuron's neighbours indexes
        pos_neighbours: [(int, int, float)] = get_neighbour_neuron_indexes(k, pos_best, r)

        # train the corresponding neurons
        for (x, y, d) in pos_neighbours:
            kohonen_grid[x][y].train(_data, get_corrected_eta(eta, d, r, neigh_dist_adj))

        # update all the w values as the iteration finished
        update_all_w(kohonen_grid)

# finished training


# matrix of matches count
country_list_grid: [[str]] = [[[] for x in range(k)] for x in range(k)]
registry_count_grid: [[int]] = np.zeros((k, k), dtype=int)
for country, _data in zip(countries, data_scaled):
    # get the best neuron position
    pos_best: (int, int) = get_best_neuron_indexes(kohonen_grid, _data)

    # update index in the corresponding position
    registry_count_grid[pos_best[0]][pos_best[1]] += 1

    # add country name to list
    country_list_grid[pos_best[0]][pos_best[1]].append(country)

print('Match matrix countries: ')
print('\n'.join('{}: {}'.format(*k) for k in enumerate(country_list_grid)))


# TODO: Check if U matrix uses radius=1 (or final) for node neighbours
# r = 1
# U matrix 
u_matrix_grid: [[int]] = np.zeros((k, k), dtype=float)
for i, row in enumerate(kohonen_grid):
    for j, n in enumerate(row):
        # get neighbours index list
        neighbours: [(int, int, float)] = get_neighbour_neuron_indexes(k, (i, j), r)

        # iterate over neighbours to accumulate eucledian distance
        sum_dist: float = 0
        for (x, y, d) in neighbours:
            sum_dist += eucledian_distance(n.get_w(), kohonen_grid[x][y].get_w())
        
        # save average eucledian distance with neighbours
        u_matrix_grid[i][j] = sum_dist / len(neighbours)
       
# if plot_boolean is true, generate plots
if plot_boolean:
    # Init plotter
    utils.init_plotter()

    # Plot registry count matrix
    utils.plot_matrix(registry_count_grid)

    # Plot U Matrix
    utils.plot_matrix(u_matrix_grid, 'gray', not_exp=True, vmin=0, vmax=1)

    # Hold execution to show plots
    utils.hold_execution()

