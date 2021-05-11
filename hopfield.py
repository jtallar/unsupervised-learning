import json
import random
import os
import numpy as np
import perceptron as p
import utils


# Build pattern array from txt file with '*', ' '
def get_pattern_array(filename: str) -> np.ndarray:
    pattern = []
    with open(filename) as file:
        for line in file:
            for car in line.strip('\n'):
                pattern.append(1 if car == '*' else -1)
    return np.array(pattern)

# Build pattern matrix from directory, len(pattern)
def pattern_matrix(dirpath: str, N: int) -> np.ndarray:
    file_list = os.listdir(pattern_dirpath)
    patterns = np.zeros((N, len(file_list)), dtype=int)
    letter_list = []
    for i, pattern_filename in enumerate(file_list):
        letter_list.append(pattern_filename)
        pattern_filepath = pattern_dirpath + '/' + pattern_filename
        patterns[:, i] = get_pattern_array(pattern_filepath)
    return letter_list, patterns

# Print side x side pattern from side x side lengthed array
def print_pattern(pattern: np.ndarray, side: int):
    for i in range(side * side):
        car = '*' if pattern[i] > 0 else ' '
        print(car, end='')
        if i != 0 and (i + 1) % side == 0:
            print('\n', end='')

# Generate pattern mutation from pattern using pm probability
def get_mutated_pattern(pattern: np.ndarray, pm: float) -> np.ndarray:
    mut_pattern = np.copy(pattern)
    for i in range(len(mut_pattern)):
        if random.random() < pm:
            mut_pattern[i] = mut_pattern[i] * -1
    return mut_pattern


# Pattern length is fixed
SIDE = 5
N = SIDE * SIDE
RED = "#FF0000"

# read config file
with open("config.json") as file:
    config = json.load(file)

pattern_dirpath: str = config["hopfield"]["pattern_dir"]
pm: float = config["hopfield"]["mutation_prob"]
max_iterations: int = config["hopfield"]["max_iterations"]

# Build pattern matrix, with [e1 e2 e3 ...], len(ei) = N
letter_list, patterns = pattern_matrix(pattern_dirpath, N)

# Get query pattern from available patterns
query_num = random.randint(0, patterns.shape[1] - 1)
query_pattern = get_mutated_pattern(patterns[:, query_num], pm)

# Initialize Hopfield perceptron
algo: p.HopfieldPerceptron = p.HopfieldPerceptron(patterns, query_pattern)

# Print initial query
print_pattern(query_pattern, SIDE)
print('------------------')

# Iterate over hopfield
s: np.ndarray
count: int = 0
while not algo.is_over() and count < max_iterations:
    s = algo.iterate()
    print_pattern(s, SIDE)
    print('------------------')
    count += 1

if count >= max_iterations:
    print(f'Se ha alcanzado el {utils.string_with_color("límite de iteraciones", RED)} (probablemente por loop). Saliendo...')
    exit()

for i in range(patterns.shape[1]):
    if np.array_equal(s, patterns[:, i]):
        print(f'El estado final {utils.string_with_color("coincide con " + letter_list[i], RED)}. Originalmente era {letter_list[query_num]}.')
        exit()

print(f'El estado final es {utils.string_with_color("espúreo", RED)}. Debería coincidir con {letter_list[query_num]}.')
