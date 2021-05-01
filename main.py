import json
import random

import numpy as np

import functions
import parser
import perceptron
import metrics

import matplotlib.pyplot as plt

with open("config.json") as file:
    config = json.load(file)

# static non changeable vars
cross_validation: bool = config["cross_validation"]
count_threshold: int = config["count_threshold"]
error_threshold: float = config["error_threshold"]
epoch_training: bool = config["epoch_training"]

# w randomization and reset configs
randomize_w: bool = config["randomize_w"]
randomize_w_ref: int = config["randomize_w_ref"]
reset_w: bool = config["reset_w"]
reset_w_iterations: int = config["reset_w_iterations"]

# adaptive eta configurations
general_adaptive_eta: bool = config["general_adaptive_eta"]
a: float = config["a"]
dec_k: int = config["delta_error_decrease_iterations"]
b: float = config["b"]
inc_k: int = config["delta_error_increase_iterations"]
adaptive_params: tuple = tuple([dec_k, inc_k, a, b])

# metrics params
normalize_out: bool = config["normalize_out"]
trust_min: float = config["trust_min"]
dec_round: int = config["float_rounding_dec"]

plot_bool: bool = config["plot"]

# read the files and get the training data, and expected out data
full_training_set, full_expected_out_set, number_class = parser.read_files(config["training_file"],
                                                                           config["expected_out_file"],
                                                                           config["system_threshold"])

# normalize expected out data if required
if config["system"] == "tanh" or config["system"] == "exp":
    full_expected_out_set = parser.normalize_data(full_expected_out_set)

# activation function and its derived, if derived is not used then returns always 1
act_funcs = functions.get_activation_functions(config["system"], config["beta"],
                                               config["retro_error_enhance"], number_class)

# randomize input distribution, used for cross validation
if cross_validation:
    full_training_set, full_expected_out_set = parser.randomize_data(full_training_set, full_expected_out_set)

# for cross validations
test_ratio: int = 100 - config["training_ratio"]
if cross_validation and test_ratio != 0 and len(full_training_set) % int(100 / test_ratio) != 0:
    print(f"Training ration must be divisor of training set length ({len(full_training_set)})")
    exit(1)
cross_validation_count: int = 1 if not cross_validation or test_ratio == 0 else int(100 / test_ratio)
j: int = 0

# for metrics
best_metrics: dict = {}
recent_metrics: dict = {}
delta_eq: float = config["delta_metrics"]

cross_acc_train = []
cross_acc_test = []
cross_err_train = []
cross_err_test = []
cross_x = []
plt.rcParams.update({'font.size': 20})

# do only one if it is not cross validation
while j < cross_validation_count:
    eta: float = config["eta"]

    # keep only a portion of the full data set for training
    training_set, expected_out_set, test_training_set, test_expected_out_set \
        = parser.extract_subset(full_training_set, full_expected_out_set, test_ratio, j)

    # initialize the perceptron completely
    c_perceptron = perceptron.ComplexPerceptron(*act_funcs, config["layout"],
                                                len(training_set[0]), len(expected_out_set[0]),
                                                config["momentum"], config["momentum_alpha"])

    # randomize the perceptron initial weights if needed
    if randomize_w:
        c_perceptron.randomize_w(randomize_w_ref)

    # start the training iterations

    # counters and length
    p: int = len(training_set)
    i: int = 0
    n: int = 0

    # error handling
    error: float = np.inf
    error_min: float = np.inf

    # adaptive eta vars
    delta_error: float = np.inf
    delta_error_dec: bool = True
    k: int = 0

    it = []
    err = []
    acc_test_total = []
    acc_train_total = []

    # finish only when error is 0 or reached max iterations/epochs
    while error > error_threshold and i < count_threshold:

        # in case there is random w, randomize it again
        if reset_w and (n > p * reset_w_iterations):
            c_perceptron.randomize_w(randomize_w_ref)
            n = 0

        # for epoch training is the full training dataset, for iterative its only one random
        train_indexes = [random.randint(0, p - 1)] if not epoch_training else range(p)
        for index in train_indexes:
            c_perceptron.train(training_set[index], expected_out_set[index], eta, epoch_training)

        # for epoch training update the w once the epoch is finished
        if epoch_training:
            c_perceptron.update_w()

        # calculate error on the output laye
        new_error = c_perceptron.error(training_set, expected_out_set, config["retro_error_enhance"])

        # adaptive eta
        if general_adaptive_eta:
            delta_error, delta_error_dec, k, eta = functions.adaptive_eta(error-new_error, delta_error, delta_error_dec,
                                                                          k, eta, *adaptive_params)
        error = new_error

        # update or not min error
        if error < error_min:
            error_min = error
        
        it.append(i)

        i += 1
        n += 1

        if plot_bool:
            # get metrics and error values for plots
            _, best_metrics = metrics.metrics({}, c_perceptron, training_set, expected_out_set,
                                                        test_training_set, test_expected_out_set, delta_eq,
                                                        normalize_out, trust_min)
            acc_train_total.append(best_metrics["acc_train"]*100)
            acc_test_total.append(best_metrics["acc_test"]*100)
            err.append(best_metrics["err_train"])

    print(error)
    # finished, perceptron trained

    # get metrics and error values, save the best perceptron
    best_metrics, recent_metrics = metrics.metrics(best_metrics, c_perceptron, training_set, expected_out_set,
                                                   test_training_set, test_expected_out_set, delta_eq,
                                                   normalize_out, trust_min)

    # print only if printing enabled and has more than one cross validation iteration
    if config["print_each_cross_validation"] and test_ratio != 0 and cross_validation:
        cross_x.append(j)
        print(f"Cross validation try {j+1}/{cross_validation_count}, "
              f"training set accuracy: {np.around(recent_metrics['acc_train'] * 100, 2)}%, "
              f"test set accuracy: {np.around(recent_metrics['acc_test'] * 100, 2)}%")
        cross_acc_train.append(recent_metrics['acc_train'] * 100)
        cross_acc_test.append(recent_metrics['acc_test'] * 100)
        cross_err_train.append(recent_metrics['err_train'])
        cross_err_test.append(recent_metrics['err_test'])
    
    if plot_bool and not cross_validation:
        metrics.plot_values(it, "Iterations", err,"Error")
        metrics.plot_multiple_values([it], "Iterations", [acc_train_total], "Accuracy", ["Training"], min_val=0, max_val=100)
        metrics.plot_multiple_values([range(len(full_expected_out_set)), range(len(full_expected_out_set))], "Indep Var", [full_expected_out_set, c_perceptron.activation(full_training_set)], "Value", ["Real", "Predicted"], marker='o')

    j += 1

if plot_bool and cross_validation:
    metrics.plot_multiple_values([cross_x, cross_x], "Iterations", [cross_err_train, cross_err_test], "Error", ["Training", "Test"], colors=["red", "yellow"])
    metrics.plot_multiple_values([cross_x, cross_x], "Iterations", [cross_acc_train, cross_acc_test], "Accuracy", ["Training", "Test"], min_val=0, max_val=100, colors=["blue", 'orange'])

plt.show(block=True)
print(f"\nBest perceptron training set accuracy: {np.around(best_metrics['acc_train'] * 100, 2)}%, "
      f"and error: {np.around(best_metrics['err_train'], dec_round)}")
if test_ratio != 0:
    print(f"Best perceptron test set accuracy: {np.around(best_metrics['acc_test'] * 100, 2)}%, "
          f"and error: {np.around(best_metrics['err_test'], dec_round)}")

# input("\nPress enter to show training set results evaluated: ")
# for data, out, pred in \
#         zip(np.around(best_metrics['training_set'], dec_round),
#             np.around(best_metrics['expected_out_set'], dec_round),
#             np.around(best_metrics['train_predicted'], dec_round)):
#     print(f"In: {data.astype(number_class)}, out: {out.astype(number_class)}, perceptron: {pred}")
#
# if test_ratio != 0:
#     input("\nPress enter to show testing set results evaluated: ")
#     for data, out, pred in \
#             zip(np.around(best_metrics['test_training_set'], dec_round),
#                 np.around(best_metrics['test_expected_out_set'], dec_round),
#                 np.around(best_metrics['test_predicted'], dec_round)):
#         print(f"In: {data.astype(number_class)}, out: {out.astype(number_class)}, perceptron: {pred}")

# input("\nPress enter to show perceptron data: ")
# print(best_metrics['perceptron'])

input("\nPress enter to print config: ")
print(config)

# finished
