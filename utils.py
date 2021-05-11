import csv
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def invalid_param(param_name):
    print(f'Error in config. Invalid or missing {param_name}!')
    sys.exit(1)

def read_config_param(config, param_name, converter_fun, valid_fun):
    if param_name in config:
        param = converter_fun(config[param_name])
        if valid_fun(param):
            return param
    invalid_param(param_name)

RESET = '\033[0m'
def get_color_escape(color_hex, background=False):
    rgb = [int(color_hex[i:i+2], 16) for i in range(1, len(color_hex), 2)]
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, rgb[0], rgb[1], rgb[2])

def print_with_color(string, color_hex):
    print(get_color_escape(color_hex) + string + RESET)

# Formatter taken from 
# https://stackoverflow.com/questions/25750170/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        s = self.fmt % x
        dec_point = '.'
        pos_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(dec_point)
        sign = tup[1][0].replace(pos_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if not exponent: exponent = 0
        exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

def init_plotter():
    plt.rcParams.update({'font.size': 20})

def plot_boxplot(values, labels, y_label, save_name=None):
    fig, ax = plt.subplots(figsize=(12, 10))  # Create a figure containing a single axes.
    ax.boxplot(values)
    ax.set_ylabel(y_label)
    plt.xticks(range(1, values.shape[1] + 1), labels)
    
    plt.grid()
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show(block=False)

def plot_values(x_values, x_label, y_values, y_label, precision=2, sci_x=False, sci_y=True, min_val=None, max_val=None, save_name=None):
    fig, ax = plt.subplots(figsize=(12, 10))  # Create a figure containing a single axes.
    ax.plot(x_values, y_values, marker='o', markersize=3)  # Plot some data on the axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if min_val is not None and max_val is not None:
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])

    if sci_x:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.xaxis.set_major_formatter(MathTextSciFormatter(f'%1.{precision}e'))
    if sci_y:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.yaxis.set_major_formatter(MathTextSciFormatter(f'%1.{precision}e'))

    plt.grid()
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show(block=False)

def plot_horiz_bar(labels, values, x_label, save_name=None):
    fig, ax = plt.subplots(figsize=(12, 10))  # Create a figure containing a single axes.
    ax.barh(labels, values)
    ax.set_xlabel(x_label)
    
    plt.grid()
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show(block=False)

def plot_two_components(pca_1, pca_2, comp_1, comp_2, var_labels, val_labels, scale=True, save_name=None):
    fig, ax = plt.subplots(figsize=(12, 10))  # Create a figure containing a single axes.

    # Scale values if needed
    scale_x, scale_y = 1.0, 1.0
    scale_arrows = 1.0
    if scale:
        scale_x = 1.0 / (pca_1.max() - pca_1.min())
        scale_y = 1.0 / (pca_2.max() - pca_2.min())
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    else:
        scale_arrows = 3

    # Plot values
    ax.scatter(pca_1 * scale_x, pca_2 * scale_y)

    # Annotate values with value labels
    for i in range(len(pca_1)):
        ax.annotate(val_labels[i], (pca_1[i] * scale_x, pca_2[i] * scale_y), fontsize=10)
    
    # Plot variable arrows with variable labels
    for i in range(len(comp_1)):
        ax.arrow(0, 0, comp_1[i] * scale_arrows, comp_2[i] * scale_arrows, color='purple', alpha=0.5)
        ax.text(comp_1[i] * scale_arrows * 1.12, comp_2[i] * scale_arrows * 1.12, var_labels[i], color='orange', ha='center', va='center', fontsize=15)
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    
    plt.grid()
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show(block=False)

def plot_matrix(matrix, cmap='inferno', not_exp=False):
    fig, ax = plt.subplots(figsize=(12, 10))  # Create a figure containing a single axes.

    caxes = ax.matshow(matrix, interpolation='none', cmap=cmap)
    plt.colorbar(caxes)

    notation = '{:0.1E}' if not_exp else '{:0.1f}'
    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, notation.format(z), ha='center', va='center', color='green')

    plt.tight_layout()
    plt.show(block=False)

def plot_mult_histogram_density(values_1, values_2, n_bins, x_label, y_label, precision=2, sci_x=False, sci_y=True):
    fig, ax = plt.subplots(figsize=(12, 10))  # Create a figure containing a single axes.
    weights = np.full(len(values_1), 1.0 / len(values_1))
    ax.hist(values_1, bins=n_bins, alpha=0.7, weights=weights, label='Initial')  # Plot some data on the axes
    weights = np.full(len(values_2), 1.0 / len(values_2))
    ax.hist(values_2, bins=n_bins, alpha=0.7, weights=weights, label='Last third')  # Plot some data on the axes
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if sci_x:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.xaxis.set_major_formatter(MathTextSciFormatter(f'%1.{precision}e'))
    if sci_y:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.yaxis.set_major_formatter(MathTextSciFormatter(f'%1.{precision}e'))

    fig.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.show(block=False)

def plot_histogram_density(values, n_bins, x_label, y_label, precision=2, sci_x=False, sci_y=True, log=False):
    fig, ax = plt.subplots(figsize=(12, 10))  # Create a figure containing a single axes.
    weights = np.full(len(values), 1.0 / len(values))
    _n, _bins, _patches = ax.hist(values, bins=n_bins, weights=weights)  # Plot some data on the axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if sci_x:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.xaxis.set_major_formatter(MathTextSciFormatter(f'%1.{precision}e'))
    if sci_y:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.yaxis.set_major_formatter(MathTextSciFormatter(f'%1.{precision}e'))

    plt.grid()
    plt.tight_layout()

    if log:
        step = n_bins[1]
        bin_center = [x + step for x in n_bins]
        fig, ax = plt.subplots(figsize=(12, 10))  # Create a figure containing a single axes.
        ax.plot(bin_center[:len(bin_center) - 1], _n)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_yscale('symlog', linthresh=1e-3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.xaxis.set_major_formatter(MathTextSciFormatter(f'%1.{precision}e'))
        plt.grid()
        plt.tight_layout()

    plt.show(block=False)

def plot_values_with_adjust(x_values, x_label, y_values, y_label, precision=2, sci=True, min_val=None, max_val=None, plot=True, save_name=None):
    # adj_coef = np.polyfit(x_values, y_values, 1)
    # poly1d_fn = np.poly1d(adj_coef)

    c, err = calculate_regression(x_values, y_values, plot)
    print(c, err)

    if not plot: return c

    fig, ax = plt.subplots(figsize=(12, 10))  # Create a figure containing a single axes.
    ax.plot(x_values, y_values, 'yo', x_values, [f_adj(x, c) for x in x_values], '-k')  # Plot some data on the axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if min_val is not None and max_val is not None:
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])

    if sci:
        ax.ticklabel_format(scilimits=(0,0))
        ax.xaxis.set_major_formatter(MathTextSciFormatter(f'%1.{precision}e'))
        ax.yaxis.set_major_formatter(MathTextSciFormatter(f'%1.{precision}e'))

    plt.grid()
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show(block=False)

    return c

def plot_multiple_values(x_values_superlist, x_label, y_values_superlist, y_label, legend_list, precision=2, sci=True, min_val=None, max_val=None, log=False, legend_loc='upper right', save_name=None):
    fig, ax = plt.subplots(figsize=(12, 10))  # Create a figure containing a single axes.

    colors = []
    for i in range(len(x_values_superlist)):
        p = ax.plot(x_values_superlist[i], y_values_superlist[i], label=legend_list[i])  # Plot some data on the axes
        colors.append(p[-1].get_color())

    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if min_val is not None and max_val is not None:
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])

    if sci:
        if not log: ax.ticklabel_format(scilimits=(0,0))
        ax.xaxis.set_major_formatter(MathTextSciFormatter(f'%1.{precision}e'))
        ax.yaxis.set_major_formatter(MathTextSciFormatter(f'%1.{precision}e'))

    plt.tight_layout()
    plt.grid()
    plt.legend(loc=legend_loc)
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show(block=False)

    return colors

def plot_error_bars_summary(x_values, x_label, sum_values, attribute, y_label, x_prec=2, sci_x=False, sci_y=True, y_min=None, y_max=None, log=False, save_name=None):
    values = []
    values_err = []
    min_dec = getattr(sum_values[0], attribute).dec_count
    for x in sum_values:
        attr = getattr(x, attribute)
        values.append(attr.media)
        values_err.append(attr.std)
        if attr.dec_count < min_dec:
            min_dec = attr.dec_count
    # min_dec += 1
    if sci_y: min_dec = 1
    print(y_label)
    print(values)
    print(values_err)
    print(min_dec)
    plot_error_bars(x_values, x_label, values, y_label, values_err, x_prec, min_dec, sci_x, sci_y, y_min, y_max, log, save_name)

def plot_error_bars(x_values, x_label, y_values, y_label, y_error, x_prec=2, y_prec=2, sci_x=False, sci_y=True, y_min=None, y_max=None, log=False, save_name=None):
    fig, ax = plt.subplots(figsize=(12, 10))  # Create a figure containing a single axes.
    (_, caps, _) = plt.errorbar(x_values, y_values, yerr=y_error, markersize=6, capsize=20, elinewidth=0.75, linestyle='-',  marker='o')  # Plot some data on the axes
    for cap in caps:
        cap.set_markeredgewidth(1)

    ax.set_ylim([y_min, y_max])
    if log:
        ax.set_yscale('symlog', linthresh=1e-3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if sci_x:
        if not log: ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.xaxis.set_major_formatter(MathTextSciFormatter(f'%1.{x_prec}e'))
    if sci_y:
        if not log: ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.yaxis.set_major_formatter(MathTextSciFormatter(f'%1.{y_prec}e'))

    plt.grid()
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show(block=False)

def hold_execution():
    plt.show(block=True)