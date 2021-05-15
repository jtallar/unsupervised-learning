import utils

rep_count = 20

pms = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50, 0.55, 0.6]
match_ok = [x / rep_count for x in [20, 20, 20, 18, 14, 13, 8, 8, 8, 1, 1, 0]]
match_wrong = [x / rep_count for x in [0, 0, 0, 1, 1, 2, 3, 4, 5, 11, 7, 4]]
loop = [x / rep_count for x in [0, 0, 0, 0, 0, 0, 2, 3, 1, 2, 0, 4]]
spurious = [x / rep_count for x in [0, 0, 0, 1, 5, 5, 7, 5, 6, 6, 12, 12]]

# Initialize plotting
utils.init_plotter()

utils.plot_multiple_values(
    [pms, pms, pms, pms], 'pm', 
    [match_ok, match_wrong, loop, spurious], 'probabilidad', 
    ['correcto', 'incorrecto', 'loop', 'espúreo'], legend_loc='upper right', precision=0,
    sci_x=False, sci_y=False, min_val_y=-0.1, max_val_y=1.1)

# Hold execution
utils.hold_execution()