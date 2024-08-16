"""Plot a trajectory of the following Feller branching diffusion model:
    every individual gives birth at rate r to a unique individual of mean size
    x_0. During its life, every individual trait varies according to a SDE
    
        dX^i_t = (1 + a * (X^i_t) / (1 + R_t))dt + 2 \sqrt(X^i_t)dB^i_t
        
    where $i$ is the label of the individual and $R_t$ is the sum of the traits
    in the population.

Usage:
======
    Choose the model parameters and the module plot_function will plot an exact 
    trajectory of the population.

    a: the positive growth rate (prompt input)
    r: the positive branching rate (prompt input)
    x_0: the positive mean size at birth (prompt input)
    n_init: initial population size, positive integer (prompt input)
    T: Positive simulation time, T < 100, (prompt input)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import feller_diffusion_methods

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams["figure.dpi"] = 100
plt.rcParams['svg.fonttype']='none'
matplotlib.rc('legend', fontsize= 9)
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.1

ft = 0 # Default latex font, change to 1 for arial font 
if ft == 1:
    font = {'family' : 'Arial',
            'size'   : 11}
    matplotlib.rc('font', **font)
else:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })
    
print(" Choose parameters values: ")
default = input("Default? (y/n) \n")
if default == 'y':
    a = 6
    r = 0.2
    x_0 = 25
    n_init = 2
    T = 30
else:
    a = abs(float(input("\n positive growth rate a = ")))
    r = abs(float(input("\n positive branching rate r = ")))
    x_0 = abs(float(input("\n positive initial mean trait x_0 = ")))
    n_init = int(input("\n positive integer initial population size N = "))
    T =abs(float( input("\n positive simulation time T = ")))
    T = np.min((T, 100))
feller_diffusion_methods.Trajectory(a, r, x_0, n_init, T)
