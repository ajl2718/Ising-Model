# Generate configurations of the Ising model using the Metropolis algorithm
# use toroidal boundary conditions

import matplotlib.pyplot as plt
import numpy as np
import sys

from math import exp
from random import randint
from random import random
from copy import deepcopy

# get the width of the lattice, number of iterations and temperature from the command line arguments
args = sys.argv

# try reading off the inverse temperature, length of lattice and number of iterations, otherwise return their default values
# inverse temperature 1 / 2.269 default
try:
    L = int(args[1])
    num_iters = int(args[2])
    beta = float(args[3])

except IndexError:
    beta = 1 / 2.269
    L = 100
    num_iters = 10**6

# initial configuration (1 is spin up and -1 is spin down)
state0 = np.random.randint(0,2, (L,L))
state0 = state0 * 2 - 1

# calculate the energy of a cross (only need the four energies around each site for the algorithm to work)
def get_energy(lat_state, inv_temp, px, py):
    en = - lat_state[px % L][py % L] * (lat_state[(px+1) % L][py % L] + lat_state[(px-1)%L][py % L] +lat_state[px % L][(py + 1)%L] + lat_state[px % L][(py - 1)%L])
    return en

# Metropolis algorithm (give it the initial state, the number of steps to run the algorithm for and the inverse temperature
def metrop(state_init, num_steps, inv_temp):
    state_ising = deepcopy(state_init)

    # run num_steps of the algorithm
    for i in range(0, num_steps):
        # choose a site at random
        rx, ry = randint(0, L - 1), randint(0, L - 1)
        
        # random floating point number
        rf = random()

        # calculate the energy before and after a spin flip to determine whether or not to update
        en0 = get_energy(state_ising, inv_temp, rx, ry) 
        
        state_ising[rx][ry] *= -1
        en1 = get_energy(state_ising, inv_temp, rx, ry)

        # determine whether or not to keep the new state or return to the previous one
        prob = exp(-(en1 - en0) * inv_temp)

        if rf > prob:
            state_ising[rx][ry] *= -1
        
        sys.stdout.write("\rRunning metropolis algorithm for " + str(L) + " by " + str(L) + " lattice with " + str(num_steps) + " iterations: " + str( int( float(i) / float(num_steps) * 100) ) + "%" )
        sys.stdout.flush()
    sys.stdout.write("\nDone!\n")
    return state_ising    

data1 = metrop(state0, num_iters, beta)

# using numpy arrays
def get_pic(data_ising):
    return  0.5 * (data_ising + 1)

data_pic1 = get_pic(data1)

# plot the output
picdata = np.array(data_pic1)
plt.imshow(picdata, cmap='binary', interpolation='nearest')
plt.show()
