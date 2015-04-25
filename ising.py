# Generate configurations of the Ising model using the Metropolis algorithm
# use toroidal boundary conditions

import matplotlib.pyplot as plt
import numpy as np
import sys

from math import exp
from random import randint
from random import random
from copy import deepcopy

# inverse temperature
beta = 1 / 2.269

# Length and width of lattice
L = 100

# initial configuration (1 is spin up and -1 is spin down)
state0 = [ [randint(0,1) * 2 - 1 for i in range(0, L)] for j in range(0, L)]

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

    sys.stdout.write("Done\n")
    return state_ising    

data1 = metrop(state0, 10**6, beta)

# convert the Ising data to 0's and 1's instead of -1's and +1's
def get_pic(data_ising):
    N = len(data_ising)
    for i in range(0, N):
        for j in range(0, N):
            data_ising[i][j] = 0.5 * (data_ising[i][j] + 1)
    return data1

data_pic1 = get_pic(data1)

# plot the output
picdata = np.array(data_pic1)
plt.imshow(picdata, cmap='binary', interpolation='nearest')
plt.show()
