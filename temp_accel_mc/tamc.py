import sys
import numpy as np
import matplotlib.pyplot as plt
import opimc_tamc_funcs as funcs


def tamc_run(prims_z, vels_z, primitives_r, num_steps, pbeads, j, capital_n, mass, omega, omegaP, beta):
    # Computing bead masses
    masses = np.zeros(pbeads+1)
    # Defining the mass for the first bead
    masses[0] = 0
    # Defining masses for endpoint beads
    for s in range(1,capital_n+1):
        masses[s*j] = (s+1) * mass / s
    # Defining masses for intermediate beads
        for k in range(1,j):
            masses[(s-1)*j + k] = (k+1) * mass / k
    # Defining the mass for the last bead
    masses[pbeads] = mass/capital_n

    # Computing bead frequencies
    bead_frequencies = np.zeros(pbeads+1)
    for s in range(capital_n):
        bead_frequencies[s*j] = omegaP / np.sqrt(j)
        for k in range(1,j):
            bead_frequencies[s*j+k] = omegaP
    bead_frequencies[pbeads] = omegaP / np.sqrt(j)

    # Call MC here to generate forces

    #=============================================================================
    # MD code starts here !!
    for i in range(1,n_steps):
        # Update velocity
        v = upd_velocity(v, f_z, dt, m_prime_k)
        # Update position
        z = upd_position(z, v, dt)
        # Add random velocity
        v = rand_kick(v, gamma, kbt, m_prime_k, dt)
        # Update position with random velocity
        z = upd_position(u, v, dt)

        # Call MC here to update forces

        # Update velocity based on random force
        v = upd_velocity(v, f_z, dt, m_prime_k)
        # MD code ends here !!!
    return lastu_bead
#=====================================================================
# Block is for parameters
n_steps = 500000                             # Number of MD steps for z
inv_temp = 15.8/3.0                          # kT
w = 3.0                                      # Set Frequency
m = 0.01                                     # Set Mass
num_beads = 12                               # Number of beads
jay = 3                                      # Number of beads in chain segment
big_n = 4                                    # Number of endpoint beads
freq_P = np.sqrt(num_beads) / inv_temp       # Set w_P
#======================================================================

initial_xpos = np.asarray([float(x.split()[0]) for x in open('x_pos.txt').readlines()])
initial_zpos = np.asarray([float(x.split()[0]) for x in open('z_pos.txt').readlines()])
initial_zvel = np.asarray([float(x.split()[0]) for x in open('z_vel.txt').readlines()])

last_bead = opimc_run(initial_zpos, initial_zvel, initial_xpos, n_steps, num_beads, jay, big_n, m, w, freq_P, inv_temp)
