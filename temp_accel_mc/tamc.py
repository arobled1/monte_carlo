import sys
import numpy as np
import matplotlib.pyplot as plt
import opimc_tamc_funcs as funcs


def tamc_run(prims_z, vels_z, prims_r, num_mc_steps, num_pi_beads, j, capital_n, real_mass, real_freq, prim_omegap, inverse_temp):
    # Computing bead masses
    masses = get_masses(num_pi_beads, j, capital_n, real_mass)

    # Computing bead frequencies
    bead_frequencies = get_bead_frequencies(num_pi_beads, j, capital_n, inverse_temp)

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
