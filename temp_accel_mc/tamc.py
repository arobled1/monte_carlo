import sys
import numpy as np
import matplotlib.pyplot as plt
import opimc_tamc_funcs as funcs


def tamc_run(prim_z, vel_z, prims_r, num_mc_steps, num_pi_beads, num_intermediates, num_groups, real_mass, real_freq, prim_omegap, inverse_temp):
    lastu_bead = np.zeros(0)
    # Computing bead masses
    masses = funcs.get_masses(num_pi_beads, num_intermediates, num_groups, real_mass)
    # Computing bead frequencies
    bead_frequencies = funcs.get_bead_frequencies(num_pi_beads, num_intermediates, num_groups, prim_omegap)
    # Call MC to generate forces
    force_z, prims_r = funcs.run_pimc(prims_r, prim_z, num_mc_steps, num_pi_beads, num_intermediates, num_groups, real_mass, real_freq, prim_omegap, inverse_temp, masses, bead_frequencies)
    lastu_bead = np.append(lastu_bead, prims_r[0] - prims_r[num_pi_beads])
    #=============================================================================
    # MD code starts here !! (constants below should be moved to outside block)
    z_mass = 0.42
    dt = 0.0003
    gamma = 0.1
    for i in range(1,5):
        # Update velocity
        vel_z = funcs.upd_velocity(vel_z, force_z, dt, z_mass)
        # Update position
        prim_z = funcs.upd_position(prim_z, vel_z, dt)
        # Add random velocity
        vel_z = funcs.rand_kick(vel_z, gamma, 1/inverse_temp, z_mass, dt)
        # Update position with random velocity
        prim_z = funcs.upd_position(prim_z, vel_z, dt)
        # Call MC here to update forces
        force_z, prims_r = funcs.run_pimc(prims_r, prim_z, num_mc_steps, num_pi_beads, num_intermediates, num_groups, real_mass, real_freq, prim_omegap, inverse_temp, masses, bead_frequencies)
        lastu_bead = np.append(lastu_bead, prims_r[0] - prims_r[num_pi_beads])
        # Update velocity based on random force
        vel_z = funcs.upd_velocity(vel_z, force_z, dt, z_mass)
        # MD code ends here !!!
    return lastu_bead

#=====================================================================
# Block is for parameters
n_steps = 40                             # Number of MD steps for z
inv_temp = 15.8/3.0                          # kT
w = 3.0                                      # Set Frequency
m = 0.01                                     # Set Mass
num_beads = 32                               # Number of beads
jay = 4                                      # Number of beads in chain segment
big_n = 8                                    # Number of endpoint beads
freq_P = np.sqrt(num_beads) / inv_temp       # Set w_P
#======================================================================

initial_xpos = np.asarray([float(x.split()[0]) for x in open('x_pos.txt').readlines()])
initial_zpos = np.asarray([float(x.split()[0]) for x in open('z_pos.txt').readlines()])
initial_zvel = np.asarray([float(x.split()[0]) for x in open('z_vel.txt').readlines()])

last_bead = tamc_run(initial_zpos[0], initial_zvel[0], initial_xpos, n_steps, num_beads, jay, big_n, m, w, freq_P, inv_temp)
