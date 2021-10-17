import numpy as np
import numba
from numba import jit

@jit(nopython=True)
def get_masses(num_beads, num_seg_beads, num_segments, mass):
    # Computing bead masses
    bead_masses = np.zeros(num_beads+1)
    # Defining the mass for the first bead
    bead_masses[0] = 0
    # Defining masses for endpoint beads
    for s in range(1,num_segments+1):
        bead_masses[s*num_seg_beads] = (s+1) * mass / s
    # Defining masses for intermediate beads
        for k in range(1,num_seg_beads):
            bead_masses[(s-1)*num_seg_beads + k] = (k+1) * mass / k
    # Defining the mass for the last bead
    bead_masses[num_beads] = mass/num_segments
    return bead_masses

@jit(nopython=True)
def get_bead_frequencies(num_beads, num_seg_beads, num_segments, omegaP):
    # Computing bead frequencies
    bead_freqs = np.zeros(num_beads+1)
    for s in range(num_segments):
        bead_freqs[s*num_seg_beads] = omegaP / np.sqrt(num_seg_beads)
        for k in range(1, num_seg_beads):
            bead_freqs[s*num_seg_beads + k] = omegaP
    bead_freqs[num_beads] = omegaP / np.sqrt(num_seg_beads)
    return bead_freqs

@jit(nopython=True)
def stage_coords(xi, num_beads, num_seg_beads, num_segments):
    up_u = np.zeros(len(xi))
    for p in range(len(xi)):
        up_u[p] = xi[p]
    # Staging the segment beads
    for s in range(num_segments):
        for k in range(1,num_seg_beads):
            up_u[(s*num_seg_beads + k)] = xi[(s*num_seg_beads + k)] - ( (k)*xi[(s*num_seg_beads + k + 1)] + xi[(s*num_seg_beads)] ) / (k+1)
    # Staging the endpoint beads
    up_u[0] = 0.5 * (xi[0] + xi[num_beads])
    # For the in between beads
    for s in range(1,num_segments):
        up_u[s*num_seg_beads] = xi[s*num_seg_beads] - (1/(s+1)) * (s * xi[((s+1)*num_seg_beads)] + xi[0])
    up_u[num_beads] = xi[0] - xi[num_beads]
    return up_u

@jit(nopython=True)
# Transform staged coordinates to primitive coordinates
def inverse_stage_coords(ui, num_beads, num_seg_beads, num_segments):
    up_x = np.zeros(len(ui))
    for p in range(len(ui)):
        up_x[p] = ui[p]
    # Transforming endpoint beads back to primitive coordinates
    # For the first bead
    up_x[0] = ui[0] + 0.5 * ui[num_beads]
    # For the in between beads
    for s in range(1,num_segments):
        sum1 = 0
        for ell in range(s,num_segments):
            sum1 += ((s)/(ell)) * ui[ell*num_seg_beads]
        up_x[s*num_seg_beads] = ui[0] + (( (num_segments/2) - s) / num_segments) * ui[num_beads] + sum1
    # For the last bead
    up_x[num_beads] = ui[0] - 0.5 * ui[num_beads]
    # Transforming segment beads back to primitive coordinates
    for s in range(num_segments):
        for k in range(num_seg_beads-1, 0, -1):
            up_x[(s*num_seg_beads + k)] = ui[(s*num_seg_beads + k)] + (k/(k+1))*up_x[(s*num_seg_beads + k + 1) % (num_beads+1)] + (1/(k+1))*up_x[(s*num_seg_beads)]
    return up_x

def bin_centers(bin_edges):
    return (bin_edges[1:]+bin_edges[:-1])/2.

def get_harmonic_density(pos, inv_temp, mass, omega):
    normalization = np.sqrt((mass*omega) / (4*np.pi*np.tanh(inv_temp*omega/2)))
    exp_constant = np.pi * normalization**2
    return normalization * np.exp(-exp_constant * pos**2 )

@jit(nopython=True)
def get_harmonic_potential(positions, mass, frequency, num_beads, left_wall, option):
    pot = 0
    # If sampling intermediate beads
    if option == "ints":
        for y in range(left_wall+1,left_wall + num_beads):
            pot += 0.5 * mass * (frequency**2) * positions[y]**2
    # If sampling endpoint beads
    elif option == "ends":
        pot += 0.5 * mass * (frequency**2) * positions[left_wall]**2
    # If sampling the whole chain
    elif option == "chain":
        pot += 0.25 * mass * (frequency**2) * positions[0]**2
        for y in range(1,num_beads-1):
            pot += 0.5 * mass * (frequency**2) * positions[y]**2
        pot += 0.25 * mass * (frequency**2) * positions[num_beads-1]**2
    return pot

@jit(nopython=True)
# Proposal move for the whole chain
def set_proposal_1(coords, bead_masses, freqs, inv_temp, num_beads, num_seg_beads, num_segments):
    # Transform entire chain to staged coordinates
    staged = stage_coords(coords, num_beads-1, num_seg_beads, num_segments)
    # Perturb the uncoupled mode variable  (PLAY AROUND WITH THIS STEP!!!!!!!!!!)
    staged[0] += np.random.uniform(-1,1)
    staged[num_beads-1] = np.random.normal(0, 1.0/np.sqrt( inv_temp * bead_masses[num_beads-1] * freqs[num_beads-1]**2) )
    # Transform back to primitive coordinates
    return inverse_stage_coords(staged, num_beads-1, num_seg_beads, num_segments)

@jit(nopython=True)
# Proposal move for a segment of the chain
def set_proposal_2(coords, num_seg_beads, num_segments, left_wall, inv_temp, bead_masses, freqs):
    staged = stage_coords(coords, len(coords)-1, num_seg_beads, num_segments)
    # Sample the guassian distribution.
    for y in range(left_wall+1,left_wall + num_seg_beads):
        staged[y] = np.random.normal(0, 1.0/np.sqrt( inv_temp * bead_masses[y] * freqs[y]**2) )
    # Transform back to primitive coords
    return inverse_stage_coords(staged, len(coords)-1, num_seg_beads, num_segments)

@jit(nopython=True)
# Proposal move for a segment of the chain
def set_proposal_3(coords, num_seg_beads, num_segments, left_wall, inv_temp, bead_masses, freqs):
    staged = stage_coords(coords, len(coords)-1, num_seg_beads, num_segments)
    # Sample the guassian distribution.
    staged[left_wall] = np.random.normal(0, 1.0/np.sqrt( inv_temp * bead_masses[left_wall] * freqs[left_wall]**2) )
    return inverse_stage_coords(staged, len(coords)-1, num_seg_beads, num_segments)
