import numpy as np
import numba
from numba import jit

@jit(nopython=True)
# Update velocity at a half step dt/2
def upd_velocity(veloc, force, deltat, mass):
    veloc = (force * deltat)/(2 * mass)
    return veloc

@jit(nopython=True)
# Update position at a half step dt/2
def upd_position(posit, veloc, deltat):
    posit = 0.5 * veloc * deltat
    return posit

@jit(nopython=True)
# Use langevin thermostat to add randomness to your velocities
def rand_kick(veloc, friction, boltz, mass, deltat):
    gaussian = np.random.normal(0,1)
    sqt1 = np.sqrt(1 - np.exp(-2*friction*deltat) )
    sqt2 = np.sqrt(boltz) / np.sqrt(mass)
    random = gaussian * sqt1 * sqt2
    veloc = veloc * np.exp(- friction * deltat) + random
    return veloc

@jit(nopython=True)
# Compute force from the harmonic springs in ring
def get_harmonic_force(force, xi, mass, frequency):
    force = - mass * (frequency**2) * xi
    return force

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
def run_pimc(primitives_r, primitive_z, num_steps, pbeads, j, capital_n, mass, omega, omegaP, beta, bead_masses, freqs, z_force_const):
    accept = 0                            # To count # of acceptances
    force_on_z = 0
    # lastu_bead = np.zeros(0)
    #============================================================================
    # PIMC starts here!!!
    counter = 1
    while counter <= num_steps:
        # Sampling intermediate beads
        if j != 1:
            for mc_count in range(capital_n):
                if counter > num_steps:
                    break
                # Pick a random segment.
                group_num = np.random.randint(capital_n)
                # Save initial primitives
                old_coords = np.zeros(pbeads+1)
                for w in range(pbeads+1):
                    old_coords[w] = primitives_r[w]
                # Define potential from initial primitives
                old_potential = get_harmonic_potential(primitives_r, primitive_z, mass, omega, j, group_num*j, "ints", z_force_const)
                # Define proposal
                primitives_r = set_proposal_2(primitives_r, j, capital_n, group_num*j, beta, bead_masses, freqs)
                # Define potential from proposed primitives
                proposed_potential = get_harmonic_potential(primitives_r, primitive_z, mass, omega, j, group_num*j, "ints", z_force_const)
                # Acceptance criteria
                Pacc = min(1,np.exp(-beta * (1/pbeads) * (proposed_potential - old_potential)) )
                # Compare Pacc to u ~ U(0,1)
                if np.random.uniform(0,1) > Pacc:
                    for w in range(pbeads+1):
                        primitives_r[w] = old_coords[w]
                else:
                    accept += 1
                # force_on_z += z_force_const*((primitives_r[0] - primitives_r[pbeads]) - primitive_z)
                counter += 1
        # Sampling endpoint beads
        for mc_count in range(capital_n-1):
            if counter > num_steps:
                break
            # Pick a random endpoint bead.
            group_num = np.random.randint(1,capital_n)
            # Save initial primitives
            old_coords = np.zeros(pbeads+1)
            for w in range(pbeads+1):
                old_coords[w] = primitives_r[w]
            # Define potential from initial primitives
            old_potential = get_harmonic_potential(primitives_r, primitive_z, mass, omega, j, group_num*j, "ends", z_force_const)
            # Define proposal
            primitives_r = set_proposal_3(primitives_r, j, capital_n, group_num*j, beta, bead_masses, freqs)
            # Define potential from proposed primitives
            proposed_potential = get_harmonic_potential(primitives_r, primitive_z, mass, omega, j, group_num*j, "ends", z_force_const)
            # Acceptance criteria
            Pacc = min(1,np.exp(-beta * (1/pbeads) * (proposed_potential - old_potential)) )
            # Compare Pacc to u ~ U(0,1)
            if np.random.uniform(0,1) > Pacc:
                for w in range(pbeads+1):
                    primitives_r[w] = old_coords[w]
            else:
                accept += 1
            # force_on_z += z_force_const*((primitives_r[0] - primitives_r[pbeads]) - primitive_z)
            counter += 1
        # Sampling the whole chain
        if counter > num_steps:
            break
        # Save initial primitives
        old_coords = np.zeros(pbeads+1)
        for w in range(pbeads+1):
            old_coords[w] = primitives_r[w]
        # Define potential from initial primitives
        old_potential = get_harmonic_potential(primitives_r, primitive_z, mass, omega, pbeads+1, group_num*j, "chain", z_force_const)
        # Define proposal
        primitives_r = set_proposal_1(primitives_r, bead_masses, freqs, beta, pbeads+1, j, capital_n)
        # Define potential from proposed primitives
        proposed_potential = get_harmonic_potential(primitives_r, primitive_z, mass, omega, pbeads+1, group_num*j, "chain", z_force_const)
        # Acceptance criteria
        Pacc = min(1,np.exp(-beta * (1/pbeads) * (proposed_potential - old_potential)) )
        # Compare Pacc to u ~ U(0,1)
        if np.random.uniform(0,1) > Pacc:
            for w in range(pbeads+1):
                primitives_r[w] = old_coords[w]
        else:
            accept += 1
        force_on_z += z_force_const*((primitives_r[0] - primitives_r[pbeads]) - primitive_z)
        counter += 1
    # print("what's acceptance ratio? ", accept)
    return force_on_z/num_steps, primitives_r

@jit(nopython=True)
def get_harmonic_potential(positions, z_pos, mass, frequency, num_beads, left_wall, option, kay):
    pot = 0
    # If sampling intermediate beads
    if option == "ints":
        for y in range(left_wall+1,left_wall + num_beads):
            pot += 0.5 * mass * (frequency**2) * positions[y]**2
        pot += 0.5 * kay * ((positions[0] - positions[len(positions)-1]) - z_pos)**2
    # If sampling endpoint beads
    elif option == "ends":
        pot += 0.5 * mass * (frequency**2) * positions[left_wall]**2
        pot += 0.5 * kay * ((positions[0] - positions[len(positions)-1]) - z_pos)**2
    # If sampling the whole chain
    elif option == "chain":
        pot += 0.25 * mass * (frequency**2) * positions[0]**2
        for y in range(1,num_beads-1):
            pot += 0.5 * mass * (frequency**2) * positions[y]**2
        pot += 0.25 * mass * (frequency**2) * positions[num_beads-1]**2
        pot += 0.5 * kay * ((positions[0] - positions[num_beads-1]) - z_pos)**2
    return pot

@jit(nopython=True)
# Proposal move for the whole chain
def set_proposal_1(coords, bead_masses, freqs, inv_temp, num_beads, num_seg_beads, num_segments):
    # Transform entire chain to staged coordinates
    staged = stage_coords(coords, num_beads-1, num_seg_beads, num_segments)
    # Perturb the uncoupled mode variable
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
