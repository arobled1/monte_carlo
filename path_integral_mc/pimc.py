import numpy as np
import matplotlib.pyplot as plt

def get_harmonic_potential(xi, mass, frequency):
    return 0.5 * mass * frequency* frequency * xi**2

def seg_stage_to_prim(ui, xi, segment_length, rand_bead, num_beads):
    # Transform segment of chain from staging to primitive variables
    for k in range(segment_length, 0, -1):
        xi[(rand_bead+k) % num_beads] = ui[k-1] + (k/(k+1))*xi[(rand_bead+k+1) % num_beads] + (1/(k+1))*xi[rand_bead % num_beads]
    return xi

#=====================================================================
# Block is for parameters
n_steps = 50000                       # Number of MC steps
beta = 1                              # 1/kT
w = 1                                 # Set Frequency
m = 1                                 # Set Mass
pbeads = 10                           # Number of beads
omegaP = np.sqrt(pbeads) / beta       # Set w_P
j = 80                                # For random bead kick
d = 1                                 # For random bead kick
zeta = 0.6                            # For random bead kick
accept = 0                            # For computing acceptance percentage
#======================================================================

# Initialize bead positions
primitives = np.zeros(pbeads)
# Set random initial positions
for r in range(pbeads):
    primitives[r] = np.random.uniform(-1,1)

# Masses for the staging coordinates
m_k = np.zeros(j)
for k in range(1,j+1):
    # m_k[k] = (k+2) * m / (k+1)
    m_k[k-1] = (k+1) * m / k

for i in range(n_steps):
    # Pick a random bead.
    l = np.random.randint(pbeads)
    # Save initial primitives
    old_coords = np.zeros(j)
    for k in range(1,j+1):
        old_coords[k-1] = primitives[(l+k) % pbeads]
    # Define potential from initial primitives
    old_potential = 0
    for k in range(j):
        old_potential += get_harmonic_potential(old_coords[k], m, w)
    old_potential = (old_potential / pbeads)
    # Initialize staged coords
    staged = np.zeros(j)
    # Sample the guassian distribution j times.
    for k in range(j):
        staged[k] = np.random.normal(0, 1/np.sqrt( beta * m_k[k] * omegaP**2 ) )
    # Convert back to primitive coords
    primitives = seg_stage_to_prim(staged, primitives, j, l, pbeads)
    # Define potential from proposed primitives
    proposed_potential = 0
    for k in range(1,j+1):
        proposed_potential += get_harmonic_potential(primitives[(l+k) % pbeads], m, w)
    proposed_potential = proposed_potential / pbeads
    # Acceptance criteria
    Pacc = min(1,np.exp(-beta * (proposed_potential - old_potential)) )
    # Compare Pacc to uni_rand
    uni_rand = np.random.uniform(0,1)
    # Reject proposal if u > Pacc
    if uni_rand > Pacc:
        for k in range(1,j+1):
            primitives[(l+k) % pbeads] = old_coords[k-1]
    # Otherwise accept proposal
    else:
        accept += 1
