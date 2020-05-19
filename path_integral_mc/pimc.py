import numpy as np
import matplotlib.pyplot as plt

def get_harmonic_potential(xi, mass, frequency):
    return 0.5 * mass * frequency* frequency * xi**2

def get_closed_chain_stage_coords(xi, num_beads):
    up_u = np.zeros(num_beads)
    # For the first bead
    up_u[0] = xi[0]
    # For the in between beads
    for k in range(1,num_beads-1):
        up_u[k] = xi[k] - ( k*xi[k+1] + xi[0] ) / (k + 1)
    # For the last bead
    up_u[num_beads-1] = xi[num_beads-1] - ( ( (num_beads-2)*xi[0] ) + xi[0] )/(num_beads - 1)
    return up_u

def closed_chain_inverse_stage_coords(ui, num_beads):
    up_x = np.zeros(num_beads)
    # For the first bead
    up_x[0] = ui[0]
    # For the P bead
    up_x[num_beads-1] = ui[num_beads-1] + ((num_beads-1)/(num_beads))*up_x[0] + (1/(num_beads))*ui[0]
    # For the in between beads (loop goes backwards from bead P-1 to bead 1)
    for k in range(num_beads - 2, 0,-1):
        up_x[k] = ui[k] + (k/(k+1))*up_x[k+1] + (1/(k+1))*ui[0]
    return up_x

def seg_stage_to_prim(ui, xi, segment_length, rand_bead, num_beads):
    # Transform segment of chain from staging to primitive variables
    for k in range(segment_length, 0, -1):
        xi[(rand_bead+k) % num_beads] = ui[k-1] + (k/(k+1))*xi[(rand_bead+k+1) % num_beads] + (1/(k+1))*xi[rand_bead % num_beads]
    return xi

#=====================================================================
# Block is for parameters
n_steps = 50000                       # Number of MC steps
beta = 5.26667                               # 1/kT
w = 3                                 # Set Frequency
m = 0.01                                 # Set Mass
pbeads = 400                           # Number of beads
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

virial = []
for i in range(n_steps):
    if i !=0 and i % (pbeads/j) == 0:
        # Define potential from initial primitives
        old_potential = 0
        for k in range(pbeads):
            old_potential += get_harmonic_potential(primitives[k], m, w)
        old_potential = (old_potential / pbeads)
        # Initialize array for saving old coordinates
        old_all_coords = np.zeros(pbeads)
        # Saving coordinates in case of rejection
        for k in range(pbeads):
            old_all_coords[k] = primitives[k]
        # Transform entire chain to staged coords
        staged_whole_chain = np.zeros(pbeads)
        staged_whole_chain = get_closed_chain_stage_coords(primitives, pbeads)
        # Define proposal (KEEP TRACK OF OLD COORDS AND NEW COORDS)
        delta = np.random.uniform(-1,1)
        # staged_whole_chain[0] += (1/np.sqrt(d)) * (zeta - 0.5) * delta
        staged_whole_chain[0] += delta
        primitives = closed_chain_inverse_stage_coords(staged_whole_chain, pbeads)
        # Define proposed potential
        proposed_potential = 0
        for k in range(pbeads):
            proposed_potential += get_harmonic_potential(primitives[k], m, w)
        proposed_potential = (proposed_potential / pbeads)
        # Acceptance criteria
        Pacc = min(1,np.exp(-beta * (proposed_potential - old_potential)) )
        # Compare Pacc to uni_rand
        uni_rand = np.random.uniform(0,1)
        if uni_rand > Pacc:
            for k in range(pbeads):
                primitives[k] = old_all_coords[k]
        else:
            accept += 1
        # Computing the virial energy estimator
        sumv = 0
        for o in range(pbeads):
            sumv += (m*w**2)*primitives[o]**2
        virial.append(sumv / pbeads)
    else:
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
        # Computing the virial energy estimator
        sumv = 0
        for o in range(pbeads):
            sumv += (m*w**2)*primitives[o]**2
        virial.append(sumv / pbeads)

steps = np.arange(1,n_steps+1)
# Plotting the virial estimator
plt.xlim(min(steps)-100, max(steps))
# plt.ylim(0,3)
plt.ylim(min(virial)-2 ,max(virial)+2)
# plt.axhline(y=1.5, linewidth=2, color='r')
plt.plot(steps, virial, '-', color='black')
plt.xlabel('# of Steps')
plt.ylabel('Energy')
plt.savefig('virial.pdf')
plt.clf()
