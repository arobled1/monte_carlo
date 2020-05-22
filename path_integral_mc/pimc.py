import numpy as np
import matplotlib.pyplot as plt
import copy

# Transform the whole chain to staging coordinates
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

# Transform the whole chain to primitive coordinates
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

# Transform a segment of the chain to staging coordinates
def segment_stage_to_prim(ui, xi, segment_length, rand_bead, num_beads):
    # Define primitive coordinates from staging coordinates.
    for k in range(segment_length, 0, -1):
        xi[(rand_bead+k) % num_beads] = ui[k-1] + (k/(k+1))*xi[(rand_bead+k+1) % num_beads] + (1/(k+1))*xi[rand_bead % num_beads]
    return xi

def get_harmonic_potential(positions, mass, frequency, num_beads, left_wall):
    # If using the entire chain
    if num_beads == len(positions):
        pot = 0
        pot = 0.5 * mass * (frequency**2) * np.dot(positions,positions)
    # If using a segment of the chain
    else:
        pot = 0
        for k in range(1,num_beads+1):
            pot += 0.5 * mass * (frequency**2) * positions[(left_wall+k) % len(positions)]**2
    return pot

# Proposal move for the whole chain
def set_proposal_1(coords):
    # Transform entire chain to staged coordinates
    staged = get_closed_chain_stage_coords(coords, len(coords))
    # Perturb the uncoupled mode variable
    staged[0] += np.random.uniform(-1,1)
    # Transform back to primitive coordinates
    return closed_chain_inverse_stage_coords(staged, len(coords))

# Proposal move for a segment of the chain
def set_proposal_2(coords, segment_length, left_wall, inv_temp, masses, omega):
    staged = []
    # Sample the guassian distribution j times.
    for k in range(segment_length):
        staged.append(np.random.normal(0, 1/np.sqrt( inv_temp * masses[k] * omega**2 ) ) )
    # Transform back to primitive coords
    return segment_stage_to_prim(staged, coords, segment_length, left_wall, len(coords))

#=====================================================================
# Block is for parameters
n_steps = 60000                       # Number of MC steps
beta = 5.26667                        # 1/kT
w = 3                                 # Set Frequency
m = 0.01                              # Set Mass
pbeads = 400                          # Number of beads
omegaP = np.sqrt(pbeads) / beta       # Set w_P
j = 80                                # Number of beads in chain segment
accept = 0                            # To count # of acceptances
#======================================================================

# Initialize bead positions
primitives = np.zeros(pbeads)
# Set initial positions by sampling from the distribution of a 1D free particle
for r in range(pbeads):
    primitives[r] = np.random.normal(0, np.sqrt(beta/m) )

# Masses for the staging coordinates
m_k = np.zeros(j)
for k in range(1,j+1):
    m_k[k-1] = (k+1) * m / k

virial = []
#============================================================================
# PIMC starts here!!!
for i in range(n_steps):
    # Move the whole chain
    if i !=0 and i % (pbeads/j) == 0:
        # Save initial primitives
        old_coords = copy.deepcopy(primitives)
        # Define potential from initial primitives
        old_potential = get_harmonic_potential(primitives, m, w, pbeads, 0)
        # Define proposal
        primitives = set_proposal_1(primitives)
        # Define potential from proposed primitives
        proposed_potential = get_harmonic_potential(primitives, m, w, pbeads, 0)
        # Acceptance criteria
        Pacc = min(1,np.exp(-beta * (1/pbeads)* (proposed_potential - old_potential)) )
        # Compare Pacc to u ~ U(0,1)
        if np.random.uniform(0,1) > Pacc:
            primitives = copy.deepcopy(old_coords)
        else:
            accept += 1
        # Computing the virial energy estimator
        sumv = 0
        for o in range(pbeads):
            sumv += (m*w**2)*primitives[o]**2
        virial.append(sumv / pbeads)
    # Or move a segment of the chain
    else:
        # Pick a random bead.
        l = np.random.randint(pbeads)
        # Save initial primitives
        old_coords = copy.deepcopy(primitives)
        # Define potential from initial primitives
        old_potential = get_harmonic_potential(primitives, m, w, j, l)
        # Define proposal
        primitives = set_proposal_2(primitives, j, l, beta, m_k, omegaP)
        # Define potential from proposed primitives
        proposed_potential = get_harmonic_potential(primitives, m, w, j, l)
        # Acceptance criteria
        Pacc = min(1,np.exp(-beta * (1/pbeads) * (proposed_potential - old_potential)) )
        # Compare Pacc to u ~ U(0,1)
        if np.random.uniform(0,1) > Pacc:
            primitives = copy.deepcopy(old_coords)
        else:
            accept += 1
        # Computing the virial energy estimator
        sumv = 0
        for o in range(pbeads):
            sumv += (m*w**2)*primitives[o]**2
        virial.append(sumv / pbeads)
# PIMC ends here!
#============================================================================
print("Percentage of acceptances: ", (accept/n_steps)*100)
# Compute cumulative average of the virial estimator
cume = np.zeros(len(virial))
cume[0] = virial[0]
for i in range(1,len(virial)):
    cume[i] = (i)/(i+1)*cume[i-1] + virial[i]/(i+1)

steps = np.arange(1,n_steps+1)
# Plotting the virial estimator
plt.xlim(min(steps)-100, max(steps))
plt.ylim(-1,7)
plt.axhline(y=1.5, linewidth=2, color='r', label=r'$\epsilon_{vir} = 1.5$')
plt.plot(steps, virial, '-', color='black', alpha=0.4, label=r'$\epsilon_{vir}$')
plt.plot(steps, cume, '-', color='blue', label='cumulative average')
plt.legend(loc='upper left')
plt.xlabel('# of Steps')
plt.ylabel(r'$\epsilon_{vir} \quad / \quad \hbar \omega$')
plt.savefig('virial.pdf')
plt.clf()
