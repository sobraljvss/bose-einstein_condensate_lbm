### BOSE EINSTEIN CONDENSATE LATTICE BOLTZMANN SIMULATION
## ALMOST 1D BUT 2D CONDENSATE
## TWO ELECTROMAGNETIC FIELDS AS SUPPORT, X WEAKER THAN Y SO IT'S NOT A SPHERE
## BOSE-EINSTEIN DISTRIBUTION AS EQUILIBRIUM AND RELATIVELY LOW KINETIC ENERGY TO SIMULATE LIQUID HELIUM COOLING
## PARTICLES WITH ENERGY ABOVE A THRESHOLD MUST BE TRAPPED OUT THE SYSTEM ONCE THEY TRAVEL TOO FAR, LOWERING AVERAGE TEMPERATURE (EVAPORATIVE COOLING)
## "BRUTE FORCE" DEACCELERATION TO SIMULATE LASER COOLING
## MULTICORE PARALLELISM WILL BE WORKED OUT

import datetime, multiprocessing as mp, numpy as np, matplotlib.pyplot as plt

t0 = datetime.datetime.now() # recording startup time for optimization comparison

n = (100,100) # matrix coordinates

# system's middle point and distances relative to it, that is, the trap's dimensions (TODO: reinsert it into the code)
mid = (n[0]//2,n[1]//2)
var = (n[0]//10,n[1]//2)

VELOCITIES = np.array(((0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1))) # in cartesian coordinates
WEIGHTS = np.array((4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36))
ITERATIONS = 100
OMEGA = 1/.8 # time step over relaxation time
SQUARED_SOUND_SPEED = 1/3 # standard for D2Q9

elements = np.zeros((*n, 9)) # lattices with 9 distribution functions (f_i)
elements[:,:,0] = 1 # all lattice start with only f_0 = 1
updated_elements = np.zeros_like(elements) # for collisions
is_fluid = np.full(n, True) # boolean matrix excluding trapped out particles (TODO: reinsert it into the code)

density = np.ones(n)
macroscopic_velocity = np.zeros((*n,2)) # in cartesian coordinates
macro_norm = np.sum(macroscopic_velocity*macroscopic_velocity, axis=2) # matrix of |u|^2
max_mach = (np.amax(macro_norm)/SQUARED_SOUND_SPEED)**(0.5) # maximum Mach number

targets = np.zeros((*n,9,3), dtype='int64')
electromagnetic_field = np.zeros_like(macroscopic_velocity)
KQ = (.05, .1) # field intensity not regarding position, it has to depend on elements matrix size in order to maintain low Mach number

# preparating matrices for streaming targeted lattices and electromagnetic fields
for i in range(n[0]):
    for j in range(n[1]):
        # distances from fields
        delta_left = j+1
        delta_right = n[1]-j
        delta_up = i+1
        delta_down = n[0]-i

        # field intensity regarding position
        electromagnetic_field[i,j] = np.array((KQ[0]*(1/(delta_left**2) - 1/(delta_right**2)), KQ[1]*(1/(delta_down**2) - 1/(delta_up**2))))

        for c in range(9):
            target = np.array([i-VELOCITIES[c,1], j+VELOCITIES[c,0], c]) # next lattice, vertical has to be inverted because matrix indexing differs from cartesian indexing (above means line before)

            # periodic boundary condition
            if target[0] < 0: target[0] = n[0]-1
            elif target[0] > n[0]-1: target[0] = 0
            if target[1] < 0: target[1] = n[1]-1
            elif target[1] > n[1]-1: target[1] = 0

            targets[i,j,c] = target

def collide():
    global updated_elements
    dotted_fv = np.einsum('ijk,lk->ijl', macroscopic_velocity, VELOCITIES) # matrix of c dot u

    # Maxwell-Boltzmann distribution (TODO: replace it with Bose-Einstein distribution)
    taylor = 1 + dotted_fv/SQUARED_SOUND_SPEED + (dotted_fv**2)/(2*(SQUARED_SOUND_SPEED**2)) - (macro_norm/(2*SQUARED_SOUND_SPEED))[:,:, np.newaxis]
    feq = WEIGHTS*taylor*density[:,:,np.newaxis] # equilibrium function
    updated_elements = (1-OMEGA)*elements + OMEGA*feq # BGK approximation

def stream(): 
    # scattering new distributions to targeted lattices
    elements[targets[...,0], targets[...,1], targets[...,2]] = updated_elements

def measure():
    global density, show_density, macroscopic_velocity, macro_norm, max_mach, is_fluid

    elements[~is_fluid] = [0,0,0,0,0,0,0,0,0] # removing particles through evaporative cooling

    density = np.sum(elements, axis=2) # summatory over f_i
    macroscopic_velocity = (np.einsum('ijk,kl->ijl', elements, VELOCITIES) + electromagnetic_field)/density[:,:,np.newaxis] # summatory over f_i*c_i    
    macroscopic_velocity = .995*macroscopic_velocity # laser cooling deacceleration (TODO: calculate arbitrary deacceleration)

    macro_norm = np.sum(macroscopic_velocity*macroscopic_velocity, axis=2) # matrix of |u|^2
    max_mach = (np.amax(macro_norm)/SQUARED_SOUND_SPEED)**(0.5) # maximum Mach number

    is_fluid.fill(True) # reset boolean matrix so empty lattices can be filled again

# heatmap for density
plt.ion()
fig, ax = plt.subplots()
img = plt.imshow(density, cmap='plasma')
plt.colorbar(img)

def iterate():
    for _ in range(ITERATIONS):
        if (_+1) % 10 == 0:
            # useful simulation info displayed every 10 iterations
            print(f'total density: {np.sum(density)}')
            print(f'max and mean squared flow speed: {np.amax(macro_norm)} | {np.mean(macro_norm)}')
            print(f'max Mach number: {max_mach}')

        collide()
        stream()
        measure()

        # updating heatmap
        img.set_data(density)
        img.set_clim(vmin=0, vmax=np.amax(density))
        plt.draw()
        plt.pause(0.01)

iterate()
t1 = datetime.datetime.now() # recording ending time for optimization comparison

plt.ioff()
plt.show()

print(t1-t0) # simulation execution time