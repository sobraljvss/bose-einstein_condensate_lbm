import datetime, multiprocessing as mp, matplotlib.pyplot as plt, numpy as np, math

t0 = datetime.datetime.now() # recording startup time for optimization comparison

n = 100 # number of lattices

VELOCITIES = np.array(((0,0), (1,0), (-1,0))) # in cartesian coordinates
WEIGHTS = np.array((4/6,1/6,1/6))
ITERATIONS = 50
OMEGA = 1/.8 # time step over relaxation time
SQUARED_SOUND_SPEED = 1/3

elements = np.zeros((1,n,3)) # lattices with 3 distribution functions (f_i)
elements[:,:,0] = 1 # all lattice start with only f_0 = 1
updated_elements = np.zeros_like(elements) # for collisions

density = np.ones((1,n))
macroscopic_velocity = np.zeros((1,n,2)) # in cartesian coordinates
macro_norm = np.sum(macroscopic_velocity*macroscopic_velocity, axis=2) # matrix of |u|^2
max_mach = (np.amax(macro_norm)/SQUARED_SOUND_SPEED)**(0.5) # maximum Mach number

targets = np.zeros((1,n,3,3), dtype='int64')

# preparating matrices for streaming targeted lattices
for i in range(n):
    for c in range(3):
        target = np.array([0, i+VELOCITIES[c,0], c]) # next lattice

        # bounce back
        if target[1] < 0: target = np.array([0, i, 1])
        elif target[1] > n-1: target = np.array([0, i, 2])

        targets[0,i,c] = target

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
    global density, macroscopic_velocity, macro_norm, max_mach

    density = np.sum(elements, axis=2) # summatory over f_i
    macroscopic_velocity = (np.einsum('ijk,kl->ijl', elements, VELOCITIES))/density[:,:,np.newaxis] # summatory over f_i*c_i    

    macro_norm = np.sum(macroscopic_velocity*macroscopic_velocity, axis=2) # matrix of |u|^2
    max_mach = (np.amax(macro_norm)/SQUARED_SOUND_SPEED)**(0.5) # maximum Mach number

# heatmap for density
plt.ion()
fig, ax = plt.subplots()
img = plt.imshow(density, cmap='plasma', aspect=50)
plt.colorbar(img)

# initial conditions for testing only
elements[0,25] *= 5
density[0,25] = 5
def iterate():
    for _ in range(ITERATIONS):
        ax.set_title(f'Iteration #{_+1}')

        if (_+1) % 10 == 0:
            # useful simulation info displayed every 10 iterations
            print(f'total density: {np.sum(density)}')
            print(f'max squared flow speed: {np.amax(macro_norm)}')
            print(f'max Mach number: {max_mach}')

        collide()
        stream()
        measure()

        # updating heatmap
        img.set_data(density)
        img.set_clim(vmin=0, vmax=1.5)
        plt.draw()
        plt.pause(0.01)

iterate()
t1 = datetime.datetime.now() # recording ending time for optimization comparison

plt.ioff()
plt.show()

print(t1-t0) # simulation execution time