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

# system's middle point and distances relative to it, that is, the trap's dimensions
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
is_fluid = np.full(n, True) # boolean matrix excluding trapped out particles

density = np.ones(n)
show_density = np.ones(n) # only for visualizing
macroscopic_velocity = np.zeros((*n,2)) # in cartesian coordinates
macro_norm = np.sum(macroscopic_velocity*macroscopic_velocity, axis=2) # matrix of |u|^2
max_mach = (np.amax(macro_norm)/SQUARED_SOUND_SPEED)**(0.5) # maximum Mach number

def collide():
    global updated_elements
    dotted_fv = np.einsum('ijk,lk->ijl', macroscopic_velocity, VELOCITIES) # matrix of c dot u

    # maxwell-boltzmann distribution (to be replaced soon)
    taylor = 1 + dotted_fv/SQUARED_SOUND_SPEED + (dotted_fv**2)/(2*(SQUARED_SOUND_SPEED**2)) - (macro_norm/(2*SQUARED_SOUND_SPEED))[:,:, np.newaxis]
    feq = WEIGHTS*taylor*density[:,:,np.newaxis]
    updated_elements = (1-OMEGA)*elements + OMEGA*feq # BGK approximation

def stream(): # still have to vectorise this one
    for i in range(n[0]):
        for j in range(n[1]):
            for c in range(9):
                if is_fluid[i,j]: # only lattices with particles can stream to other lattices
                    target = np.array([i-VELOCITIES[c,1], j+VELOCITIES[c,0]]) # next lattice, vertical has to be inverted because matrix indexing differs from cartesian indexing (above means line before)

                    # periodic boundary condition (electromagnetic fields will impose its own boundary condition soon)
                    if target[0] < 0: target[0] = n[0]-1
                    elif target[0] > n[0]-1: target[0] = 0
                    if target[1] < 0: target[1] = n[1]-1
                    elif target[1] > n[1]-1: target[1] = 0

                    elements[*target, c] = updated_elements[i,j,c]

def simulate_electromagnetic_fields():
    # TODO: replace sample values for acceleration and velocity threshold with the ones based in electromagnetic force and maximum allowed kinectic energy
    for i in range(n[0]):
        for j in range(n[1]):
            # builds a rectangle trap in the middle, smooth the edges out so that it becomes ellipsoidal
            if i in range(mid[0]-var[0],mid[0]+var[0]) and j in range(mid[1]-var[1],mid[1]+var[1]):
                # macroscopic velocity towards middle point increases with distance from it
                macroscopic_velocity[i,j] += [.005*((mid[1])-j)/(var[1]),-.01*((mid[0])-i)/(var[0])]
            else:
                # evaporative cooling: trapping high energy particles out
                if np.mean(macro_norm) > .5 and macro_norm[i,j] >= 2*np.mean(macro_norm):
                    macroscopic_velocity[i,j] = [0,0]
                    is_fluid[i,j] = False
                else:
                    # attracts particles far from center with maximum intensity
                    if i < mid[0]-var[0]:
                        macroscopic_velocity[i,j] += [0,-.01]
                    elif i > mid[0]+var[0]:
                        macroscopic_velocity[i,j] += [0,.01]
                    if j < mid[1]-var[1]:
                        macroscopic_velocity[i,j] += [.005,0]
                    elif j > mid[1]+var[1]:
                        macroscopic_velocity[i,j] += [-.005,0]            
    
def measure():
    global density, show_density, macroscopic_velocity, macro_norm, max_mach, is_fluid
    density = np.sum(elements, axis=2) # summatory over f_i
    show_density = np.sum(elements, axis=2)

    # avoiding zero density on empty lattices so there is no division by zero; flow speed will be (0,0) anyway
    density[~is_fluid] = 10^-5
    show_density[~is_fluid] = 0

    macroscopic_velocity = (np.einsum('ijk,kl->ijl', elements, VELOCITIES)/density[:,:,np.newaxis]) # summatory over f_i*c_i
    simulate_electromagnetic_fields()
    
    macroscopic_velocity[~is_fluid] = [0,0]
    macro_norm = np.sum(macroscopic_velocity*macroscopic_velocity, axis=2) # matrix of |u|^2
    max_mach = (np.amax(macro_norm)/SQUARED_SOUND_SPEED)**(0.5) # maximum Mach number

    is_fluid = np.full(n, True) # reset boolean matrix so empty lattices can be filled again


# heatmap for density
plt.ion()
fig, ax = plt.subplots()
img = plt.imshow(show_density, cmap='plasma')
plt.colorbar(img)
    
def iterate():
    for _ in range(ITERATIONS):
        if _ % 10 == 0:
            # useful simulation info displayed every 10 iterations
            print(f'total density: {np.sum(show_density)}')
            print(f'max squared flow speed: {np.amax(macro_norm)}')
            print(f'max Mach number: {max_mach}')

        collide()
        stream()
        measure()

        # updating heatmap
        img.set_data(show_density)
        img.set_clim(vmin=0, vmax=np.amax(show_density))
        plt.draw()
        plt.pause(0.01)


iterate()
t1 = datetime.datetime.now() # recording ending time for optimization comparison

plt.ioff()
plt.show()

print(t1-t0) # simulation execution time