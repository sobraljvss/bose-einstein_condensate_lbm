### BOSE EINSTEIN CONDENSATE LATTICE BOLTZMANN SIMULATION
## ALMOST 1D BUT 2D CONDENSATE
## TWO ELECTROMAGNETIC FIELDS AS SUPPORT, X WEAKER THAN Y SO ITS NOT A SPHERE
## BOSE-EINSTEIN DISTRIBUTION AS EQUILIBRIUM AND RELATIVELY LOW KINETIC ENERGY TO SIMULATE LIQUID HELIUM COOLING
## PARTICLES WITH ENERGY ABOVE A THRESHOLD MUST BE TRAPPED OUT THE SYSTEM ONCE THEY TRAVEL TOO FAR, LOWERING AVERAGE TEMPERATURE (just like coffee)
## "BRUTE FORCE" DEACCELERATION TO SIMULATE LASER COOLING
## MULTICORE PARALLELISM WILL BE WORKED OUT

import datetime, multiprocessing as mp, numpy as np, matplotlib.pyplot as plt

a = datetime.datetime.now()

n = (125,125)
velocities = np.array(((0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)))
weights = np.array((4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36))
iterations = 100
omega = 1/.8
squared_sound_speed = 1/3 # standard for D2Q9

elements = np.zeros((*n, 9)) # all lattice start with only n_0
elements[:,:,0] = 1
updated_elements = np.zeros_like(elements) # for collisions

density = np.ones(n)
macroscopic_velocity = np.zeros((*n,2))

def collide():
    global updated_elements
    dotted_fv = np.einsum('ijk,lk->ijl', macroscopic_velocity, velocities) # matrix of c dot u
    macro_norm = np.sum(macroscopic_velocity*macroscopic_velocity, axis=2) # matrix of |u|^2

    # maxwell-boltzmann distribution (to be replaced soon)
    taylor = 1 + dotted_fv/squared_sound_speed + (dotted_fv**2)/(2*(squared_sound_speed**2)) - (macro_norm/(2*squared_sound_speed))[:,:, np.newaxis]
    feq = weights*taylor*density[:,:,np.newaxis] 

    updated_elements = (1-omega)*elements + omega*feq # BGK approximation


def stream(): # still have to vectorise this one
    for x in range(n[0]):
        for y in range(n[1]):
            for c in range(9):
                target = [x+velocities[c,0], y-velocities[c,1]] # next lattice, vertical has to be inverted because matrix indexing differs from cartesian indexing (above means line before)

                # periodic boundary condition (electromagnetic fields will impose its own boundary condition soon)
                if target[0] < 0: target[0] = n[0]-1
                elif target[0] > n[0]-1: target[0] = 0
                if target[1] < 0: target[1] = n[1]-1
                elif target[1] > n[1]-1: target[1] = 0

                elements[*target, c] = updated_elements[x,y,c]
    
def measure():
    global density, macroscopic_velocity
    density = np.sum(elements, axis=2) # summatory over n from 1 to 9
    macroscopic_velocity = np.einsum('ijk,kl->ijl', elements, velocities)/density[:,:,np.newaxis] # summatory over n*c from 1 to 9
    
# heatmap for density
plt.ion()
fig, ax = plt.subplots()
img = plt.imshow(density, cmap='plasma')
plt.colorbar(img)
    
def iterate():
    for _ in range(iterations):
        if _ % 10 == 0:
            print(f'total density: {np.sum(density)}')
            print(f'max flow speed: {np.amax(np.sum(macroscopic_velocity*macroscopic_velocity, axis=2)**(.5))}')

        collide()
        stream()
        measure()

        # updating heatmap
        img.set_data(density)
        img.set_clim(vmin=np.amin(density), vmax=np.amax(density))
        plt.draw()
        plt.pause(0.01)

iterate()
c = datetime.datetime.now()

plt.ioff()
plt.show()

print(c-a)