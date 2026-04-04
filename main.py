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
velocities = [[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]]
weights = [4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]
iterations = 100
omega = 1/.8
squared_sound_speed = 1/3 # standard for D2Q9

elements = [[[1,0,0,0,0,0,0,0,0] for j in range(n[1])] for i in range(n[0])] # all lattice start with only n_0
updated_elements = [[[0,0,0,0,0,0,0,0,0] for j in range(n[1])] for i in range(n[0])] # for collisions
density = [[1 for j in range(n[1])] for i in range(n[0])]
macroscopic_velocity = [[[0,0] for j in range(n[1])] for i in range(n[0])]

def dot(vector1, vector2):
    return vector1[0]*vector2[0] + vector1[1]*vector2[1]

def collide(y, x):
    for c in range(9):
        d = dot(macroscopic_velocity[y][x], velocities[c])
        taylor = 1 + d/squared_sound_speed + (d**2)/(2*(squared_sound_speed**2)) - dot(macroscopic_velocity[y][x], macroscopic_velocity[y][x])/(2*squared_sound_speed)
        feq = weights[c]*density[y][x]*taylor # maxwell-boltzmann distribution (to be replaced soon)
        updated_elements[y][x][c] = (1-omega)*elements[y][x][c] + omega*feq # BGK approximation

def stream(y, x):
    for c in range(9):
        target = [x+velocities[c][0], y-velocities[c][1]] # next lattice

        # periodic boundary condition (to become bounce-back soon)
        if target[0] < 0: target[0] = n[0]-1
        elif target[0] > n[0]-1: target[0] = 0
        if target[1] < 0: target[1] = n[1]-1
        elif target[1] > n[1]-1: target[1] = 0

        elements[target[1]][target[0]][c] = updated_elements[y][x][c]
    
def measure(y, x):
    density[y][x] = sum(elements[y][x])
    macroscopic_velocity[y][x][0] = sum([velocities[_][0]*elements[y][x][_] for _ in range(9)])/density[y][x]
    macroscopic_velocity[y][x][1] = sum([velocities[_][1]*elements[y][x][_] for _ in range(9)])/density[y][x]
    
# heatmap for density
plt.ion()
fig, ax = plt.subplots()
img = plt.imshow(density, cmap='viridis')
plt.colorbar(img)
    
def iterate():
    density[(n[0]-1)//2][(n[1]-1)//2] = 5

    for _ in range(iterations):
        if _ % 10 == 0:
            print(f'total density: {sum(map(lambda x: sum(x), density))}')
            print(f'max flow speed: {max(map(lambda x: dot(x,x)**(0.5), [a for b in macroscopic_velocity for a in b]))}')

        for i in range(n[0]):
            for j in range(n[1]): 
                collide(i,j)
        
        for i in range(n[0]):
            for j in range(n[1]):
                stream(i,j)

        for i in range(n[0]):
            for j in range(n[1]):
                measure(i,j)

        # updating heatmap
        img.set_data(density)
        img.set_clim(vmin=min(min(density)), vmax=max(max(density)))
        plt.draw()
        plt.pause(0.01)

iterate()
c = datetime.datetime.now()

plt.ioff()
plt.show()

print(c-a)