### BOSE EINSTEIN CONDENSATE LATTICE BOLTZMANN SIMULATION
## CONDENSATE SHALL BE 2D, BUT ALMOST 1D
## THERE MUST BE TWO ELECTROMAGNETIC FIELDS AS SUPPORT, X WEAKER THAN Y SO ITS NOT A SPHERE
## PARTICLES SHALL HAVE BOSE-EINSTEIN DISTRIBUTION AS EQUILIBRIUM AND RELATIVELY LOW KINETIC ENERGY,
#  SIMULATING THEY HAVE ALREADY BEEN COOLED DOWN WITH LIQUID HELIUM
## STILL, PARTICLES WITH ENERGY ABOVE A THRESHOLD MUST BE TRAPPED OUT THE SYSTEM ONCE THEY TRAVEL TOO FAR,
#  SIMULATING QUANTUM COOLING
## LASER COOLING WILL BE IMPLEMENTED THROUGH A "BRUTE FORCE" DEACCELERATION
## MULTICORE PARALLELISM WILL BE WORKED OUT

import datetime


a = datetime.datetime.now()

n = (11,11)
#                6       2      5         3      o      1         7       4       8
velocities = [[[-1,1], [0,1], [1,1]], [[-1,0], [0,0], [1,0]], [[-1,-1], [0,-1], [1,-1]]]
weights = [[1/36,1/9,1/36],[1/9,1/4,1/9],[1/36,1/9,1/36]]
dx = dy = dt = 1
iterations = 10
omega = 1/2.25

elements = [[velocities[:] for j in range(n[0])] for i in range(n[1])]
elements[(n[0]-1)//2][(n[1]-1)//2] = [[[-1/36,1/36], [0,1/9], [1/36,1/36]], [[-1/9,0], [0,0], [1/9,0]], [[-1/36,-1/36], [0,-1/9], [1/36,-1/36]]]
updated_elements = [[[[[0]*2]*3]*3 for j in range(n[0])] for i in range(n[1])]

feqs = [[[[[0]*2]*3]*3 for j in range(n[0])] for i in range(n[1])]
feqs[(n[0]-1)//2][(n[1]-1)//2] = [[[-1/36,1/36], [0,1/9], [1/36,1/36]], [[-1/9,0], [0,0], [1/9,0]], [[-1/36,-1/36], [0,-1/9], [1/36,-1/36]]]

def mod(vector):
    return (vector[0]**2 + vector[1]**2)**(0.5)

def collide(x, y):
    for g in range(3):
        for c in range(3):
            updated_elements[x][y][g][c][0] = (1-omega)*elements[x][y][g][c][0] + omega*feqs[x][y][g][c][0]
            updated_elements[x][y][g][c][1] = (1-omega)*elements[x][y][g][c][1] + omega*feqs[x][y][g][c][1]

def stream(x, y): # OH BOY
    elements[x][y][1][1] = updated_elements[x][y][1][1] #f0
    elements[x][y+dy if y < n[1]-1 else y][1][2] = updated_elements[x][y][1][2] #f1
    elements[x-dx if x > 0 else x][y][0][1] = updated_elements[x][y][0][1] #f2
    elements[x][y-dy if y > 0 else y][1][0] = updated_elements[x][y][1][0] #f3
    elements[x+dx if x < n[0]-1 else x][y][2][1] = updated_elements[x][y][2][1] #f4
    elements[x-dx if x > 0 else x][y+dy if y < n[1]-1 else y][0][2] = updated_elements[x][y][0][2] #f5
    elements[x-dx if x > 0 else x][y-dy if y > 0 else y][0][0] = updated_elements[x][y][0][0] #f6
    elements[x+dx if x < n[0]-1 else x][y-dy if y > 0 else y][2][0] = updated_elements[x][y][2][0] #f7
    elements[x+dx if x < n[0]-1 else x][y+dy if y < n[1]-1 else y][2][2] = updated_elements[x][y][2][2] #f8
    # NO BB RIGHT NOW
    

def measure(x, y):
    temp = sum(map(lambda x: mod(x), [*[_ for _ in [*elements[x][y][0]]], *[_ for _ in [*elements[x][y][1]]], *[_ for _ in [*elements[x][y][2]]]]))
    #feqs[x][y] = [[weights[i][j]*temp for j in range(3)] for i in range(3)]
    feqs[x][y] = [[[(-1/36)*temp,(1/36)*temp], [0,(1/9)*temp], [(1/36)*temp,(1/36)*temp]], [[(-1/9)*temp,0], [0,0], [(1/9)*temp,0]], [[(-1/36)*temp,(-1/36)*temp], [0,(-1/9)*temp], [(1/36)*temp,(-1/36)*temp]]]

for _ in range(iterations):
    print(f'collision #{_}')
    for i in range(n[0]):
        for j in range(n[1]): 
            collide(i,j)
    
    print(f'stream #{_}')
    for i in range(n[0]):
        for j in range(n[1]):
            stream(i,j)

    for i in range(n[0]):
        for j in range(n[1]): measure(i,j)


temperatures = [[sum(map(lambda x: mod(x), [*[_ for _ in [*elements[i][j][0]]], *[_ for _ in [*elements[i][j][1]]], *[_ for _ in [*elements[i][j][2]]]]))for j in range(n[1])] for i in range(n[0])]

for i in range(n[0]):
    print(temperatures[i])

c = datetime.datetime.now()


print(c-a)


'''             +
  6  2  5    6  2  5    6  2  5
  3  o  1    3  o  1    3  o  1
  7  4  8    7  4  8    7  4  8

  6  2  5    6  2  5    6  2  5
- 3  o  1    3  o  1    3  o  1 +
  7  4  8    7  4  8    7  4  8

  6  2  5    6  2  5    6  2  5
  3  o  1    3  o  1    3  o  1
  7  4  8    7  4  8    7  4  8
                -

(1,1)



'''