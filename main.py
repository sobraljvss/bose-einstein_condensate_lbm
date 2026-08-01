### BOSE EINSTEIN CONDENSATE LATTICE BOLTZMANN SIMULATION
## ALMOST 1D BUT 2D CONDENSATE
## TWO ELECTROMAGNETIC FIELDS AS SUPPORT, X WEAKER THAN Y SO IT'S NOT A SPHERE
## BOSE-EINSTEIN DISTRIBUTION AS EQUILIBRIUM AND RELATIVELY LOW KINETIC ENERGY TO SIMULATE LIQUID HELIUM COOLING
## PARTICLES WITH ENERGY ABOVE A THRESHOLD MUST BE TRAPPED OUT THE SYSTEM ONCE THEY TRAVEL TOO FAR, LOWERING AVERAGE TEMPERATURE (EVAPORATIVE COOLING)
## "BRUTE FORCE" DEACCELERATION TO SIMULATE LASER COOLING
## MULTICORE PARALLELISM WILL BE WORKED OUT

import numpy as np, matplotlib.pyplot as plt

a,b,N = 0,10,100 # interval and steps
dx = (b-a)/N # step size
n = 1/dx
tolerance_nk = 10**-6 # tolerance

def f(laser_intensity=1, alpha=1, chemical_potential=1, particle_interation=1):
    F_n = np.arange(a,b+dx,dx) # wavefunction
    F_np = np.sin(np.arange(a,b+dx,dx)) # next iteration of wavefunction

    # Newton-Kantorovich
    while np.max(np.abs(F_np - F_n)) > tolerance_nk:
        # ordinary differential equation produced by Frechét derivative
        F_n = F_np.copy()
        g = laser_intensity*np.cos(alpha*np.arange(a,b+dx,dx)) + chemical_potential - np.arange(a,b+dx,dx)**2 - 3*particle_interation*F_n**2
        S = 2*particle_interation*F_n**3

        # Finite Differences
        A = np.zeros((N+1,N+1))
        B = np.zeros(N+1)

        # initial condition: F(a) = 0
        A[0,0] = 1
        B[0] = 0

        # initial condition: F'(b) = 0
        A[-1, -2:] = np.array([-1, 1])
        B[-1] = 0

        for i in range(1, N):
            A[i, i-1:i+2] = np.array([n**2, g[i]-2*n**2, n**2])
            B[i] = -S[i]

        F_np = np.linalg.solve(A,B)

    sq_F = np.pow(F_np, 2) # squared values for probability density
    sq_F /= np.sum(sq_F) # normalization
    return sq_F

# graph
fig, ax = plt.subplots()
data = f()
line, = ax.plot(np.arange(a,b+dx,dx), data)
fig.subplots_adjust(bottom=.35)

# sliders for (arbitraries, for the time being) parameters
laser_slider = plt.Slider(ax=fig.add_axes((0.25, 0.1, 0.65, 0.03)),label='laser_intensity',valmin=0.1,valmax=10,valinit=1)
alpha_slider = plt.Slider(ax=fig.add_axes((0.25, 0.15, 0.65, 0.03)),label='alpha',valmin=0.1,valmax=10,valinit=1)
chemp_slider = plt.Slider(ax=fig.add_axes((0.25, 0.2, 0.65, 0.03)),label='chemical_potential',valmin=0.1,valmax=10,valinit=1)
pi_slider = plt.Slider(ax=fig.add_axes((0.25, 0.25, 0.65, 0.03)),label='particle_interation',valmin=0.1,valmax=10,valinit=1)

def update(val):
    # update graph with sliders values
    data =f(laser_slider.val,alpha_slider.val,chemp_slider.val,pi_slider.val)
    ax.set_ylim(np.min(data), np.max(data))
    line.set_ydata(data)
    fig.canvas.draw_idle()

laser_slider.on_changed(update)
alpha_slider.on_changed(update)
chemp_slider.on_changed(update)
pi_slider.on_changed(update)

plt.show()