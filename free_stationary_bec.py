import numpy as np, matplotlib.pyplot as plt, math

a,b,N = 0,1,5000 # interval and steps
x = np.linspace(a,b,N+1) # interval of x
dx = (b-a)/N # step size
n = 1/dx
tolerance_nk = -6 # tolerance

# mu = chemical potential
# sigma = particle interation

def numerical(mu=1, sigma=-1): # numerical approximation via Newton-Kantorovich method and Finite Differences
    F_n = np.zeros((N+1)) # wavefunction
    F_np = np.ones((N+1)) # next iteration of wavefunction

    # Newton-Kantorovich
    for j in range(1000000):
        # ordinary differential equation produced by Frechét derivative
        F_n = F_np.copy()
        g = -2*mu -6*sigma*np.pow(F_n, 2)
        S = 4*sigma*np.pow(F_n, 3)

        # Finite Differences
        A = np.zeros((N+1,N+1))
        B = np.zeros(N+1)

        # initial condition: F(a) = sqrt(2μ)
        A[0,0] = 1
        B[0] = math.sqrt(2*mu)

        # initial condition: F'(a) = 0
        A[-1, :2] = n*np.array([-1, 1])
        B[-1] = 0

        for i in range(1, N):
            A[i, i-1:i+2] = np.array([n**2, g[i]-2*n**2, n**2])
            B[i] = -S[i]

        F_np = np.linalg.solve(A,B)

        if np.max(np.abs(F_np - F_n)) < 10**tolerance_nk: break

    return F_np

def analytical(mu=1, sigma=-1): # analytical solution
    return math.sqrt(2*mu)/np.cosh(math.sqrt(2*mu)*x) # this solution works for mu > 0 and sigma = -1

# graph
fig = plt.figure()
axs = [fig.add_axes([0.15,.5,.8,.4]), fig.add_axes([0.15,.1,.3,.3])]

# solutions plot
data = numerical()
line, = axs[0].plot(x, data, 'steelblue', linewidth=6, label='Aproximação numérica')

data2 = analytical()
line, = axs[0].plot(x, data2, 'red', label='Solução analítica')

axs[0].set_title('Soluções')
axs[0].legend(loc=3)
axs[0].set_xlim(0,1)

# error plot
error = data2-data
line_e, = axs[1].plot(x,error,color='green')

axs[1].set_title(f'Erro')
axs[1].set_xlim(0,1)

# additional info
plt.text(.95,0.1,f'''-(1/2)ϕ" + σ|ϕ|²ϕ = -μϕ; μ = 1, σ = -1\n
Condições iniciais: ϕ(0) = √(2μ); ϕ\'(0) = 0\n
Quantidade de passos: {N}\n
Tolerância: 10^{tolerance_nk}\n
Erro máximo: {100*error[np.abs(error).argmax()]/data2[np.abs(error).argmax()]}%''', transform=plt.gcf().transFigure, horizontalalignment='right')

plt.tight_layout()
plt.show()