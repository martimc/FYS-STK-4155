import SolverPDE
import numpy as np
import matplotlib.pyplot as plt

## initializing the solution
def u_init(x):
    return np.sin(np.pi*x)

def analytical_sol(x,t):
    # analytical solution to the heat equation
    return np.sin(np.pi*x)*np.exp(-np.pi**2*t)

dx = 1/10
dt = 1/2*dx**2

L = 1
n_x = int(round(L/dx))
x = np.linspace(0,L,n_x+1)
x0 = 0; xL = 0
u0 = u_init(x[1:-1])

T = 1
n_t = int(round(T/dt))
t = np.linspace(0,T,n_t+1)

solver = SolverPDE.Explicit_ForwardEuler(u0, x0, xL, dx)
u, t_solve = solver.solve(t)

indx1 = 0
indx2 = int(n_t/4)
indx3 = n_t-1

plt.title('time is: %.3fs' % t_solve[indx1], fontsize='xx-large')
plt.plot(x, analytical_sol(x, t_solve[indx1]), 'g-', label='analytical')
plt.plot(x, u[indx1], 'r--', label='numerical')
plt.ylabel('u(x,t)', fontsize='x-large')
plt.xlabel('x', fontsize='x-large')
plt.ylim(0,1)
plt.legend()
plt.show()

plt.title('time is: %.3fs' % t_solve[indx2], fontsize='xx-large')
plt.plot(x, analytical_sol(x, t_solve[indx2]), 'g-', label='analytical')
plt.plot(x, u[indx2], 'r--', label='numerical')
plt.ylabel('u(x,t)', fontsize='x-large')
plt.xlabel('x', fontsize='x-large')
plt.ylim(0,1)
plt.legend()
plt.show()

plt.title('time is: %.3fs' % t_solve[indx3], fontsize='xx-large')
plt.plot(x, analytical_sol(x, t_solve[indx3]), 'g-', label='analytical')
plt.plot(x, u[indx3], 'r--', label='numerical')
plt.ylabel('u(x,t)', fontsize='x-large')
plt.xlabel('x', fontsize='x-large')
plt.ylim(0,1)
plt.legend()
plt.show()
