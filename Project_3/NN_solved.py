import NN_class
import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt

### Use the neural network:
npr.seed(15)

## Decide the vales of arguments to the function to solve
Nx = 100; Nt = 10
x = np.linspace(0, 1, Nx)
t = np.linspace(0,1,Nt)

## Set up the parameters for the network
num_hidden_neurons = [100, 25]
num_iter = 50
lmb = 0.05

Solver = NN_class.Neural_Network(x, t, num_hidden_neurons, num_iter, lmb)
Solver.train()

## Store the results
u_ffnn_ag = np.zeros((Nx, Nt))
U_analytical = np.zeros((Nx, Nt))
for i,x_ in enumerate(x):
    for j, t_ in enumerate(t):
        point = np.array([x_, t_])
        u_ffnn_ag[i,j] = Solver.output_function(point)

        U_analytical[i,j] = Solver.analytic_func(point)

# Find the map difference between the analytical and the computed solution
diff_ag = np.abs(u_ffnn_ag - U_analytical)
print('Max absolute difference between the analytical solution and the network: %g'%np.max(diff_ag))

## Take some slices of the 3D plots just to see the solutions at particular times
indx1 = 0
indx2 = int(Nt/4)
indx3 = Nt-1

t1 = t[indx1]
t2 = t[indx2]
t3 = t[indx3]

# Slice the results from the FFNN
res1 = u_ffnn_ag[:,indx1]
res2 = u_ffnn_ag[:,indx2]
res3 = u_ffnn_ag[:,indx3]

# Slice the analytical results
res_analytical1 = U_analytical[:,indx1]
res_analytical2 = U_analytical[:,indx2]
res_analytical3 = U_analytical[:,indx3]

# Plot the slices
plt.title('time is: %.3fs' % t1, fontsize='xx-large')
plt.plot(x, res_analytical1, 'g-', label='analytical')
plt.plot(x, res1, 'r--', label='numerical')
plt.ylabel('u(x,t)', fontsize='x-large')
plt.xlabel('x', fontsize='x-large')
plt.ylim(-0.1,1)
plt.legend()
plt.show()

plt.title('time is: %.3fs' % t2, fontsize='xx-large')
plt.plot(x, res_analytical2, 'g-', label='analytical')
plt.plot(x, res2, 'r--', label='numerical')
plt.ylabel('u(x,t)', fontsize='x-large')
plt.xlabel('x', fontsize='x-large')
plt.ylim(-0.1,1)
plt.legend()
plt.show()

plt.title('time is: %.3fs' % t3, fontsize='xx-large')
plt.plot(x, res_analytical3, 'g-', label='analytical')
plt.plot(x, res3, 'r--', label='numerical')
plt.ylabel('u(x,t)', fontsize='x-large')
plt.xlabel('x', fontsize='x-large')
plt.ylim(0,1)
plt.legend()
plt.show()

plt.figure(figsize=(10,10))
plt.title("time is: %g"%t1)
plt.plot(x, res1)
plt.plot(x,res_analytical1)
plt.legend(['ffnn','analytical'])

plt.figure(figsize=(10,10))
plt.title("time is: %g"%t2)
plt.plot(x, res2)
plt.plot(x,res_analytical2)
plt.legend(['ffnn','analytical'])

plt.figure(figsize=(10,10))
plt.title("time is: %g"%t3)
plt.plot(x, res3)
plt.plot(x,res_analytical3)
plt.legend(['ffnn','analytical'])

plt.show()
