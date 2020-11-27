import autograd.numpy as np
from autograd import jacobian,hessian,grad
import autograd.numpy.random as npr
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class Neural_Network:
    def __init__(self, X, t, num_neurons, epochs, gamma):
        self.X_data = X
        self.t_data = t
        self.gamma = gamma

        self.num_neurons = num_neurons
        self.N_neurons = np.size(self.num_neurons)
        self.N_outputs = 1
        self.iter = epochs

        self.initialize_deep_param()

    def initialize_deep_param(self):
        self.N_hidden = np.size(self.num_neurons)

        self.P = [None]*(self.N_hidden+1)
        self.P[0] = npr.randn(self.num_neurons[0], 2+1)
        for i in range(1, self.N_hidden):
            self.P[i] = npr.randn(self.num_neurons[i], self.num_neurons[i-1] + 1)

        self.P[-1] = npr.randn(1, self.num_neurons[-1] + 1)

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def f(self, x):
        return np.sin(np.pi*x)

    def trial_func(self, point, P):
        x,t = point
        return (1-t)*self.f(x) + x*(1-x)*t*self.feed_forward(P,point)

    def cost_func(self, P, X, T):
        cost_sum = 0

        gt_jacobi_func = jacobian(self.trial_func)
        gt_hessian_func = hessian(self.trial_func)

        for x in X:
            for t in T:
                point = np.array([x, t])

                gt = self.trial_func(point, P)
                gt_jacobi = gt_jacobi_func(point, P)
                gt_hessian = gt_hessian_func(point, P)

                gt_dt = gt_jacobi[1]
                gt_dx2 = gt_hessian[0][0]

                err = (gt_dt - gt_dx2)**2
                cost_sum += err
        return cost_sum /( np.size(X)*np.size(T) )

    def feed_forward(self, P, point):
        num_coords = np.size(point,0)
        x = point.reshape(num_coords, -1)

        num_values = np.size(x,1)

        x_input = x
        x_prev = x_input

        for l in range(self.N_hidden):
            w_hidden = P[l]

            x_prev = np.concatenate((np.ones((1,num_values)), x_prev), axis=0)

            z_h = np.matmul(w_hidden,x_prev)
            x_h = self.sigmoid(z_h)

            x_prev = x_h

        w_out = P[-1]

        x_prev = np.concatenate((np.ones((1,num_values)), x_prev), axis=0)

        z_out = np.matmul(w_out,x_prev)
        x_out = z_out

        return x_out[0][0]

    def output_function(self, point):
        u = self.trial_func(point, self.P)
        return u

    def analytic_func(self, point):
        x = point[0]
        t = point[1]
        return np.sin(np.pi*x)*np.exp(-np.pi**2*t)

    def train(self):
        print('initial cost: ', self.cost_func(self.P, self.X_data, self.t_data))

        cost_func_grad = grad(self.cost_func, 0)

        for j in range(self.iter):
            cost_grad = cost_func_grad(self.P, self.X_data, self.t_data)

            for l in range(self.N_hidden+1):
                self.P[l] = self.P[l] - self.gamma*cost_grad[l]

            print(j)

        print('final cost: ', self.cost_func(self.P, self.X_data, self.t_data))

if __name__ == '__main__':
    ### Use the neural network:
    npr.seed(15)

    ## Decide the vales of arguments to the function to solve
    Nx = 10; Nt = 10
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0,1,Nt)

    ## Set up the parameters for the network
    num_hidden_neurons = [100, 25]
    num_iter = 20
    lmb = 0.05

    Solver = Neural_Network(x, t, num_hidden_neurons, num_iter, lmb)
    Solver.train()

    ## Store the results
    g_dnn_ag = np.zeros((Nx, Nt))
    G_analytical = np.zeros((Nx, Nt))
    for i,x_ in enumerate(x):
        for j, t_ in enumerate(t):
            point = np.array([x_, t_])
            g_dnn_ag[i,j] = Solver.output_function(point)

            G_analytical[i,j] = Solver.analytic_func(point)

    # Find the map difference between the analytical and the computed solution
    diff_ag = np.abs(g_dnn_ag - G_analytical)
    print('Max absolute difference between the analytical solution and the network: %g'%np.max(diff_ag))

    ## Plot the solutions in two dimensions, that being in position and time

    T,X = np.meshgrid(t,x)

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.set_title('Solution from the deep neural network w/ %d layer'%len(num_hidden_neurons))
    s = ax.plot_surface(T,X,g_dnn_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$');


    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.set_title('Analytical solution')
    s = ax.plot_surface(T,X,G_analytical,linewidth=0,antialiased=False,cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$');

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.set_title('Difference')
    s = ax.plot_surface(T,X,diff_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$');

    ## Take some slices of the 3D plots just to see the solutions at particular times
    indx1 = 0
    indx2 = int(Nt/2)
    indx3 = Nt-1

    t1 = t[indx1]
    t2 = t[indx2]
    t3 = t[indx3]

    # Slice the results from the DNN
    res1 = g_dnn_ag[:,indx1]
    res2 = g_dnn_ag[:,indx2]
    res3 = g_dnn_ag[:,indx3]

    # Slice the analytical results
    res_analytical1 = G_analytical[:,indx1]
    res_analytical2 = G_analytical[:,indx2]
    res_analytical3 = G_analytical[:,indx3]

    # Plot the slices
    plt.figure(figsize=(10,10))
    plt.title("Computed solutions at time = %g"%t1)
    plt.plot(x, res1)
    plt.plot(x,res_analytical1)
    plt.legend(['dnn','analytical'])

    plt.figure(figsize=(10,10))
    plt.title("Computed solutions at time = %g"%t2)
    plt.plot(x, res2)
    plt.plot(x,res_analytical2)
    plt.legend(['dnn','analytical'])

    plt.figure(figsize=(10,10))
    plt.title("Computed solutions at time = %g"%t3)
    plt.plot(x, res3)
    plt.plot(x,res_analytical3)
    plt.legend(['dnn','analytical'])

    plt.show()
