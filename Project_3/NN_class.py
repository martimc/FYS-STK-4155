import autograd.numpy as np
from autograd import jacobian,hessian,grad
import autograd.numpy.random as npr
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class Neural_Network:
    def __init__(self, X, t, num_neurons, epochs, gamma):
        #initializing the solver class
        self.X_data = X
        self.t_data = t
        self.gamma = gamma

        self.num_neurons = num_neurons
        self.N_neurons = np.size(self.num_neurons)
        self.N_outputs = 1
        self.iter = epochs

        self.initialize_deep_param()

    def initialize_deep_param(self):
        #initializing the deep parameters using random numbers
        self.N_hidden = np.size(self.num_neurons)

        self.P = [None]*(self.N_hidden+1)
        self.P[0] = npr.randn(self.num_neurons[0], 2+1)
        for i in range(1, self.N_hidden):
            self.P[i] = npr.randn(self.num_neurons[i], self.num_neurons[i-1] + 1)

        self.P[-1] = npr.randn(1, self.num_neurons[-1] + 1)

    def sigmoid(self, z):
        #activation function
        return 1/(1 + np.exp(-z))

    def f(self, x):
        return np.sin(np.pi*x)

    def trial_func(self, point, P):
        #trial function that is used to calculate x,t and to update the
        #deep parameters
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
        #the feed forward algorithm

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
            #back propagation by finding the gradient of the cost function
            #and updating the deep parameters
            cost_grad = cost_func_grad(self.P, self.X_data, self.t_data)

            for l in range(self.N_hidden+1):
                self.P[l] = self.P[l] - self.gamma*cost_grad[l]

            print(j)

        print('final cost: ', self.cost_func(self.P, self.X_data, self.t_data))
