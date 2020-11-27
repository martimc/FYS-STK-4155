import numpy as np

def terminate(u, t, step_no):
    eps = 1.0E-6
    return np.sum((u[step_no+1]-u[step_no]**2)) < eps

class SolverPDE(object):
    def __init__(self, u0, x0, xL, dx):
        if isinstance(u0, (float,int)):
            self.neq = 1
            u0 = float(u0)
        else:
            u0 = np.asarray(u0)
            self.neq = u0.size
        self.u0 = u0

        self.x0 = x0
        self.xL = xL

        self.dx = dx
        #self.f = lambda u, t: np.asarray(f(u, t), float)

    def advance(self):
        raise NotImplementedError

    def solve(self, time_points, terminate=None):
        if terminate is None:
            terminate = lambda u, t, step_no: False

        self.t = np.asarray(time_points)
        self.dt = self.t[1]-self.t[0]
        n = self.t.size
        self.u = np.zeros((n, self.neq+2))

        self.u[0] = np.concatenate(([self.x0],self.u0,[self.xL]))

        for k in range(n-1):
            self.k = k
            self.u[k+1] = self.advance()
            if terminate(self.u, self.t, self.k):
                break
        return self.u[:k+2], self.t[:k+2]


class Explicit_ForwardEuler(SolverPDE):
    def advance(self):
        u, k, t = self.u, self.k, self.t
        u = u[k, 1:-1] + self.dt/self.dx**2 * (u[k, :-2] - 2*u[k, 1:-1] + u[k, 2:])
        u = np.concatenate(([self.x0], u, [self.xL]))
        return u
