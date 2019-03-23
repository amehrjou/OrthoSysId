# This module contain necessary classes and functions to generate trajectories from the dynamcis
import numpy as np
import copy
__author__ = 'Arash Mehrjou'

class Trajectory_Simulator(object):
    class noise_generator():
        def __init__(self, dim, std):
            self.dim = dim
            self.std = std
        
        def sample(self, nsamples):
            data = np.zeros((nsamples, self.dim))
            for i in range(nsamples):
                for d in range(self.dim):
                    data[i, d] = np.random.normal(0, self.std)
            return np.squeeze(data)
    
    def __init__(self, dynamics, dt, noise_var):        
        self.dim = dynamics.dim
        self.f = dynamics.f
        self.dt = dt
        self.noise = self.noise_generator(self.dim , noise_var)
        
    def sample(self, x0, nsteps):
        self.data = np.zeros((nsteps, self.dim))
        x0 = np.atleast_2d(copy.deepcopy(x0)).reshape(1, self.dim) # reshaping is necessary for dimensional consistency
        # self.data[0] = x0 + self.noise.sample(1)
        x_current = x0
        for t in range(nsteps):
            self.data[t] = x_current + self.noise.sample(1) # observation noise
            x_current = x_current + self.f(x_current) * self.dt
        return self.data

class Transit_Funct():
    # A wrapper around dynamics functions containing its function and the dymension
    def __init__(self, f, dim):
        self.f = f
        self.dim = dim
    