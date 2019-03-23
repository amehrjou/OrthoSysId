import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
__author__ = 'Arash Mehrjou'

class KernelRegressorModel(nn.Module):
    def __init__(self, centers, kernel, input_dim, output_dim):
        super(KernelRegressorModel, self).__init__() 
        self.k = kernel
        self.centers = centers
        self.output_dim, self.input_dim = input_dim, output_dim
        self.n_centers = self.centers.shape[0]
        self.W = nn.Parameter(torch.tensor(np.random.normal(0, 1, (output_dim, self.n_centers))).float())

        
    def forward(self, x):
        # x: (n data points, in_dim)
        Gram_matrix = self.k(x, self.centers)
        y = F.linear(self.W, Gram_matrix).t()
        return y


class LinearRegressionModel(nn.Module):
    # The learns the linear part of the rhs
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__() 
#         self.W = nn.Parameter(torch.zeros(input_dim, output_dim, dtype=torch.float))
        self.W = nn.Parameter(torch.tensor(np.random.normal(0, 1, (output_dim, input_dim))).float())
#         self.b = torch.zeros(input_dim, dtype=torch.float)

    def forward(self, x):
        out = F.linear(x, self.W)
        return out
    
class NonLinearRegressionModel(nn.Module):
    # The nn that learns the RHS (dynamics)
    def __init__(self, input_dim, output_dim):
        super(NonLinearRegressionModel, self).__init__() 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dh1 = 2 ** 7
        self.dh2 = 2 ** 5
        self.dh3 = 2 ** 7
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.dh1),
            nn.Tanh(),
            nn.Linear(self.dh1, self.dh2),
            nn.Tanh(),
            nn.Linear(self.dh2, self.dh3),
            nn.Tanh(),
            nn.Linear(self.dh3, self.output_dim),  
        )
        
    def forward(self, x):
        out = self.net(x)
        return out
        