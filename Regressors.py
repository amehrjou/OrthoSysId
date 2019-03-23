# This file contains different regressors for smoothing the trajectories

# import torch
# from torch import nn
# import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from net_utils import KernelRegressorModel
import torch.utils.data as data_utils
import torch
from future.utils import with_metaclass
import numpy as np
from sklearn.kernel_ridge import KernelRidge
__author__ = 'Arash Mehrjou'


class Smoother(with_metaclass(ABCMeta, object)):
    """Abstract class for trajectory smoothers. Inputs to all methods are Pytorch tensors."""

    @abstractmethod
    def eval(self, X):
        """
        Evaluate the smoother on locations X
        X: nx x dx where each row represents one point and dx is the input dimenssion
        return nx x dy where dy is the output dimensions
        """
        pass



class KernelSmootherScikit(Smoother):
    """
    A smoother based on kernel regression implemented in Scikit-learn
    """
    def __init__(self, kernel, X, Y, alpha, degree=3, gamma=None):
        """
        kernel: kernel used for smoothing (as string: {'linear', 'polynomial'})
        X: (nx, dx) input data (time for ODEs)
        Y: (ny, dy) output data to be smoothed (state values for ODEs)
        alpha: Small positive values of alpha improve the conditioning of the problem and 
        reduce the variance of the estimates
        degree: Degree of the polynomial kernel. Ignored by the other kernels.
        """
        self.kernel = kernel
        self.X = X
        self.Y = Y
        self.input_dim, self.output_dim = self.X.shape[1], self.Y.shape[1]
        self.alpha = alpha
        self.degree = degree
        self.gamma = gamma
    
    def init_trainer(self, args):
        self.kr = KernelRidge(kernel=self.kernel, alpha=self.alpha, gamma=self.gamma)

    def train(self, args):
        self.kr.fit(self.X, self.Y)

    def __call__(self, X):
        return self.eval(X)

    def eval(self, X):
        Y_hat = self.kr.predict(X)
        return Y_hat
        
class KernelSmoother(Smoother):
    """
    A smoother based on kernel regression implemented in PyTorch
    """
    def __init__(self, kernel, X, Y, regualirization_factor, center_method='all_points', ncenters=None):
        """
        kernel: kernel used for smoothing (as kernel class)
        X: (nx, dx) input data (time for ODEs)
        Y: (ny, dy) output data to be smoothed (state values for ODEs)
        """
        self.kernel = kernel
        self.X = X
        self.Y = Y
        self.input_dim, self.output_dim = self.X.shape[1], self.Y.shape[1]
        self.centers = self.find_centers(self.X, method=center_method, ncenters=ncenters)

        self.kr = KernelRegressorModel(self.centers, self.kernel, self.input_dim, self.output_dim)
        self.reg_factor = regualirization_factor

    def find_centers(self, X, method='all_points', ncenters=None):
        """
        This function takes obserevd input data and determine the location of kernels
        X: input data
        output: n_centers x dx
        """
        if method == "all_points":
            centers = X
        elif method == "random":
            inds = np.random.choice(range(len(X)), ncenters, replace=False)
            centers = X[inds]
        return centers

    def eval(self, X):
        """
        This function takes obserevd input data and determine the location of kernels
        X: input data for test
        output: interpolated values at X locations
        """
        Y_hat = self.kr(X)
        return Y_hat

    def init_trainer(self, args):
        """
        This function initializes the parameters of the trainer.
        """
        self.batch_size = args.batch_size
        dataset = data_utils.TensorDataset(self.X, self.Y) # create your datset
        self.dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True) # create your dataloader
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.kr.parameters(), lr=args.lr) #Stochastic Gradient Descent

    def train(self, args):
        """
        This function trains the weights of the kernel smoother for n_epochs number of epochs
        n_epochs: number of epochs to train the kernel smoother
        output: updated trained version of self
        """
        for epoch in range(args.smoother_training_epochs):
            avg_loss = 0
            total_batch = self.X.shape[0] // self.batch_size
            for i, (inputs, targets) in enumerate(self.dataloader):
                # print("Weights:")
                # print(self.kr.W)
                # print("centers:")
                # print(self.centers)
                self.optimizer.zero_grad()
                outputs = self.kr(inputs)
                loss = self.criterion(outputs, targets) + self.reg_factor * torch.norm(list(self.kr.parameters())[0], 2)
                loss = torch.log(loss)
                # print(loss)
                loss.backward()# back props
                self.optimizer.step()# update the parameters
                avg_loss += loss / total_batch
            if args.verbose:
                print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_loss.item()))
        if args.verbose:
            print('Learning Finished!')

    def __call__(self, X):
        return self.eval(X)






        
    


    
        


