# This file contains different regressors for approximating dynamics

# import torch
# from torch import nn
# import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from net_utils import KernelRegressorModel
import torch.utils.data as data_utils
import torch
from future.utils import with_metaclass
import numpy as np
from utils import compute_grad
from sklearn.kernel_ridge import KernelRidge
import copy
import statsmodels.api as sm # for OLS

__author__ = 'Arash Mehrjou'


class Dynamics_Approximator(with_metaclass(ABCMeta, object)):
    """Abstract class for approximating the dynamics. Inputs to all methods are Pytorch tensors."""

    @abstractmethod
    def eval(self, X):
        """
        Evaluate the Dynnamics approximator on locations X
        X: nx x dx where each row represents one point and dx is the input dimenssion
        return nx x dy where dy is the output dimensions
        Output is an approximation to dx/dt.
        """
        pass



class KernelRidge_Dynamics_Approximator(Dynamics_Approximator):
    """
    A regressor based on Dynamics approximator
    """
    def __init__(self, kernel, interpolants, t, approximator, args):
        """
        interpolant: an smooth function that inerpolate between the values of X.
        X: (nx, dy) The locations in which the dynamics approximator should be trained. (X is time here)
        """
        self.kernel = kernel
        self.interpolants = interpolants
        self.X, self.Y = KernelRidge_Dynamics_Approximator.compute_inputs_targets(self.interpolants, t, args)
        self.input_dim = self.interpolants[0].output_dim
        self.approximator = KernelRidge()
        self.alpha = args.dynamics_alpha
        self.degree = args.dynamics_kernel_degree
        self.gamma = 1./args.dynamics_kernel_bandwidth
        self.kr = approximator


    
    @classmethod
    def compute_inputs_targets(cls, interpolants, t, args):
        inputs = []
        targets = []
        for env_ind, interpolant in enumerate(interpolants):
            X = interpolant(t)
            Y = KernelRidge_Dynamics_Approximator.compute_targets(interpolant, t, args)
            inputs.append(X)
            targets.append(Y)
        inputs = np.concatenate(inputs, axis=0)
        targets = np.concatenate(targets, axis=0)
        return inputs, targets


    @classmethod
    def compute_targets(cls, interpolant, t, args):
        targets = compute_grad(interpolant, t, args.dynamics_backend, args)
        return targets


    def eval(self, X):
        Y_hat = self.kr.predict(X)
        return Y_hat

    def init_trainer(self, args):
        pass
        # self.kr = KernelRidge(kernel=self.kernel, alpha=self.alpha, gamma=self.gamma)


    def train(self, args):
        self.kr.fit(self.X, self.Y)

    def __call__(self, X):
        return self.eval(X)


    
class NeuralNet_Dynamics_Approximator(Dynamics_Approximator):
    """
    A regressor based on Dynamics approximator
    """
    def __init__(self, interpolants, t, approximator, device, args):
        """
        kernel: kernel used for smoothing
        interpolant: an smooth function that inerpolate between the values of X.
        X: (nx, dy) The locations in which the dynamics approximator should be trained. (X is time here)
        """
        self.device = device
        self.interpolants = interpolants
        self.X, self.Y = NeuralNet_Dynamics_Approximator.compute_inputs_targets(self.interpolants, t, args)
        self.X = torch.tensor(self.X, dtype=torch.float)
        self.Y = torch.tensor(self.Y, dtype=torch.float)
        self.input_dim = self.interpolants[0].output_dim
        self.net = approximator.to(device)




    @classmethod
    def compute_inputs_targets(cls, interpolants, t, args):
        inputs = []
        targets = []
        for env_ind, interpolant in enumerate(interpolants):
            X = interpolant(t)
            Y = NeuralNet_Dynamics_Approximator.compute_targets(interpolant, t, args)
            inputs.append(X)
            targets.append(Y)
        inputs = np.concatenate(inputs, axis=0)
        targets = np.concatenate(targets, axis=0)
        return inputs, targets


    
    @classmethod
    def compute_targets(cls, interpolant, t, args):
        targets = compute_grad(interpolant, t, args.dynamics_backend, args)
        return targets


    def eval(self, X):
        """
        This function takes state and outputs the dynamics vector of that location
        X: input data for test
        output: dx/dt
        """
        Xdot = self.net(X)
        return Xdot

    def init_trainer(self, args):
        """
        This function initializes the parameters of the trainer.
        """
        self.batch_size = args.batch_size
        dataset = data_utils.TensorDataset(self.X, self.Y) # here we don't need labels since X is just time
        self.dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True) # create your dataloader
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.lr) #Stochastic Gradient Descent


    def train(self, args):
        """
        This function trains the weights of the dynamics approximator for n_epochs number of epochs
        n_epochs: number of epochs to train the kernel smoother
        output: updated trained version of self
        """
        for epoch in range(args.dynamics_training_epochs):
            avg_loss = 0
            total_batch = self.X.shape[0] // self.batch_size
            for i, (inputs, targets) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                outputs = self.net(inputs.to(self.device))
                loss = self.criterion(outputs, targets.to(self.device))
                loss.backward()# back props
                self.optimizer.step()# update the parameters
                avg_loss += loss / total_batch
            if args.verbose:
                print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_loss.cpu().item()))
        if args.verbose:
            print('Learning Finished!')

####DML Regressor


class OLS_Dynamics_Approximator():

    def __init__(self, interpolants, t, fe, y_ind, device, args):
        self.device = device
        self.fe = fe
        self.interpolants = interpolants
        self.states, self.statesdot, self.D= DML_Dynamics_Approximator.compute_inputs_targets(self.interpolants, t, self.fe, args)
        self.Y = self.statesdot[:, y_ind].reshape(-1, 1)
        self.X = self.states

    def run_ols_estimator(self, args):
        theta_est = self.estimate_theta_OLS(self.D, self.Y)
        return theta_est

    def estimate_theta_OLS(self, D, Y):
        self.OLS = sm.OLS(Y,D)
        results = self.OLS.fit()
        theta_est = results.params[0]
        return theta_est




class DML_Dynamics_Approximator():
    """
    In DML literature we have
    # Y = theta * D + g(X) + U,     E[U | X, D] = 0
    # D = m(X) + V,                 E[V | X] = 0
    and we are interested in estimating theta while D, X and Y are measurements.
    """
    def __init__(self, interpolants, t, g_approximator, m_approximator, fe, y_ind, device, sample_split, args):
        """
        g_approximator: The neural network used to approximate g which is mapping from X to Y
        m_approximator: The neural network used to approximate m which is mapping from X to D
        y_ind : The index of the state whose time derivative is taken as Y
        fe: defines the generative model by which D is produced from X (D = fe(X))
        interpolants: A list consisting of interpolants for different environments corresponding to different states.
        t: The time array on which the interpolants are estimated.
        sample_split: Determine if we split the samples {True, False}
        """
        # Creating the dataset from iterpolants
        self.device = device
        self.batch_size = args.batch_size
        self.interpolants = interpolants
        self.fe = fe
        self.states, self.statesdot, self.D= DML_Dynamics_Approximator.compute_inputs_targets(self.interpolants, t, self.fe, args)
        self.Y = self.statesdot[:, y_ind].reshape(-1, 1)
        self.X = self.states
        self.X_tensor = torch.tensor(self.X, dtype=torch.float)
        self.Y_tensor = torch.tensor(self.Y, dtype=torch.float)
        self.D_tensor = torch.tensor(self.D, dtype=torch.float)
        self.input_dim = self.interpolants[0][0].output_dim
        self.sample_split = sample_split
        if self.sample_split:
            # split dataset
            nsamples = self.X.shape[0]
            set_a_inds = np.random.choice(range(nsamples), nsamples//2, replace=False)
            set_b_inds = np.setdiff1d(range(nsamples), set_a_inds)
            self.X_a, self.X_b = self.X[set_a_inds], self.X[set_b_inds]
            self.X_tensor_a, self.X_tensor_b = self.X_tensor[set_a_inds], self.X_tensor[set_b_inds]
            self.Y_a, self.Y_b = self.Y[set_a_inds], self.Y[set_b_inds]
            self.Y_tensor_a, self.Y_tensor_b = self.Y_tensor[set_a_inds], self.Y_tensor[set_b_inds]
            self.D_a, self.D_b = self.D[set_a_inds], self.D[set_b_inds]
            self.D_tensor_a, self.D_tensor_b = self.D_tensor[set_a_inds], self.D_tensor[set_b_inds]
            self.X2D_approximator_a = m_approximator
            self.X2D_approximator_b = copy.deepcopy(m_approximator)
            self.X2Y_approximator_a = g_approximator
            self.X2Y_approximator_b = copy.deepcopy(g_approximator)

        if args.dynamics_backend == 'pytorch':             
            # Creating the regressors
            self.X2D_approximator = m_approximator.to(device) # (input_dim, output_dim) m(x) in DML literature
            # self.D2Y = KernelRidge(kernel='linear', alpha=0.1, gamma=1)
            self.X2Y_approximator = g_approximator.to(device) # (input_dim, output_dim) g(x) in DML literature
        elif args.dynamics_backend == 'scikit':
            self.X2D_approximator = m_approximator
            self.X2Y_approximator = g_approximator

            


        


        
    @classmethod
    def compute_inputs_targets(cls, interpolants, t, fe, args):
        states_data = []# [x1 x2]
        statesdot_data = [] # [x1dot, x2dot] 
        for state_ind in range(len(interpolants)):
            inputs = []
            targets = []
            for env_ind, interpolant in enumerate(interpolants[state_ind]):
                X = interpolant(t)
                Y = compute_grad(interpolant, t, args.dynamics_backend, args)
                inputs.append(X)
                targets.append(Y)
            states_data.append(np.concatenate(inputs, axis=0))
            statesdot_data.append(np.concatenate(targets, axis=0))
        # compute D
        X = np.concatenate(states_data, axis=1)
        Y = np.concatenate(statesdot_data, axis=1)
        D = np.apply_along_axis(fe, 1, X).reshape(X.shape[0], -1)
        return  X, Y, D 


    def run_naive_DML(self, args):
        theta_est = self.estimate_theta_naiveDML(self.X, self.Y, self.D, args)
        return theta_est

    def run_crossfitting_DML(self, args):
        theta_est = self.estimate_theta_crossfittingDML(self.X_a, self.Y_a, self.D_a, self.X_b, self.Y_b, self.D_b, args)
        return theta_est
    

    # def eval(self, X):
    #     """
    #     This function takes state and outputs the dynamics vector of that location
    #     X: input data for test
    #     output: dx/dt
    #     """
    #     Xdot = self.net(X)
    #     return Xdot


    def estimate_theta_naiveDML(self, X, Y, D, args):
        """
        This function takes X, Y, D as the components of DML and estimates theta
        """
        self.X2Y_approximator.fit(X,Y)
        Ghat = self.X2Y_approximator.predict(X)
        self.X2D_approximator.fit(X,D)
        Mhat = self.X2D_approximator.predict(X)
        Vhat = D - Mhat
        theta_est = np.mean(np.dot(Vhat,Y-Ghat)) / np.mean(np.dot(Vhat,D))
        return theta_est

    
    def estimate_theta_crossfittingDML(self, Xa, Ya, Da, Xb, Yb, Db, args):
        """
        This function takes two distinct splits of samples X, Y, D and gives a cross-fitting estimate of theta
        """
        self.X2Y_approximator_a.fit(Xa, Ya)
        self.X2Y_approximator_b.fit(Xb, Yb)
        self.X2D_approximator_a.fit(Xa, Ya)
        self.X2D_approximator_b.fit(Xb, Yb)
        Ghat_1 = self.X2Y_approximator_a.predict(Xb)
        Ghat_2 = self.X2Y_approximator_b.predict(Xa)
        Mhat_1 = self.X2D_approximator_a.predict(Xb)
        Mhat_2 = self.X2D_approximator_b.predict(Xa)
        Vhat_1 = Db - Mhat_1
        Vhat_2 = Da - Mhat_2
        theta_1 =  np.mean(np.dot(Vhat_1,(Yb - Ghat_1))) / np.mean(np.dot(Vhat_1, Db))
        theta_2 =  np.mean(np.dot(Vhat_2,(Ya - Ghat_2))) / np.mean(np.dot(Vhat_2, Da))
        theta_est = 0.5 * (theta_1 + theta_2)
        return theta_est

        


    # def estimate_theta_samplesplitDML(self, Xa, Ya, Da, Xb, Yb, Db, args):



    # def init_trainers(self, args):
    #     """
    #     This function initializes the parameters of the trainer.
    #     """
    #     # init states_to_D_approximator
    #     lr_X2D = args.lr
    #     batch_size = args.batch_size
    #     dataset_X2D_a = data_utils.TensorDataset(self.X_tensor_a, self.D_tensor_a) # here we don't need labels since X is just time
    #     dataset_X2D_b = data_utils.TensorDataset(self.X_tensor_b, self.D_tensor_b) # here we don't need labels since X is just time
    #     self.dataloader_X2D_a = data_utils.DataLoader(dataset_X2D_a, batch_size=batch_size, shuffle=True) # create your dataloader
    #     self.dataloader_X2D_b = data_utils.DataLoader(dataset_X2D_b, batch_size=batch_size, shuffle=True) # create your dataloader
    #     self.criterion_X2D = torch.nn.MSELoss()
    #     self.optimizer_X2D = torch.optim.SGD(self.X2D_approximator.parameters(), lr=lr_X2D) #Stochastic Gradient Descent

    #     lr_X2Y = args.lr
    #     batch_size = args.batch_size
    #     dataset_X2Y_a = data_utils.TensorDataset(self.X_tensor_a, self.Y_tensor_a) # here we don't need labels since X is just time
    #     dataset_X2Y_b = data_utils.TensorDataset(self.X_tensor_b, self.Y_tensor_b) # here we don't need labels since X is just time
    #     self.dataloader_X2Y_a = data_utils.DataLoader(dataset_X2Y_a, batch_size=batch_size, shuffle=True) # create your dataloader
    #     self.dataloader_X2Y_b = data_utils.DataLoader(dataset_X2Y_b, batch_size=batch_size, shuffle=True) # create your dataloader
    #     self.criterion_X2Y = torch.nn.MSELoss()
    #     self.optimizer_X2Y = torch.optim.SGD(self.X2Y_approximator.parameters(), lr=lr_X2Y) #Stochastic Gradient Descent
    #     print("DML regressors initialized.")

    # def train(self, args):
    #     """
    #     This function trains the weights of the dynamics approximators for n_epochs number of epochs
    #     n_epochs: number of epochs to train the kernel smoother
    #     output: updated trained version of self
    #     """
    #     nepochs_X2D = args.dynamics_training_epochs
    #     nepochs_X2Y = args.dynamics_training_epochs

    #     print('Training m(X) -> D')
    #     for epoch in range(nepochs_X2D):
    #         avg_loss = 0
    #         total_batch = self.X.shape[0] // self.batch_size
    #         for i, (inputs, targets) in enumerate(self.dataloader_X2D_a):
    #             self.optimizer_X2D.zero_grad()
    #             outputs = self.X2D_approximator(inputs.to(self.device))
    #             loss = self.criterion_X2D(outputs, targets.to(self.device))
    #             loss.backward()# back props
    #             self.optimizer_X2D.step()# update the parameters
    #             avg_loss += loss / total_batch
    #         if args.verbose:
    #             print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_loss.cpu().detach().numpy()))
    #     if args.verbose:
    #         print('Learning m(X) -> D finished!')


        

    #     print('Training g(X) -> Y')
    #     for epoch in range(nepochs_X2Y):
    #         avg_loss = 0
    #         total_batch = self.X.shape[0] // self.batch_size
    #         for i, (inputs, targets) in enumerate(self.dataloader_X2Y_a):
    #             self.optimizer_X2Y.zero_grad()
    #             outputs = self.X2Y_approximator(inputs.to(self.device))
    #             loss = self.criterion_X2Y(outputs, targets.to(self.device))
    #             loss.backward()# back props
    #             self.optimizer_X2Y.step()# update the parameters
    #             avg_loss += loss / total_batch
    #         if args.verbose:
    #             print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_loss.cpu().detach().numpy()))
    #     if args.verbose:
    #         print('Learning g(X) -> Y finished!')

        



    #     print('Training theta * D -> Y')
    #     self.D2Y.fit(self.D, self.Y)
    #     print('Learning theta * D -> Y finished!')


    def compute_theta(self):
        self.X2D_approximator_cpu  = self.X2D_approximator.to(torch.device('cpu'))
        self.X2Y_approximator_cpu = self.X2Y_approximator.to(torch.device('cpu'))
        Vhat_a = (self.D_tensor_a - self.X2D_approximator_cpu(self.X_tensor_a)).detach().numpy()
        D_b = self.D_b
        Yhat_a = self.X2Y_approximator_cpu(self.X_tensor_a).detach().numpy()
        Dhat_b = Yhat_a - self.X2Y_approximator_cpu(self.X_tensor_b).detach().numpy()
        theta_a = (1.0 / (np.sum(Vhat_a * D_b) / Vhat_a.shape[0])) * (np.sum(Vhat_a * Dhat_b) / Vhat_a.shape[0])
        # print("mass = {}".format(1.0 / theta_a))
        return theta_a


