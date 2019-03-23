"""Module containing kernel related classes"""
from __future__ import division

from builtins import str
from past.utils import old_div
from builtins import object
from future.utils import with_metaclass
__author__ = 'Arash Mehrjou'

from abc import ABCMeta, abstractmethod
# import autograd
# import autograd.numpy as np
import numpy as np
import torch

class Kernel(with_metaclass(ABCMeta, object)):
    """Abstract class for kernels. Inputs to all methods are numpy arrays."""

    def __call__(self, X, Y):
        return self.eval(X, Y)

    @abstractmethod
    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        X: nx x d where each row represents one point
        Y: ny x d
        return nx x ny Gram matrix
        """
        pass

    @abstractmethod
    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...
        
        X: n x d where each row represents one point
        Y: n x d
        return a 1d numpy array of length n.
        """
        pass


    def get_feature_map(self):
        """
        Return the underlying feature map (an instance of FeatureMap) of the
        kernel.  Return None if a closed-form feature map is not available
        e.g., the output of the underlying feature map is infinite-dimensional.
        """
        return None

    def feature_map_available(self):
        """
        Return True if an explicit feature map is available.
        """
        return self.get_feature_map() is not None

# end Kernel

class FeatureMap(with_metaclass(ABCMeta, object)):
    """
    Abstract class for a feature map of a kernel.
    """

    @abstractmethod
    def output_shape(self):
        """
        Return the output shape of this feature map.
        """
        raise NotImplementedError()

    @abstractmethod
    def input_shape(self):
        """
        Return the expected input shape of this feature map.
        """
        raise NotImplementedError()

# end class FeatureMap


class PTKernel(Kernel):
    """
    An abstract class for a kernel for Pytorch.
    Subclasses implementing this should rely on only operations which are
    compatible with Pytorch.
    """
    pass

# end PTKernel

class PTKLinear(PTKernel):
    """
    Linear kernel. Pytorch implementation.
    """
    def __init__(self):
        pass

    def eval(self, X, Y):
        return X.mm(Y.t())

    def pair_eval(self, X, Y):
        return torch.sum(X*Y, 1)

# end class PTKLinear

class PTSumKernel(PTKernel):
    """
    A kernel given by summing PTKernel's.
    """
    def __init__(self, kernels, weights=None):
        """
        kernels: a list of PTKernel's
        weights: linear combination weights. If None, then use uniform weights.
        """
        self.kernels = kernels
        if weights is None:
            weights = torch.ones(len(kernels))/float(len(kernels))
        self.weights = weights

    def eval(self, X, Y):
        nx, ny = X.shape[0], Y.shape[0]
        kernels = self.kernels
        K = torch.zeros(nx, ny)
        for i, k in enumerate(kernels):
            K += self.weights[i]*k.eval(X, Y)
        return K

    def pair_eval(self, X, Y):
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same first dimension')
        m = X.shape[0]
        Kv = torch.zeros(m)
        for i, k in enumerate(self.kernels):
            Kv += self.weights[i]*k.pair_eval(X, Y)
        return Kv

# end class PTSumKernel


class PTKIMQ(PTKernel):
    """
    The inverse multiquadric (IMQ) kernel studied in 

    Measure Sample Quality with Kernels 
    Jackson Gorham, Lester Mackey

    k(x,y) = (c^2 + ||x-y||^2)^b 
    where c > 0 and b < 0. Following a theorem in the paper, this kernel is 
    convergence-determining only when -1 < b < 0. In the experiments, 
    the paper sets b = -1/2 and c = 1.
    """

    def __init__(self, b=-0.5, c=1.0):
        if not b < 0:
            raise ValueError('b has to be negative. Was {}'.format(b))
        if not c > 0:
            raise ValueError('c has to be positive. Was {}'.format(c))
        self.b = b
        self.c = c

    def eval(self, X, Y):
        b = self.b
        c = self.c
        sumx2 = torch.sum(X**2, 1).reshape(-1, 1)
        sumy2 = torch.sum(Y**2, 1).reshape(1, -1)
        D2 = sumx2 - 2.0*X.mm(Y.t()) + sumy2
        K = (c**2 + D2)**b
        return K

    def pair_eval(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        b = self.b
        c = self.c
        return (c**2 + torch.sum((X-Y)**2, 1))**b


# end class PTKIMQ

class PTKIMQGorhamMackey(PTKernel):
    """
    The inverse multiquadric (IMQ) kernel studied in 

    Measure Sample Quality with Kernels 
    Jackson Gorham, Lester Mackey

    k(x,y) = (1 + ||x-y||^2)^(-0.5) 
    """

    def __init__(self):
        pass

    def eval(self, X, Y):
        sumx2 = torch.sum(X**2, 1).reshape(-1, 1)
        sumy2 = torch.sum(Y**2, 1).reshape(1, -1)
        D2 = sumx2 - 2.0*X.mm(Y.t()) - sumy2
        K = 1.0/( torch.sqrt(1.0 + D2) )
        return K

    def pair_eval(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        D2 = torch.sum((X-Y)**2, 1)
        return 1.0/( torch.sqrt(1.0 + D2) )

# end class PTKIMQGorhamMackey


class PTExplicitKernel(PTKernel):
    """
    A class for kernel that is defined as 
        k(x,y) = <f(x), f(y)> 
    for a finite-output f (of type FeatureMap).
    """
    def __init__(self, fm):
        """
        fm: a FeatureMap parameterizing the kernel. This feature map is
            expected to take in a Pytorch tensor as the input.
        """
        self.fm = fm

    def eval(self, X, Y):
        """
        Evaluate the kernel on Pytorch tensors X and Y
        X: nx x d where each row represents one point
        Y: ny x d
        return nx x ny Gram matrix
        """
        f = self.fm
        FX = f(X)
        FY = f(Y)
        K = FX.mm(FY.t())
        return K

    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...
        
        X: n x d where each row represents one point
        Y: n x d
        return a 1d Pytorch array of length n.
        """
        f = self.fm
        FX = f(X)
        FY = f(Y)
        vec = torch.sum(FX*FY, 1)
        return vec


    def get_feature_map(self):
        return self.fm

# end PTExplicitKernel

class PTKModuleCompose(PTKernel):
    """
    A kernel given by k'(x,y) = k(f(x), f(y)), where f is a torch.nn.Module and
    k is the specified kernel.
    """
    def __init__(self, k, f):
        """
        k: a PTKernel
        f: a torch.nn.Module representing the function to transform the inputs
        """
        self.k = k
        self.f = f

    def eval(self, X, Y):
        f = self.f
        k = self.k
        fx = f.forward(X)
        fy = f.forward(Y)
        return k.eval(fx, fy)

    def pair_eval(self, X, Y):
        f = self.f
        k = self.k
        fx = f.forward(X)
        fy = f.forward(Y)
        return k.pair_eval(fx, fy)

# end class PTKFuncCompose

class PTKGauss(PTKernel):
    """
    Pytorch implementation of the isotropic Gaussian kernel.
    Parameterization is the same as in the density of the standard normal
    distribution. sigma2 is analogous to the variance.
    """
    def __init__(self, sigma2):
        assert sigma2 > 0, 'sigma2 must be > 0. Was {}'.format(sigma2)
        self.sigma2 = sigma2

    def eval(self, X, Y):
        """
        Evaluate the Gaussian kernel on the two two-dimensional Pytorch tensors
        X, Y.

        * X: n1 x d Pytorch
        * Y: n2 x d Pytorch

        Return
        ------
        K: an n1 x n2 Gram matrix.
        """
        sumx2 = torch.sum(X**2, 1).reshape(-1, 1)
        sumy2 = torch.sum(Y**2, 1).reshape(1, -1)
        D2 = sumx2 - 2.0*X.mm(Y.t()) + sumy2
        K = torch.exp(-D2/(2.0*self.sigma2))
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d Pytorch tensors

        Return
        -------
        a Torch tensor with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        D2 = torch.sum( (X-Y)**2, 1)
        Kvec = torch.exp(old_div(-D2,(2.0*self.sigma2)))
        return Kvec

# end class PTKGauss


class KGauss(Kernel):
    """
    The standard isotropic Gaussian kernel.
    Parameterization is the same as in the density of the standard normal
    distribution. sigma2 is analogous to the variance.
    """

    def __init__(self, sigma2):
        assert sigma2 > 0, 'sigma2 must be > 0. Was %s'%str(sigma2)
        self.sigma2 = sigma2

    def eval(self, X, Y):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X : n1 x d numpy array
        Y : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        #(n1, d1) = X.shape
        #(n2, d2) = Y.shape
        #assert d1==d2, 'Dimensions of the two inputs must be the same'
        sumx2 = np.reshape(np.sum(X**2, 1), (-1, 1))
        sumy2 = np.reshape(np.sum(Y**2, 1), (1, -1))
        D2 = sumx2 - 2*np.dot(X, Y.T) + sumy2
        K = np.exp(old_div(-D2,(2.0*self.sigma2)))
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d numpy array

        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        D2 = np.sum( (X-Y)**2, 1)
        Kvec = np.exp(old_div(-D2,(2.0*self.sigma2)))
        return Kvec

    def __str__(self):
        return "KGauss(%.3f)"%self.sigma2

# end class KGauss

class PTKPoly(PTKernel):
    """
    Pytorch implementation of the polynomial kernel.
    """
    def __init__(self, degree):
        assert degree > 0, 'degree must be > 0. Was {}'.format(sigma2)
        self.degree = degree

    def eval(self, X, Y):
        """
        Evaluate the Polynomial kernel on the two 2d numpy arrays X, Y.

        * X: n1 x d Pytorch
        * Y: n2 x d Pytorch

        Return
        ------
        K: an n1 x n2 Gram matrix.
        """
        M = 1.0 + X.mm(Y.t())
        K = M.pow(self.degree)
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d Pytorch tensors

        Return
        -------
        a Torch tensor with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        M = 1.0 + torch.sum(X * Y, dim=1) # X * Y is element-wise product
        Kvec = M.pow(self.degree)
        return Kvec

# end class PTKPoly

class KPoly(Kernel):
    """
    The polynomial kernel.
    """

    def __init__(self, degree):
        assert degree > 0, 'degree of the polynomial must be > 0. Was %s'%str(sigma2)
        self.degree = degree

    def eval(self, X, Y):
        """
        Evaluate the Polynomial kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X : n1 x d numpy array
        Y : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        #(n1, d1) = X.shape
        #(n2, d2) = Y.shape
        #assert d1==d2, 'Dimensions of the two inputs must be the same'
        M = 1.0 + np.matmul(X, Y.transpose())
        K = np.power(M, self.degree)
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d numpy array

        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        M = 1.0 + np.sum(np.multiply(X, Y), axis=1)
        Kvec = np.power(M, self.degree)
        return Kvec

    def __str__(self):
        return "Polynomial(degree=%.3f)"%self.degree

# end class KPoly

