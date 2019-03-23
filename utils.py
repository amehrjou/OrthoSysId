import torch
import sys
import numpy as np
__author__ = 'Arash Mehrjou'


def compute_grad(f, x, backend, args):
    """
    Computes df/dx at locations x
    f: function
    x: (nx, dx)
    method: method to compute grads {'exact', 'finite_diff'}
    """
    
    method = args.diff
    if backend == 'pytorch':
        if args.diff_method == 'exact':
            y = torch.zeros(x.shape)
            for i, element in enumerate(x):
                if element.requires_grad == False:
                    element.requires_grad = True
                    element = element.reshape(-1,1)
                y[i] = torch.autograd.grad(f(element), element)[0]
        elif args.diff_method=='finite_diff':
            y = (f(x + args.diff) - f(x)) / args.diff
    elif backend == 'scikit':
        y = (f(x + args.diff) - f(x)) / args.diff
    
    return y

def print_no_newline(string): 
    """Print with replacement without going to the new line
    Useful for showing the progress of training or search
    """
    sys.stdout.write(string)
    sys.stdout.flush()

def compute_nrows_ncolumns(nplots):
    """
    Takes the total number of plots and calculate the number
    of rows and columns.
    """
    n_rows = int(np.sqrt(nplots)) + (np.sqrt(nplots) != int(np.sqrt(nplots))) * 1
    n_columns = int(nplots / n_rows) + (nplots / n_rows != int(nplots / n_rows)) * 1
    return n_rows, n_columns

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')