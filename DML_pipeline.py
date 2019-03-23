import argparse
from data_generation import generate_dataset
from run_DML import run_DML
from utils import str2bool
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def getArgs_data_generation(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--system_name', default='x', type=str, help='system name')
    parser.add_argument('--dataset_dir', default='x', type=str, help='directory to store generated data')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--T', default=10, type=int, help='simulation time')
    parser.add_argument('--L', default=100, type=int, help='number of steps in a trajectory')
    parser.add_argument('--n_envs', default=5, type=int, help='number of environments')
    parser.add_argument('--n_reps', default=3, type=int, help='number of repititions for each environment')
    parser.add_argument('--obs_noise_std', default=0.01, type=float, help='standard deviation of observation noise')
    parser.add_argument('--X2D_obs_noise_std', default=0.01, type=float, help='standard deviation of observation noise from X to D (only used in DML)')
    return parser.parse_args(argv)

input_args_str = "--system_name mass_field\
--T 10\
--L 500\
--n_envs 64\
--n_reps 2\
--obs_noise_std 0.00\
--X2D_obs_noise_std 0.00\
--dataset_dir datasets\
--batch_size 12"

input_args_temp = input_args_str.split("--")
input_args = []
for ind, twins in enumerate(input_args_temp[1:]):
    a, b = twins.split(" ")
    a = "--{}".format(a)
    input_args.append(a)
    input_args.append(b)
args_data_generation = getArgs_data_generation(input_args)



def getArgs_DML(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoother_backend', default='scikit', type=str, help='detemine the smooter backend: {scikit, pytorch}')
    parser.add_argument('--dynamics_backend', default='scikit', type=str, help='detemine the dynamics backend: {scikit, pytorch}')
    parser.add_argument('--dataset_filename', default='x', type=str, help='dataset filename')
    parser.add_argument('--dataset_dir', default='x', type=str, help='directory to the dataset')
    parser.add_argument('--n_envs_to_train', default=-1, type=int, help='how many environments are used for training(default:all)')
    parser.add_argument('--state_to_learn', default=1, type=int, help='which state of the dynamical system to learn(starting from 1)')
    parser.add_argument('--repetition_use', default='denoise', type=str, help='how to use the repititions in each environment {denoise,learn}')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--smoother_kernel', default='linear', type=str, help='kernel used for smoothing')
    parser.add_argument('--dynamics_kernel', default='linear', type=str, help='kernel used for dynamics')
    parser.add_argument('--reg_factor', default=0.01, type=float, help='regularisation coefficient for kernel ridge regression (smoother)')
    parser.add_argument('--smoother_alpha', default=0.01, type=float, help='condition number used by scikit learn (ignored by pytorch backend)')
    parser.add_argument('--dynamics_alpha', default=0.01, type=float, help='condition number used by scikit learn (ignored by pytorch backend)')
    parser.add_argument('--diff', default=1.0, type=float, help='the dt value to compute finite difference approximation to differentiation')
    parser.add_argument('--diff_method', default='finite_diff', type=str, help='method of differentiation for the smoother')
    parser.add_argument('--smoother_kernel_bandwidth', default=0.01, type=float, help='sigma (not sigma2) of the RBF, Laplace, Exponential kernels. Ignored by other kernels')
    parser.add_argument('--dynamics_kernel_bandwidth', default=0.01, type=float, help='sigma (not sigma2) of the RBF, Laplace, Exponential kernels. Ignored by other kernels')
    parser.add_argument('--smoother_kernel_degree', default=3, type=int, help='degree of the polynomial degree. Ignored by other kernels')
    parser.add_argument('--dynamics_kernel_degree', default=3, type=int, help='degree of the polynomial degree. Ignored by other kernels')
    parser.add_argument('--smoother_training_epochs', default=100, type=int, help='the number of epochs that the smoother is trained')
    parser.add_argument('--dynamics_training_epochs', default=100, type=int, help='the number of epochs that the dynamics is trained')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate for all learners')
    parser.add_argument('--centers_method', default='random', type=str, help='the method to choose centers for the kernel ridge regression(smoother){ranodm, all_points}')
    parser.add_argument('--ncenters', default=10, type=int, help='number of centers for the kernel ridge regression(smoother)')
    parser.add_argument('--verbose', default=False, type=str2bool, help='print out the state of the learners')
    parser.add_argument('--use_cuda', default=True, type=str2bool, help='Use GPU or CPU')
    return parser.parse_args(argv)

input_args_str = "\
--smoother_backend scikit\
--dynamics_backend scikit\
--dataset_filename mass_field\
--dataset_dir datasets\
--batch_size 64\
--reg_factor 5\
--smoother_alpha 1.0\
--dynamics_alpha 1.0\
--smoother_kernel rbf\
--dynamics_kernel rbf\
--smoother_kernel_bandwidth 1.0\
--dynamics_kernel_bandwidth 1.0\
--smoother_kernel_degree 7\
--dynamics_kernel_degree 7\
--smoother_training_epochs 100\
--dynamics_training_epochs 10\
--lr 0.001\
--centers_method all_points\
--ncenters 10\
--repetition_use denoise\
--verbose True\
--state_to_learn 0\
--diff_method finite_diff\
--diff 0.001\
--n_envs_to_train 20\
--use_cuda False"


input_args_temp = input_args_str.split("--")
input_args = []
for ind, twins in enumerate(input_args_temp[1:]):
    a, b = twins.split(" ")
    a = "--{}".format(a)
    input_args.append(a)
    input_args.append(b)
args_DML = getArgs_DML(input_args)


n_trials = 20
theta_est_dml = np.zeros(n_trials)
theta_est_ols = np.zeros(n_trials)
for k in range(n_trials):
    print("Trial: {}".format(k))
    generate_dataset(args_data_generation, save_data=True)
    theta_est_dml[k], theta_est_ols[k] = run_DML(args_DML)
    print("Estimated mass by DML is:{}".format(1./theta_est_dml[k]))
    print("Estimated mass by OLS is:{}".format(1./theta_est_ols[k]))

np.mean(theta_est_dml)
np.mean(theta_est_ols)





