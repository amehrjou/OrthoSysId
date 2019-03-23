import argparse
from data_generation import generate_dataset
import pickle
import seaborn as sns
__author__ = 'Arash Mehrjou'

## Test with arguments
def getArgs(argv=None):
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
args = getArgs(input_args)

generate_dataset(args, save_data=True)


# load and plot datasets
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
dataset_file_path = 'datasets/mass_field.pkl'
pickle_file = open(dataset_file_path,"rb")
dataset = pickle.load(pickle_file)
pickle_file.close()
envs, trajectories, meta_info = dataset['envs'], dataset['trajectories'], dataset['meta_info']
env_ind = 0
state_ind = 0
n_reps = len(trajectories[0])
# for r in range(n_reps):
#     plt.plot(trajectories[env_ind][r][:, state_ind])
# plt.show()
