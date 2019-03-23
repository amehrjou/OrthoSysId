# This script is supposed to generate a dataset of trajectroeis and store it
import numpy as np
import seaborn as sns
from trajectory_utils import Trajectory_Simulator
from trajectory_utils import Transit_Funct
import matplotlib.pyplot as plt
import dynamics
from matplotlib import rc
rc('text', usetex=True)
import os
import pickle
import argparse
__author__ = 'Arash Mehrjou'


class dynamical_system_simulator():
    def __init__(self, system_name, dynamics, n_states, T, L):
        self.system_name = system_name
        self.f = dynamics
        self.T = T
        self.L = L
        self.dt = T / (L - 1)
        self.n_states = n_states

    def simulate(self, x0, obs_noise_std):
        # simulate for T steps
        self.noise_std = obs_noise_std
        traj = Trajectory_Simulator(Transit_Funct(self.f, self.n_states), self.dt, noise_var=self.noise_std)
        return traj.sample(x0, self.L)


def generate_trajectories(envs, args):
    ## generate trajectories for the provided system and the list of environments
    
    #Inputs
    #envs: a list whose elements are different environment (e.g. initial points)
    #system name: the name of the system for which the trajectories are requested
    
    #Outputs
    # data: is a dictionary whose "envs" entry gives a list of environment and the
    # "trajectories" gives a list of size n_envs whose every element is a list of size n_reps whose every element is 
    # of size L x  n_states of repititins of trajectories for a particular starting point
    trajectories = []
    if args.system_name.lower() == 'voltrra':
        simulator = dynamical_system_simulator('voltrra', dynamics.Lotka_Voltrra, 2, args.T, args.L)
    elif args.system_name.lower() == 'bioreactor':
        simulator = dynamical_system_simulator('bioreactor', dynamics.bioreactor, 11, args.T, args.L)
    elif args.system_name.lower() == 'pendulums':
        simulator = dynamical_system_simulator('pendulums', dynamics.concatenated_pendulums, 6, args.T, args.L)
    elif args.system_name.lower() == 'tanh_system':
        simulator = dynamical_system_simulator('tanh_system', dynamics.tanh_system, 1, args.T, args.L)
    elif args.system_name.lower() == 'sine_system':
        simulator = dynamical_system_simulator('sine_system', dynamics.sine_system, 1, args.T, args.L)
    elif args.system_name.lower() == 'mass_field':
        simulator = dynamical_system_simulator('mass_field', dynamics.mass_field, 2, args.T, args.L)
    data = {"envs":envs, "trajectories":[]}
    x0 = envs # in this case environments are initial states
    trajectories = []
    for k in range(args.n_envs):
        trajs_per_env = []
        for r in range(args.n_reps):
            trajs_per_env.append(simulator.simulate(x0[k], args.obs_noise_std))
        trajectories.append(trajs_per_env)
    data['trajectories'] = trajectories
    data['meta_info'] = {'system_name':args.system_name , 'T':args.T, 'L':args.L, 'obs_noise_std':args.obs_noise_std}
    return data

def generate_environments(system_name, n_envs):
    ## generate n_envs environments (e.g. initial states) for the dynamical system system_name
    # output : a list whose every item is an array: e.g. a list of n_envs x n_states array 
    # when every environment corresponds to a new initial point.
    envs = []
    if system_name.lower() == 'voltrra':
        for i in range(n_envs):
            # envs.append(np.array([5., 3.]).reshape(-1, 1))
            x1_0 = np.random.uniform(1,5)
            x2_0 = np.random.uniform(1,5)
            envs.append(np.array([x1_0, x2_0]).reshape(-1, 2))
    elif system_name.lower() == 'pendulums':
        for i in range(n_envs):
            ## Pendulum system
            x0 = np.array([  np.random.uniform(-np.pi/4, np.pi/4), 
            0, 
            np.random.uniform(-np.pi/4, np.pi/4), 
            0, 
            np.random.uniform(-np.pi/4, np.pi/4), 
            0]).reshape(1, -1)  # Set angle of each pendulum as uniformly random in [-pi/4, pi/4] 
                # and set angular velocity of each pendulum to 0
            envs.append(x0)
    elif system_name.lower() == 'bioreactor':
        states_dict = {'Glu':1,
            'Fru':2,
            'Formic Acid':3,
            'Triose':4,
            'Acetic Acid':5,
            'Cn':6,
            'Amadori':7,
            'AMP':8,
            'C5':9,
            'lysR':10,
            'Melanoidin':11}
        FormicAcid0 = 0
        Triose0 = 0
        AceticAcid0 = 0 
        Cn0 = 0
        Amadori0 = 0
        AMP0 = 0
        C50 = 0
        lysR0 = np.random.uniform(0, 5.15)
        Melanoidin0 =  0
        for i in range(n_envs):
            Glu0 = np.random.uniform(0, 5.160)
            Fru0 = np.random.uniform(0, 5.160)
            x0 = np.array(list([Glu0, Fru0, FormicAcid0, Triose0, AceticAcid0, Cn0, Amadori0, AMP0, C50, lysR0, Melanoidin0])).reshape(1, -1)
            envs.append(x0)

    elif system_name.lower() == 'tanh_system':
        for i in range(n_envs):
            x0 = np.array([np.random.uniform(-10, 10)])
            envs.append(x0)

    elif system_name.lower() == 'sine_system':
        for i in range(n_envs):
            x0 = np.array([np.random.uniform(-5, 5)])
            envs.append(x0)
    
    elif system_name.lower() == 'mass_field':
        for i in range(n_envs):
            x1_0 = np.random.uniform(-1, 1)
            x2_0 = np.random.uniform(-1, 1)
            envs.append(np.array([x1_0, x2_0]).reshape(-1, 2))

    return envs

def generate_dataset(args, save_data=True):
    envs = generate_environments(args.system_name, args.n_envs)
    data = generate_trajectories(envs, args)
    if save_data:
        dataset_file_path = args.dataset_dir + '/' + args.system_name + '.pkl'
        pickle_file = open(dataset_file_path,"wb")
        pickle.dump(data, pickle_file)
        pickle_file.close()
    return data
    
    



