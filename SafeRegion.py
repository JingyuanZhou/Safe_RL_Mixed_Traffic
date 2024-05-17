import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import gym
from ppo_agent import PPOAgent
from platoon_env import PlatoonEnv
import pandas as pd
import argparse
from tqdm import tqdm
import cv2 as cv
from scipy.interpolate import interp1d
plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['mathtext.default'] = 'regular'

def cal_SafeRegion(time_range, acceleration_range, step, mode, safety_layer_enabled, save_region = False):
    """
    :param model: the trained model
    :param env: the environment
    :param time_range: the range of time
    :param acceleration_range: the range of acceleration
    :param step: the step of the grid
    :param mode: the mode of the safe region
    :return: the safe region
    """
    # get the state space
    env = PlatoonEnv(select_scenario = mode, set_disturbance = True)
    nn_cbf_enabled = False
    nn_cbf_update = False
    safety_layer_no_grad = False
    filter_update = False
    SIDE_update = False
    if safety_layer_enabled:
        SIDE_enabled = True
    else:
        SIDE_enabled = False
    device = 'cpu'
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--is_adv_norm", type=bool, default=True, help="Advantage normalization")
    parser.add_argument("--is_state_norm", type=bool, default=True, help="State normalization")
    parser.add_argument("--is_reward_norm", type=bool, default=False, help="Reward normalization")
    parser.add_argument("--is_reward_scaling", type=bool, default=True, help="Reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Policy entropy")
    parser.add_argument("--is_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--is_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--is_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--adam_eps", type=float, default=True, help="Set Adam epsilon=1e-5")
    parser.add_argument("--is_tanh", type=float, default=True, help="Tanh activation function")
    parser.add_argument("--safety_layer_enabled", type=bool, default=safety_layer_enabled, help="Safety layer enabled or not")
    parser.add_argument("--nn_cbf_enabled", type=bool, default = nn_cbf_enabled, help="NN dynamics enabled or not")
    parser.add_argument("--cbf_tau", type=float, default=0.3, help="CAV index in the platoon")
    parser.add_argument("--cbf_gamma", type=float, default=2, help="CAV index in the platoon")
    parser.add_argument("--CAV_index", type=float, default=1, help="CAV index in the platoon")
    parser.add_argument("--CAV_idx", type=float, default=1, help="CAV index in the platoon")
    parser.add_argument("--FV1_idx", type=float, default=2, help="CAV index in the platoon")
    parser.add_argument("--FV2_idx", type=float, default=3, help="CAV index in the platoon")
    parser.add_argument("--Lf_CAV", type=float, default=0.5, help="CAV index in the platoon")
    parser.add_argument("--Lg_CAV", type=float, default=0.5, help="CAV index in the platoon")
    parser.add_argument("--Lf_FV1", type=float, default=0.5, help="CAV index in the platoon")
    parser.add_argument("--Lg_FV1", type=float, default=0.5, help="CAV index in the platoon")
    parser.add_argument("--Lf_FV2", type=float, default=0.5, help="CAV index in the platoon")
    parser.add_argument("--Lg_FV2", type=float, default=0.5, help="CAV index in the platoon") 
    parser.add_argument("--dt", type=float, default=0.1, help="CAV index in the platoon")
    parser.add_argument("--lr_cbf", type=float, default=1e-4, help="CAV index in the platoon")
    parser.add_argument("--state_size_nncbf", type=float, default=4, help="CAV index in the platoon")
    parser.add_argument("--hidden_size_nncbf", type=float, default=128, help="CAV index in the platoon")
    parser.add_argument("--output_size_nncbf", type=float, default=1, help="CAV index in the platoon")
    parser.add_argument("--safety_layer_no_grad", type=bool, default=safety_layer_no_grad, help="CAV index in the platoon")
    parser.add_argument("--car_following_parameters", type=list, default=[3,3,3], help="car following parameters initialized") #[1.2566, 1.5000, 0.9000]
    parser.add_argument("--nn_cbf_update",type=bool, default=nn_cbf_update, help="NN dynamics online update enabled or not")
    parser.add_argument("--num_episodes",type=int, default = 500, help="number of training episodes")
    parser.add_argument("--vehicle_num",type=int, default = 5, help="number of vehicles in the platoon")
    parser.add_argument("--filter_update", type=bool, default=filter_update, help="filter update enabled or not")
    parser.add_argument("--SIDE_update", type=bool, default=SIDE_update, help="SIDE update enabled or not")
    parser.add_argument("--lr_cf", type=float, default=1e-3, help="SI learning rate")
    parser.add_argument("--lr_de", type=float, default=1e-3, help="DE learning rate")
    parser.add_argument("--batch_size_SIDE", type=int, default=64, help="SIDE batch size")
    parser.add_argument("--buffer_size_SIDE", type=int, default=10000, help="SIDE buffer size")
    parser.add_argument("--SIDE_enabled", type=bool, default=SIDE_enabled, help="SIDE enabled or not")
    args = parser.parse_args()
    args.device = device
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = 5.0
    args.max_episode_steps = env.max_steps
    agent = PPOAgent(args)
    agent.load('model_parameters/', 500)

    time = np.arange(time_range[0], time_range[1], step)
    acceleration = np.arange(acceleration_range[0], acceleration_range[1], step)
    disturbance_space = np.array(np.meshgrid(time, acceleration)).T.reshape(-1, 2)
    safe_region = np.zeros((len(time), len(acceleration)))

    for disturbance in tqdm(disturbance_space):
        if_collision = test_safe(agent, env, 400, disturbance)
        safe_region[np.where(time == disturbance[0]), np.where(acceleration == disturbance[1])] = if_collision
        
    if save_region:
        np.save('SafeRegion/safe_region_' + str(mode) + '_safe_layer_' + str(safety_layer_enabled) + '.npy', safe_region)

    return safe_region

def test_safe(agent, env, max_timesteps, disturbance):
    # Test the agent
    state, acceleration = env.reset(disturbance)
    spacing = env.get_spacing()
    collision = 0

    # Run the simulation
    for t in range(max_timesteps):
        # Select an action using different policies

        action, action_prob = agent.act(state, evaluate = True, acceleration = acceleration)
        next_state, reward, next_acceleration, done, _ = env.step(action)
        state = next_state
        acceleration = next_acceleration
        # Update spacing
        spacing = env.get_spacing()

        # Check collision
        if np.min(spacing) < 0:
            collision = 1
            break
        if done:
            break
    return collision


def plot_safe_region(safe_region, time_range, acceleration_range, step, mode, safety_layer_enabled):
    """
    :param safe_region: the safe region
    :param time_range: the range of time
    :param acceleration_range: the range of acceleration
    :param step: the step of the grid
    :param mode: the mode of the safe region
    :return: the plot of the safe region
    """
    # get the state space
    time = np.arange(time_range[0], time_range[1], step)
    acceleration = np.arange(acceleration_range[0], acceleration_range[1], step)

    # plot the safe region
    plt.figure()
    plt.imshow(safe_region, origin='lower', extent=[acceleration_range[0], acceleration_range[1], time_range[0], time_range[1]])
    plt.xlabel('Acceleration')
    plt.ylabel('Time')
    # plt.title('Safe Region')
    plt.colorbar()
    plt.savefig('SafeRegion/safe_region_' + str(mode) + '_safe_layer_' + str(safety_layer_enabled) + '.png')
    plt.show()

def load_safe_region(mode, safety_layer_enabled):
    safe_region = np.load('SafeRegion/safe_region_' + str(mode) + '_safe_layer_' + str(safety_layer_enabled) + '.npy')
    return safe_region

def plot_safe_region_comparison(safe_region1, safe_region2, time_range, acceleration_range, step, mode):
    """
    :param safe_region: the safe region
    :param time_range: the range of time
    :param acceleration_range: the range of acceleration
    :param step: the step of the grid
    :param mode: the mode of the safe region
    :return: the plot of the safe region
    """
    # get the state space
    time = np.arange(time_range[0], time_range[1], step)
    acceleration = np.arange(acceleration_range[0], acceleration_range[1], step)

    # region combination
    safe_region_combined = np.zeros((len(time), len(acceleration)))
    safe_region_combined[np.where(safe_region1 == 0)] = 1
    safe_region_combined[np.where(safe_region2 == 0)] = 2

    # Find the boundary
    safe_region1_edge = np.where(cv.Canny(safe_region1.astype(np.uint8), 0, 1)>0)
    safe_region2_edge = np.where(cv.Canny(safe_region2.astype(np.uint8), 0, 1)>0)
    z1 = np.polyfit(safe_region1_edge[0], safe_region1_edge[1], 5) 
    x1 = np.arange(min(safe_region1_edge[0]), max(safe_region1_edge[0]))
    yvals1 = np.polyval(z1,x1)

    z2 = np.polyfit(safe_region2_edge[0], safe_region2_edge[1], 5)
    x2 = np.arange(min(safe_region2_edge[0]), max(safe_region2_edge[0]))
    yvals2 = np.polyval(z2,x2)

    plt.figure(figsize=(4,6))
    plt.plot(x1,yvals1)
    plt.plot(x2,yvals2)
    

    # plot the safe region
    plt.figure(figsize=(5.5,7.5))
    plt.imshow(safe_region_combined, origin='lower', extent=[acceleration_range[0], acceleration_range[1], time_range[0], time_range[1]], cmap='Blues', interpolation='none',vmin=0, vmax=2.5)
    plt.xlabel('Acceleration ($m/s^2$)', fontdict={'family' : 'Times New Roman', 'size'   : 28})
    plt.ylabel('Time (s)', fontdict={'family' : 'Times New Roman', 'size'   : 28})
    plt.yticks(fontproperties = 'Times New Roman', size = 22)
    plt.xticks(fontproperties = 'Times New Roman', size = 22)
    plt.legend(['with safety layer', 'w\o safety layer'], frameon=False, prop={'family' : 'Times New Roman', 'size'   : 18})
    # plt.title('Safe Region')
    plt.savefig('SafeRegion/safe_region_' + str(mode) + '.pdf')
    plt.show()

if __name__ == "__main__":
    generate_new_map = False
    mode = 4#2

    if generate_new_map:
        # set the range of time and acceleration
        time_range = [0,8]#[4, 10]
        acceleration_range = [0,6]#[0,3] [-6 0]
        step = 0.1
        safety_layer_enabled = False
        # get the safe region
        safe_region = cal_SafeRegion(time_range, acceleration_range, step, mode, safety_layer_enabled, save_region = True)
        # plot the safe region
        plot_safe_region(safe_region, time_range, acceleration_range, step, mode, safety_layer_enabled)
    else:
        # load the safe region
        safety_layer_enabled = True
        safe_region1 = load_safe_region(mode, safety_layer_enabled)
        safety_layer_enabled = False
        safe_region2 = load_safe_region(mode, safety_layer_enabled)
        # plot the safe region
        time_range = [0,8]#[4, 10]
        acceleration_range = [-6,0]# [0,6]
        step = 0.1
        plot_safe_region_comparison(safe_region1, safe_region2, time_range, acceleration_range, step, mode)