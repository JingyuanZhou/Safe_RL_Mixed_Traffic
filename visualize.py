import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.animation as animation
from platoon_env import PlatoonEnv
from ddpg_agent import DDPGAgent
from ppo_agent import PPOAgent
import torch
from matplotlib.colors import ListedColormap, Normalize
import argparse
import torch.nn as nn
plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['mathtext.default'] = 'regular'

def plot_rewards(rewards, window_size=1, save_path=None, label = None, episode = 500):
    # calculate the moving average
    rewards = np.array(rewards)
    cum_rewards = np.cumsum(rewards)
    moving_average = (cum_rewards[window_size:] - cum_rewards[:-window_size]) / window_size

    # plot the moving average
    plt.plot(np.arange(len(moving_average)) + window_size, moving_average, label = label, alpha=0.9, linewidth=0.9)
    plt.xlabel('Episode', fontdict={'family' : 'Times New Roman', 'size'   : 30})
    #plt.ylabel('Reward', fontdict={'family' : 'Times New Roman', 'size'   : 35})

    # save the figure
    if save_path:
        plt.savefig(save_path)


def plot_velocity_and_spacing(velocity, spacing, veh_id=None, save_path=None):
    # plot the velocity and spacing
    t = np.arange(len(velocity))
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(t, velocity)
    plt.xlabel('Time Step')
    plt.ylabel('Velocity')
    plt.title('Vehicle Velocities')

    plt.subplot(2, 1, 2)
    plt.plot(t, spacing)
    plt.xlabel('Time Step')
    plt.ylabel('Spacing')
    plt.title('Vehicle Spacing')

    plt.tight_layout()
    plt.legend('Vehicle {}'.format(veh_id))

    if save_path:
        plt.savefig(save_path)


def test(agent, env, max_timesteps, agent_type = 'ddpg'):
    # Test the agent
    state, acceleration = env.reset()
    velocity = env.get_velocity()
    spacing = env.get_spacing()
    acceleration = env.get_acceleration()
    states = np.concatenate((velocity, spacing), axis = 0)
    acceleration_ls = acceleration[np.newaxis,:]
    states = states[np.newaxis,:]
    # Run the simulation
    for t in range(max_timesteps):
        # Select an action using different policies
        if agent_type == 'ddpg':
            action, action_safe = agent.act(state, add_noise=False)
            next_state, reward, done, _ = env.step(action_safe)

        elif agent_type == 'ppo':
            action, action_prob = agent.act(state, evaluate = True, acceleration = acceleration)
            next_state, reward, next_acceleration, done, _ = env.step(action)

        # Update the state
        state = next_state
        acceleration = next_acceleration

        # Update the velocity and spacing
        velocity = env.get_velocity()
        spacing = env.get_spacing()
        temp_state = np.concatenate((velocity,spacing), axis = 0)
        temp_state = temp_state[np.newaxis,:]
        states = np.concatenate((states,temp_state), axis = 0)
        acceleration_ls = np.concatenate((acceleration_ls, acceleration[np.newaxis,:]), axis = 0)
        if done:
            break
    return states, acceleration_ls

def plot_traj(states, vehicle_number, CAV_index, save_path=None, acceleration_ls = None, scenario = 0):
    # time steps for plotting
    t = np.arange(len(states)) 

    # color for different vehicles
    color = ['deepskyblue' if i in CAV_index else ((i+2)/(vehicle_number+5),(i+2)/(vehicle_number+5),(i+2)/(vehicle_number+5)) for i in range(vehicle_number)]

    # plot the velocity
    plt.figure(figsize=(8, 6))
    for i in range(vehicle_number):
        plt.plot(t, states[:,i], label = 'Vehicle {}'.format(i), color = color[i], lw = 2)

    # set the font size of the axis
    plt.xlabel('Time Step', fontdict={'family' : 'Times New Roman', 'size'   : 35})
    plt.ylabel('Velocity ($m/s$)', fontdict={'family' : 'Times New Roman', 'size'   : 35})
    plt.yticks(fontproperties = 'Times New Roman', size = 35)
    plt.xticks(fontproperties = 'Times New Roman', size = 35)
    plt.grid(True)
    plt.legend(frameon=0.5, prop={'family' : 'Times New Roman', 'size'   : 22}, loc = 'lower right')
    plt.tight_layout()
    # save the figure
    if save_path:
        plt.savefig(save_path + '_velocity_scenario_' + str(scenario) +'.pdf')

    # plot the spacing
    plt.figure(figsize=(8, 6))
    for i in range(vehicle_number):
        plt.plot(t, states[:,i + vehicle_number], label = 'Vehicle {}'.format(i), color = color[i], lw = 2)

    # set the font size of the axis
    plt.xlabel('Time Step', fontdict={'family' : 'Times New Roman', 'size'   : 35})
    plt.ylabel('Spacing ($m$)', fontdict={'family' : 'Times New Roman', 'size'   : 35})
    plt.axhline(y=0,ls=":",c="black")
    plt.yticks(fontproperties = 'Times New Roman', size = 35)
    plt.xticks(fontproperties = 'Times New Roman', size = 35)
    plt.grid(True)
    plt.legend(frameon=0.5, prop={'family' : 'Times New Roman', 'size'   : 22}, loc = 'lower right')
    plt.tight_layout()

    # save the figure
    if save_path:
        plt.savefig(save_path + '_spacing_scenario_' + str(scenario) + '.pdf')

    # plot the acceleration
    if acceleration_ls is not None:
        plt.figure(figsize=(8, 6))
        for i in range(vehicle_number):
            plt.plot(t, acceleration_ls[:,i], label = 'Vehicle {}'.format(i), color = color[i], lw = 2)

        # set the font size of the axis
        plt.xlabel('Time Step', fontdict={'family' : 'Times New Roman', 'size'   : 35})
        plt.ylabel('Acceleration ($m/s^2$)', fontdict={'family' : 'Times New Roman', 'size'   : 35})
        plt.yticks(fontproperties = 'Times New Roman', size = 35)
        plt.xticks(fontproperties = 'Times New Roman', size = 35)
        plt.axhline(y=-5,ls=":",c="red")
        plt.axhline(y=5,ls=":",c="red")
        plt.grid(True)
        plt.legend(frameon=0.5, prop={'family' : 'Times New Roman', 'size'   : 22}, loc = 'lower right')
        plt.tight_layout()

        # save the figure
        if save_path:
            plt.savefig(save_path + '_acceleration_scenario_' + str(scenario) +'.pdf')

def visualize(states, vehicle_number, CAV_index, gif_filename):
    # set the figure
    fig, ax = plt.subplots()
    ax.set_xlim(0, 121)
    ax.set_ylim(0, 25)
    #ax.set_xlabel('Position (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.axes.xaxis.set_ticklabels([])

    # set the color for different vehicles
    color = ['deepskyblue' if i in CAV_index else ((i+2)/(vehicle_number+5),(i+2)/(vehicle_number+5),(i+2)/(vehicle_number+5)) for i in range(vehicle_number)]
    scatters = [ax.scatter([], [], c=c, label=f'Vehicle {i}') for i, c in enumerate(color, 0)]

    ax.legend(frameon=False)
    plt.grid(True)

    # set the animation
    def update(frame):
        # update the position and velocity
        spacing_accumulation = 0
        for i, scatter in enumerate(scatters):
            spacing_accumulation += states[frame][i + vehicle_number] 
            position = (vehicle_number + 2) * 20 - spacing_accumulation
            velocity = states[frame][i]
            scatter.set_offsets(np.array([[position, velocity]]))
        ax.set_title(f'Time: {round(frame*0.1,1)}')

    # start the animation
    ani = animation.FuncAnimation(fig, update, frames=len(states), interval=100)
    # save the animation
    ani.save(gif_filename, writer='imagemagick', fps=10)
    

if __name__ == '__main__':

    # Set the parameters and environment (emergency braking)
    safety_layer_enabled = False
    safety_layer_no_grad = False
    pure_HDV = False
    nn_cbf_enabled = False
    nn_cbf_update = False
    # Set if update the filter
    filter_update = False
    # Set if update the SIDE
    SIDE_update = False
    SIDE_enabled = False
    if_plot = True

    max_timesteps = 400
    scenario = 3
    env_rl = PlatoonEnv(select_scenario=scenario, max_steps=max_timesteps)
    env_cf = PlatoonEnv(select_scenario=scenario, max_steps=max_timesteps, pure_car_following=True)
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 

    agent_type = 'ppo'

    # Create the agent
    if agent_type == 'ddpg':
        parser = argparse.ArgumentParser("Hyperparameters Setting for DDPG")
        parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
        parser.add_argument("--hidden_dims", type=list, default=[128, 128],
                            help="The number of neurons in hidden layers of the neural network")
        parser.add_argument("--lr_actor", type=float, default=1e-4, help="Learning rate of actor")
        parser.add_argument("--lr_critic", type=float, default=1e-3, help="Learning rate of critic")
        parser.add_argument("--buffer_size", type=int, default=int(1e5), help="Size of the replay buffer")
        parser.add_argument("--tau", type=float, default=1e-3, help="Update rate of target network")
        parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
        parser.add_argument("--safety_layer_enabled", type=bool, default=safety_layer_enabled, #
                            help="Safety layer enabled or not")
        parser.add_argument("--cbf_tau", type=float, default=0.1, help="CAV index in the platoon")
        parser.add_argument("--cbf_gamma", type=float, default=1, help="CAV index in the platoon")
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
        parser.add_argument("--state_size_nncbf", type=float, default=12, help="CAV index in the platoon")
        parser.add_argument("--hidden_size_nncbf", type=float, default=100, help="CAV index in the platoon")
        parser.add_argument("--output_size_nncbf", type=float, default=2, help="CAV index in the platoon")
        parser.add_argument("--safety_layer_no_grad", type=bool, default=True, help="CAV index in the platoon")
        args = parser.parse_args()
        args.device = device
        args.state_dim = env_rl.observation_space.shape[0]
        args.action_dim = env_rl.action_space.shape[0]
        args.max_action = 5.0
        args.max_episode_steps = env_rl.max_steps
        agent = DDPGAgent(args)

    elif agent_type == 'ppo':
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
        parser.add_argument("--safety_layer_enabled", type=bool, default=safety_layer_enabled, help="Safety layer enabled or not") #default = safety_layer_enabled
        parser.add_argument("--nn_cbf_enabled", type=bool, default=nn_cbf_enabled, help="NN dynamics enabled or not")
        parser.add_argument("--CAV_index", type=float, default=1, help="CAV index in the platoon")
        parser.add_argument("--cbf_tau", type=float, default=0.3, help="CAV index in the platoon")
        parser.add_argument("--cbf_gamma", type=float, default=1, help="CAV index in the platoon")
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
        parser.add_argument("--car_following_parameters", type=list, default=[3,3,3], help="car following parameters initialized") #0.5,0.4,0.3 [2,2,2] [1.2566, 1.5000, 0.9000]
        parser.add_argument("--nn_cbf_update",type=bool, default=nn_cbf_update, help="NN dynamics online update enabled or not")
        parser.add_argument("--num_episodes",type=int, default=100, help="number of episodes for training")
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
        args.state_dim = env_rl.observation_space.shape[0]
        args.action_dim = env_rl.action_space.shape[0]
        args.max_action = 5.0
        args.max_episode_steps = env_rl.max_steps
        agent = PPOAgent(args)

    # Load the pre-trained model
    agent.load('model_parameters/', 500)
    # Test the pre-trained model
    vehicle_number = 5
    CAV_index_zero = []
    CAV_index_rl = [2]
    
    # Test pure HDV scenario
    if pure_HDV:
        states_cf, acc_ls_cf = test(agent, env_cf, max_timesteps, agent_type)
        if if_plot:
            plot_traj(states_cf, vehicle_number, CAV_index_zero, save_path='results/test_cf_traj',scenario=scenario, acceleration_ls = acc_ls_cf)
            visualize(states_cf, vehicle_number, CAV_index_zero, gif_filename='results/test_cf_traj' + '_scenario_'+ str(scenario) +'.gif')
        # save the results
        np.save('results/test_cf_traj' + '_scenario_'+ str(scenario) +'.npy', states_cf)

    # Test RL scenario
    if nn_cbf_enabled:
        prompt = 'nn_cbf'
        agent.barrier_optimizer_FV1.load('model_parameters/', 500, 'FV1')
        agent.barrier_optimizer_FV2.load('model_parameters/', 500, 'FV2')
    else:
        prompt = 'no_nn_cbf'
    states_rl, acc_ls_rl = test(agent, env_rl, max_timesteps, agent_type)
    

    if safety_layer_enabled:
        if if_plot:
            plot_traj(states_rl, vehicle_number, CAV_index_rl, save_path='results/' + agent_type + '_test_rl_traj' + '_' + prompt+'_SIDE_'+str(SIDE_enabled), acceleration_ls = acc_ls_rl, scenario = scenario)
            visualize(states_rl, vehicle_number, CAV_index_rl, gif_filename='results/' + agent_type + '_test_rl_traj' + '_' + prompt + '_scenario_'+ str(scenario) + '_SI_enabled_'+ str(SIDE_enabled) + '.gif')
        # save the results
        np.save('results/' + agent_type + '_test_rl_traj' + '_' + prompt + '_scenario_'+ str(scenario) + '_SI_enabled_'+ str(SIDE_enabled) + '.npy', states_rl)
    elif safety_layer_no_grad:
        if if_plot:
            plot_traj(states_rl, vehicle_number, CAV_index_rl, save_path='results/' + agent_type + '_test_rl_traj_no_grad' + '_' + prompt, acceleration_ls = acc_ls_rl, scenario = scenario)
            visualize(states_rl, vehicle_number, CAV_index_rl, gif_filename='results/' + agent_type + '_test_rl_traj_no_grad' + '_' + prompt + '_scenario_'+ str(scenario) + '_SI_enabled_'+ str(SIDE_enabled) + '.gif')
        # save the results
        np.save('results/' + agent_type + '_test_rl_traj_no_grad' + '_' + prompt + '_scenario_'+ str(scenario) + '_SI_enabled_'+ str(SIDE_enabled) + '.npy', states_rl)
    else:
        if if_plot:
            plot_traj(states_rl, vehicle_number, CAV_index_rl, save_path='results/' + agent_type + '_test_rl_traj_no_safe', acceleration_ls = acc_ls_rl, scenario = scenario)
            visualize(states_rl, vehicle_number, CAV_index_rl, gif_filename='results/' + agent_type + '_test_rl_traj_no_safe' + '_scenario_'+ str(scenario) + '.gif')
        # save the results
        np.save('results/' + agent_type + '_test_rl_traj_no_safe' + '_scenario_'+ str(scenario) + '.npy', states_rl)
    plt.show()


    


