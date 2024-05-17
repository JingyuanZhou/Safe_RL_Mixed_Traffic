import numpy as np
from ddpg_agent import DDPGAgent
from ppo_agent import PPOAgent
from platoon_env import PlatoonEnv
from visualize import plot_rewards, plot_velocity_and_spacing
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
from torch.utils.tensorboard import SummaryWriter
import random

def train(agent, env, num_episodes=500, agent_type = 'ppo', safety_layer_enabled = False, nn_cbf_update = False, filter_update = True, SIDE_update = False):
    '''
    Train the agent
    '''
    episode_rewards = []
    velocity_data = []
    spacing_data = []
    # Build a tensorboard
    writer = SummaryWriter(log_dir='training_traj/platoon_' + agent_type + '_safety_layer_' + str(safety_layer_enabled))
    # Train the agent
    for episode in range(num_episodes+1):
        # Reset the environment
        state, acceleration = env.reset()
        env_select = random.random()
        if env_select < 0.5:
            env.select_scenario = 0
        elif env_select < 0.75:
            env.select_scenario = 1
        #elif env_select < 0.75:
        #    env.select_scenario = 2
        #elif env_select < 0.875:
        #    env.select_scenario = 3
        else:
            env.select_scenario = 4

        env.select_scenario = 0

        done = False
        episode_reward = 0
        total_steps = 0
        while not done:
            # Select an action using different policies
            if agent_type == 'ddpg':
                action, action_safe = agent.act(state)
                next_state, reward, done, _ = env.step(action_safe)
                agent.step(state, action, action_safe, reward, next_state, done, total_steps)

            elif agent_type == 'ppo':
                action, action_prob = agent.act(state, acceleration = acceleration)
                next_state, reward, next_acceleration, done, _ = env.step(action)
                agent.step(state, action, action_prob, reward, next_state, done, total_steps, acceleration)

            # Update the state
            state = next_state
            acceleration = next_acceleration
            episode_reward += reward

            # Collect data for visualization
            velocity_data.append(env.get_velocity())
            spacing_data.append(env.get_spacing())

        # Update the learning rate
        total_steps += 1
        agent.lr_decay(total_steps)

        # Save the rewards
        episode_rewards.append(episode_reward)
        # Save the rewards to tensorboard
        writer.add_scalar('step_rewards_' + agent_type, episode_reward, global_step = episode)
            
        if episode%10 == 0:
            if SIDE_update:
                agent.SIDE_FV1.save_model('model_parameters/SIDE_FV1_')
                agent.SIDE_FV2.save_model('model_parameters/SIDE_FV2_')
        # Save the model parameters
        if episode%50 == 0:
            agent.save("model_parameters", episode)
            if nn_cbf_update:
                agent.barrier_optimizer_FV1.save("model_parameters", episode, 'FV1')
                agent.barrier_optimizer_FV2.save("model_parameters", episode, 'FV2')
        if nn_cbf_update:
            print("Episode: {}, Reward: {}, Loss_FV1: {}, Loss_FV2: {}".format(episode, episode_reward, np.mean(agent.barrier_optimizer_FV1.loss_lst[-2:-1]), np.mean(agent.barrier_optimizer_FV2.loss_lst[-2:-1])))
        elif filter_update:
            print("Episode: {}, Reward: {}, Parameter FW1: {}, Parameter FW2: {}".format(episode, episode_reward, agent.FW1_parameters, agent.FW2_parameters))
        elif SIDE_update:
            print("Episode: {}, Reward: {}, Parameter FW1: {}, Parameter FW2: {}, Loss CF FW1: {}, Loss CF FW2: {}, Loss Bias FW1: {}, Loss Bias FW2: {}".format(episode, episode_reward, agent.FW1_parameters, agent.FW2_parameters, agent.SIDE_FV1.loss_cf_lst[-1], agent.SIDE_FV2.loss_cf_lst[-1], agent.SIDE_FV1.loss_de_lst[-1], agent.SIDE_FV2.loss_de_lst[-1]))
        else:
            print("Episode: {}, Reward: {}".format(episode, episode_reward))

    # Convert data to NumPy arrays
    velocity_data = np.array(velocity_data)
    spacing_data = np.array(spacing_data)
    
    if nn_cbf_update:
        agent.barrier_optimizer_FV1.save_loss_data('training_traj/loss_data_neural_barrier_FV1_' + agent_type + '.csv')
        agent.barrier_optimizer_FV2.save_loss_data('training_traj/loss_data_neural_barrier_FV2_' + agent_type + '.csv')
    if SIDE_update:
        agent.SIDE_FV1.save_model('model_parameters/SIDE_FV1_')
        agent.SIDE_FV2.save_model('model_parameters/SIDE_FV2_')

    print('Following vehicle 1 weights:', agent.FW1_parameters, ', Following vehicle 2 weights:', agent.FW2_parameters)

    return episode_rewards, velocity_data, spacing_data


def test(agent, env, agent_type, num_episodes=50):
    '''
    Test the agent
    '''
    episode_rewards = []
    velocity_data = []
    spacing_data = []
    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()
        done = False
        episode_reward = 0

        # Run the episode
        while not done:
            # Select an action using different policies
            if agent_type == 'ddpg':
                _, action = agent.act(state, add_noise=False)
            elif agent_type == 'ppo':
                action, _ = agent.act(state)
            # Take the action
            next_state, reward, done, _ = env.step(action)

            # Update the state
            state = next_state
            episode_reward += reward
            # Collect data for visualization
            velocity_data.append(env.get_velocity())
            spacing_data.append(env.get_spacing())

        # Print the test result
        print("Test Episode: {}".format(episode))

        # Save the rewards
        episode_rewards.append(episode_reward)

    # Convert data to NumPy arrays
    velocity_data = np.array(velocity_data)
    spacing_data = np.array(spacing_data)

    return episode_rewards, velocity_data, spacing_data

if __name__ == "__main__":
    # Create the environment
    env = PlatoonEnv()
    # Set the device
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    # Set the safety layer
    safety_layer_enabled = True
    safety_layer_no_grad = False
    nn_cbf_enabled = False
    nn_cbf_update = False
    # Select the agent
    agent_select = 'ppo'
    # Set if train the agent
    agent_train = True
    # Set if update the filter
    filter_update = False
    # Set if update the SIDE
    SIDE_update = True
    SIDE_enabled = True

    # Create the agent
    if agent_select == 'ddpg':
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
        parser.add_argument("--safety_layer_enabled", type=bool, default=safety_layer_enabled,
                            help="Safety layer enabled or not")
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
        parser.add_argument("--state_size_nncbf", type=float, default=12, help="CAV index in the platoon")
        parser.add_argument("--hidden_size_nncbf", type=float, default=100, help="CAV index in the platoon")
        parser.add_argument("--output_size_nncbf", type=float, default=1, help="CAV index in the platoon")
        
        args = parser.parse_args()
        args.device = device
        args.state_dim = env.observation_space.shape[0]
        args.action_dim = env.action_space.shape[0]
        args.max_action = 5.0
        args.max_episode_steps = env.max_steps
        agent = DDPGAgent(args)

    elif agent_select == 'ppo':
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
        parser.add_argument("--car_following_parameters", type=list, default=[0.5,0.5,0.5], help="car following parameters initialized") #[1.2566, 1.5000, 0.9000]
        parser.add_argument("--nn_cbf_update",type=bool, default=nn_cbf_update, help="NN dynamics online update enabled or not")
        parser.add_argument("--num_episodes",type=int, default = 500, help="number of training episodes")
        parser.add_argument("--vehicle_num",type=int, default = 5, help="number of vehicles in the platoon")
        parser.add_argument("--filter_update", type=bool, default=filter_update, help="filter update enabled or not")
        parser.add_argument("--SIDE_update", type=bool, default=SIDE_update, help="SIDE update enabled or not")
        parser.add_argument("--lr_cf", type=float, default=1e-4, help="SI learning rate")
        parser.add_argument("--lr_de", type=float, default=1e-4, help="DE learning rate")
        parser.add_argument("--batch_size_SIDE", type=int, default=256, help="SIDE batch size")
        parser.add_argument("--buffer_size_SIDE", type=int, default=10000, help="SIDE buffer size")
        parser.add_argument("--SIDE_enabled", type=bool, default=SIDE_enabled, help="SIDE enabled or not")
        args = parser.parse_args()
        args.device = device
        args.state_dim = env.observation_space.shape[0]
        args.action_dim = env.action_space.shape[0]
        args.max_action = 5.0
        args.max_episode_steps = env.max_steps
        agent = PPOAgent(args)

    else:
        raise ValueError('Invalid agent type')

    # Train new model or load pretrained model
    if agent_train:
        # Train the agent
        episode_rewards, velocity_data, spacing_data = train(agent, env, agent_type = agent_select, safety_layer_enabled = safety_layer_enabled, nn_cbf_update = nn_cbf_update, filter_update = filter_update, SIDE_update = SIDE_update)
        # Save the training data
        if safety_layer_enabled:
            episode_rewards_pd = pd.DataFrame(episode_rewards).to_csv('training_traj/episode_rewards_' + agent_select + '.csv')
            velocity_data_pd = pd.DataFrame(velocity_data).to_csv('training_traj/velocity_data_' + agent_select + '.csv')
            spacing_data_pd = pd.DataFrame(spacing_data).to_csv('training_traj/spacing_data_' + agent_select + '.csv')
        else:
            episode_rewards_pd = pd.DataFrame(episode_rewards).to_csv('training_traj/episode_rewards_no_safety_' + agent_select + '.csv')
            velocity_data_pd = pd.DataFrame(velocity_data).to_csv('training_traj/velocity_data_no_safety_' + agent_select + '.csv')
            spacing_data_pd = pd.DataFrame(spacing_data).to_csv('training_traj/spacing_data_no_safety_' + agent_select + '.csv')
        # Visualize the collected data
        plot_rewards(episode_rewards, episode=500)
        plot_velocity_and_spacing(velocity_data, spacing_data)
        plt.show()
    else:
        agent.load("model_parameters", 500)
        # Test the agent
        episode_rewards, velocity_data, spacing_data = test(agent, env, agent_select)
        # Visualize the collected data
        plot_rewards(episode_rewards, episode=50)
        plot_velocity_and_spacing(velocity_data, spacing_data)
        plt.show()


