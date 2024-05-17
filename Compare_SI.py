import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.animation as animation
from platoon_env import PlatoonEnv
from ppo_agent import PPOAgent
import torch
import argparse
import torch.nn as nn
from SI_pretain import FC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['mathtext.default'] = 'regular'

def test(agent, env, max_timesteps, agent_type = 'ddpg'):
    # Test the agent
    state, acceleration = env.reset()
    velocity = env.get_velocity()
    spacing = env.get_spacing()
    acceleration = env.get_acceleration()
    states = np.concatenate((velocity, spacing), axis = 0)
    acceleration_ls = acceleration[np.newaxis,:]
    states = states[np.newaxis,:]
    estimated_acceleration_NN_ls = []
    estimated_acceleration_filter_ls = []
    estimated_acceleration_FC = []

    velocity_data = np.load('SI_pretrain/velocity_data.npy')
    spacing_data = np.load('SI_pretrain/spacing_data.npy')
    acceleration_data = np.load('SI_pretrain/acceleration_data.npy')
    
    X = np.array([spacing_data[:,4], velocity_data[:,4], velocity_data[:,3]]).transpose(1,0)
    scale = MinMaxScaler(feature_range=(0,1))
    scale.fit(X)

    # Run the simulation
    for t in range(max_timesteps):
        # Select an action using different policies
        if agent_type == 'ddpg':
            action, action_safe = agent.act(state, add_noise=False)
            next_state, reward, done, _ = env.step(action_safe)

        elif agent_type == 'ppo':
            action, action_prob = agent.act(state, evaluate = True, acceleration = acceleration)
            next_state, reward, next_acceleration, done, _ = env.step(action)
            agent.parameter_estimation(state,next_state, acceleration)  

        # Update the state
        state = next_state
        acceleration = next_acceleration
        
        veh_idx = 2
        
        new_state = [state[veh_idx]- agent.s_star, -state[veh_idx + 4] + agent.v_star, state[veh_idx + 4 -1]-agent.v_star]
        estimated_acceleration_old = agent.SIDE_FV1._get_car_following_estimation(new_state)
        new_state.append(estimated_acceleration_old[0][0])
        estimated_acceleration_NN = estimated_acceleration_old[0][0] + agent.SIDE_FV1._get_disturbance_estimation_2(new_state)
        estimated_acceleration_NN_ls.append(estimated_acceleration_NN)

        new_state = [state[veh_idx]- agent.s_star, -state[veh_idx + 4] + agent.v_star, state[veh_idx + 4 -1]-agent.v_star]
        cf_para = agent.filt_1.w
        estimated_acceleration_filter = cf_para[0] * new_state[0] + cf_para[1] * new_state[1] + cf_para[2] * new_state[2]
        estimated_acceleration_filter_ls.append(estimated_acceleration_filter)
            
        
        model = FC(3)
        model.load_state_dict(torch.load('SI_pretrain/fc_wo_equi.pth'))
        model.eval()
        
        x = np.array([[state[veh_idx], state[veh_idx + 4], state[veh_idx + 4 -1]]])
        x = scale.transform(x)
        estimated_acceleration_FC.append(model(torch.tensor(x, dtype=torch.float)).detach().numpy()[0][0])

        # Update the velocity and spacing
        velocity = env.get_velocity()
        spacing = env.get_spacing()
        temp_state = np.concatenate((velocity,spacing), axis = 0)
        temp_state = temp_state[np.newaxis,:]
        states = np.concatenate((states,temp_state), axis = 0)
        acceleration_ls = np.concatenate((acceleration_ls, acceleration[np.newaxis,:]), axis = 0)
        if done:
            break
    return acceleration_ls[:,veh_idx+1], np.array(estimated_acceleration_NN_ls), np.array(estimated_acceleration_filter_ls), np.array(estimated_acceleration_FC)

if __name__ == '__main__':
    
    # Set the parameters and environment (emergency braking)
    safety_layer_enabled = True
    safety_layer_no_grad = False
    pure_HDV = True
    nn_cbf_enabled = False
    nn_cbf_update = False

    filter_update = True
    SIDE_update = False
    SIDE_enabled = True

    if_plot = True
    agent_type = 'ppo'
    max_timesteps = 200
    scenario = 0
    env_rl = PlatoonEnv(select_scenario=scenario, max_steps=max_timesteps)
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 

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
    
    acc_ls, acc_ls_NN_estimated, acc_ls_filter_estimated, estimated_acceleration_FC = test(agent, env_rl, max_timesteps, agent_type)
    plt.figure(figsize=(11, 8))
    x1 = np.arange(0, max_timesteps+1, 1)
    x2 = np.arange(1, max_timesteps+1, 1)

    error_NN = np.var(acc_ls[1:] - acc_ls_NN_estimated)
    error_filter = np.var(acc_ls[1:] - acc_ls_filter_estimated)
    error_FC = np.var(acc_ls[1:] - estimated_acceleration_FC)
    print(error_NN)
    print(error_filter)
    print(error_FC)
    
    plt.plot(x2, acc_ls[1:], color = 'grey')
    plt.plot(x2, acc_ls_NN_estimated)
    plt.plot(x2, acc_ls_filter_estimated, color = 'red')
    #plt.plot(x2, estimated_acceleration_FC, color = 'pink')
    plt.legend(['Ground Truth','Proposed Method', 'RLS'],frameon=False, prop={'family' : 'Times New Roman', 'size'   : 25})
    plt.xlabel('Time step', fontdict={'family' : 'Times New Roman', 'size'   : 35})
    plt.ylabel('Acceleration ($m/s^2$)', fontdict={'family' : 'Times New Roman', 'size'   : 35})
    plt.yticks(fontproperties = 'Times New Roman', size = 35)
    plt.xticks(fontproperties = 'Times New Roman', size = 35)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/SI_comparison.pdf')

    plt.show()