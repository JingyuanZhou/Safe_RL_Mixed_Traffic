import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from platoon_env import PlatoonEnv
from ppo_agent import PPOAgent
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from NN_SI import NN_SI_DE_Module, OVM_Estimator, Disturbance_Estimator
import random
from torch.optim.lr_scheduler import LinearLR
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class FC(nn.Module):
    def __init__(self, state_num):
        super(FC,self).__init__()
        self.fc1 = nn.Linear(state_num, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 1)
        self.tanh = nn.Tanh()
        self.ReLU = nn.ReLU()
        
    def forward(self,x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

def load_pretrain_model_and_test():
    # Create the environment
    max_timesteps = 10000
    env = PlatoonEnv(max_steps=max_timesteps)
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
    agent_train = False
    # Set if update the filter
    filter_update = False
    # Set if update the SIDE
    SIDE_update = False
    SIDE_enabled = True

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

    agent.load("model_parameters", 500)
    #env.select_scenario = 4
    #velocity_data, spacing_data = test(agent, env, agent_select)

    return agent, env, agent_select #velocity_data, spacing_data

def test(agent, env, train_type, num_episodes=1):
    '''
    Test the agent
    '''
    episode_rewards = []
    velocity_data = []
    spacing_data = []
    acceleration_data = []
    for episode in range(num_episodes):
        # Reset the environment
        state, acceleration = env.reset()

        if train_type == 0:
            env.select_scenario = 0
        else:
            env_select = random.random()
            if env_select < 0.8:
                env.select_scenario = 0
            elif env_select < 0.9:
                env.select_scenario = 1
            #elif env_select < 0.75:
            #    env.select_scenario = 2
            #elif env_select < 0.875:
            #    env.select_scenario = 3
            else:
                env.select_scenario = 4

        done = False
        episode_reward = 0

        # Run the episode
        while not done:
            # Select an action using different policies
            action, action_prob = agent.act(state, acceleration = acceleration)
            next_state, reward, next_acceleration, done, _ = env.step(action)

            # Update the state
            state = next_state
            episode_reward += reward
            # Collect data for visualization
            velocity_data.append(env.get_velocity())
            spacing_data.append(env.get_spacing())
            acceleration_data.append(env.get_acceleration())

        # Print the test result
        # print("Test Episode: {}".format(episode))

        # Save the rewards
        episode_rewards.append(episode_reward)

    # Convert data to NumPy arrays
    s_star = 20
    v_star = 15
    velocity_data = np.array(velocity_data)
    spacing_data = np.array(spacing_data)
    acceleration_data = np.array(acceleration_data)

    return velocity_data, spacing_data, acceleration_data

if __name__ == "__main__":
    data_saving = False
    if data_saving:
        agent, env, agent_select = load_pretrain_model_and_test()
        velocity_data, spacing_data, acceleration_data = test(agent, env, 0)
        np.save('SI_pretrain/velocity_data.npy', velocity_data)
        np.save('SI_pretrain/spacing_data.npy', spacing_data)
        np.save('SI_pretrain/acceleration_data.npy', acceleration_data)
    else:
        velocity_data = np.load('SI_pretrain/velocity_data.npy')
        spacing_data = np.load('SI_pretrain/spacing_data.npy')
        acceleration_data = np.load('SI_pretrain/acceleration_data.npy')
        
        X = np.array([spacing_data[:,4], velocity_data[:,4], velocity_data[:,3]]).transpose(1,0)
        scale = MinMaxScaler(feature_range=(0,1))
        scale.fit(X)
        new_spacing_data= scale.transform(X)[:,0]
        new_velocity_data = scale.transform(X)[:,1]
        new_preceding_velocity_data = scale.transform(X)[:,2]

        train_data = Data.TensorDataset(torch.tensor([new_spacing_data,new_velocity_data,new_preceding_velocity_data], dtype=torch.float32).transpose(0, 1), torch.tensor(acceleration_data[:,4], dtype=torch.float32))
        
        
        train_data_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)
        
        lr = 1e-3
        state_num = 3
        device = 'cpu'

        model = FC(state_num).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        epochs = 5000

        scheduler = LinearLR(optimizer, start_factor=1, end_factor=1/100, total_iters=epochs)

        dt = 0.1
        #agent, env, agent_select = load_pretrain_model_and_test()

        for i in range(epochs):
            for step, (state, acceleration) in enumerate(train_data_loader):
                optimizer.zero_grad()
                action_pred = model(state)
                loss = F.mse_loss(action_pred.squeeze(1),  acceleration)
                loss.backward()
                optimizer.step()
                scheduler.step()

            if i % 1 == 0:
                print('epochs: ', i, 'loss: ', loss.item())
            torch.save(model.state_dict(), 'SI_pretrain/fc_wo_equi.pth')
        '''
        lr_cf = 1e-2
        lr_de = 1e-2
        state_num = 3
        action_num = 1

        device = 'cpu'
        car_following_estimator = OVM_Estimator().to(device)
        disturbance_estimator = Disturbance_Estimator(state_num, action_num).to(device)

        optimizer_cf = torch.optim.Adam(car_following_estimator.parameters(), lr = lr_cf)
        optimizer_de = torch.optim.Adam(disturbance_estimator.parameters(), lr = lr_de)

        epochs = 200
        scheduler_cf = LinearLR(optimizer_cf, start_factor=1, end_factor=1/400, total_iters=epochs)
        scheduler_de = LinearLR(optimizer_de, start_factor=1, end_factor=1/400, total_iters=epochs)

        dt = 0.1
        agent, env, agent_select = load_pretrain_model_and_test()
        for i in range(epochs):
            velocity_data, spacing_data = test(agent, env, 0)
            state = torch.tensor([spacing_data[:-1,4], -velocity_data[:-1,4], velocity_data[:-1,3]], dtype=torch.float32).transpose(0, 1)
            next_state = torch.tensor([spacing_data[1:,4], -velocity_data[1:,4], velocity_data[1:,3]], dtype=torch.float32).transpose(0, 1)

            optimizer_cf.zero_grad()

            action_pred = car_following_estimator(state)

            loss_cf = F.mse_loss(action_pred.squeeze(1),  next_state[:,1]-state[:,1])
            loss_cf.backward()
            optimizer_cf.step()
            scheduler_cf.step()

            if i % 1 == 0:
                print('epochs: ', i, 'loss_cf: ', loss_cf.item())
                print('car-following model parameters: ', [car_following_estimator.alpha1.cpu().detach().numpy().tolist(), car_following_estimator.alpha2.cpu().detach().numpy().tolist(), car_following_estimator.alpha3.cpu().detach().numpy().tolist()])
            
        torch.save(car_following_estimator.state_dict(), 'SI_pretrain/car_following_estimator.pth')
    
        for i in range(epochs):
            velocity_data, spacing_data = test(agent, env, 1)
            state = torch.tensor([spacing_data[:-1,5], -velocity_data[:-1,5], velocity_data[:-1,3]], dtype=torch.float32).transpose(0, 1)
            next_state = torch.tensor([spacing_data[1:,5], -velocity_data[1:,5], velocity_data[1:,3]], dtype=torch.float32).transpose(0, 1)

            optimizer_de.zero_grad()

            action_pred = car_following_estimator(state)
            action_disturbance = disturbance_estimator(torch.cat((state, action_pred.detach().cpu()), 1))

            next_state_pred_w_de = -state[:, 1] + dt * (action_pred.detach().cpu().squeeze(1) + action_disturbance)

            loss_de = F.mse_loss(action_pred.squeeze(1),  next_state[:,1]-state[:,1])
            loss_de.backward()
            optimizer_de.step()
            scheduler_de.step()

            if i % 1 == 0:
                print('epochs: ', i, 'loss_de: ', loss_de.item())

        torch.save(disturbance_estimator.state_dict(), 'SI_pretrain/disturbance_estimator.pth')

        '''
        

