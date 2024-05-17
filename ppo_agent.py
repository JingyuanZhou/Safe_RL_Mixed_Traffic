import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Normal
from utils import ReplayBuffer_PPO
import os
from BarrierNet import BarrierLayer
from NeuralBarrier import *
import argparse
from platoon_env import PlatoonEnv
import padasip as pa
from NN_SI import NN_SI_DE_Module

# Orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        # Initialize the parameters of the network
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.is_tanh]  #  use tanh
        self.safety_layer_enabled = args.safety_layer_enabled
        self.tau = args.cbf_tau
        self.CAV_index = args.CAV_idx
        self.safety_layer_no_grad = args.safety_layer_no_grad
        self.car_following_parameters = args.car_following_parameters
        if self.safety_layer_enabled or self.safety_layer_no_grad:
            self.safeLayer = BarrierLayer(args.state_dim, self.car_following_parameters, self.safety_layer_no_grad, SIDE_enabled=args.SIDE_enabled)
        else:
            self.safeLayer = None
        # Use orthogonal initialization
        if args.is_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s, La_FV1 = None, La_FV2 = None, Learning_CBF = False, acceleration = None, cf_saturation_FW1 = None, cf_saturation_FW2 = None):
        # Get the mean of the Gaussian distribution based on the current state
        x = self.activate_func(self.fc1(s))
        x = self.activate_func(self.fc2(x))
        mean = self.max_action * torch.tanh(self.mean_layer(x))  # [-1,1]->[-max_action,max_action]
        if self.safety_layer_enabled or self.safety_layer_no_grad:
            mean_safe = self.safeLayer(mean, s, self.tau, 1, self.CAV_index, La_FV1, La_FV2, Learning_CBF, acceleration, cf_saturation_FW1, cf_saturation_FW2)
            mean = mean + mean_safe
        return mean

    def get_dist(self, s, acceleration = None, cf_saturation_FW1 = None, cf_saturation_FW2 = None):
        # Get the Gaussian distribution based on the current state
        mean = self.forward(s, acceleration = acceleration, cf_saturation_FW1 = cf_saturation_FW1, cf_saturation_FW2 = cf_saturation_FW2)
        log_std = self.log_std.expand_as(mean)  # To expand 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Generate the Gaussian distribution based on mean and std
        return dist
    
    def get_act_from_dist(self, s, La_FV1 = None, La_FV2 = None, Learning_CBF = False, acceleration = None, cf_saturation_FW1 = None, cf_saturation_FW2 = None):
        gamma = 1
        dist = self.get_dist(s, acceleration, cf_saturation_FW1, cf_saturation_FW2)
        # Sample the action according to the probability distribution (reparameterization trick)
        a = dist.rsample() 
        a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
        a_logprob = dist.log_prob(a)  # The log probability density of the action

        # Use the safety layer to add the safety control input
        if self.safety_layer_enabled or self.safety_layer_no_grad:
            a_safe = self.safeLayer(a, s, self.tau, gamma, self.CAV_index, La_FV1, La_FV2, Learning_CBF, acceleration, cf_saturation_FW1, cf_saturation_FW2)
            a = a + a_safe
        return a, a_logprob


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        # Initialize the parameters of the network
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.is_tanh]  # use tanh activation function

        # Use orthogonal initialization
        if args.is_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        # Get the value of the current state
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPOAgent():
    def __init__(self, args):
        # Initialize the parameters of the agent
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.adam_eps = args.adam_eps
        self.is_grad_clip = args.is_grad_clip
        self.is_lr_decay = args.is_lr_decay
        self.is_adv_norm = args.is_adv_norm
        self.safety_layer_enabled = args.safety_layer_enabled
        self.CAV_index = args.CAV_index
        self.device = args.device
        self.safety_layer_no_grad = args.safety_layer_no_grad
        self.nn_cbf_enabled = args.nn_cbf_enabled
        self.replay_buffer = ReplayBuffer_PPO(args)
        self.nn_cbf_update = args.nn_cbf_update
        self.FV1_idx = args.FV1_idx
        self.FV2_idx = args.FV2_idx
        self.filter_update = args.filter_update
        self.SIDE_update = args.SIDE_update
        self.SIDE_enabled = args.SIDE_enabled
        
        self.FW1_parameters = args.car_following_parameters
        self.FW2_parameters = args.car_following_parameters
        self.s_star = 20
        self.v_star = 15
        self.filt_enable = False
        # Initialize the actor and critic networks
        self.actor = Actor(args).to(self.device)
        self.critic = Critic(args).to(self.device)

        if self.adam_eps:  # Set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.car_following_parameters = args.car_following_parameters #[1.2566, 1.5000, 0.9000]
        
        
        if self.nn_cbf_enabled:
            self.barrier_optimizer_FV1 = barrier_optimizer(args.state_size_nncbf, args.hidden_size_nncbf, args.output_size_nncbf, 10000, args.Lf_FV1, args.Lg_FV1, args.gamma, args.cbf_tau, args.CAV_idx, args.FV1_idx, args.dt, args.lr_cbf, 512, args.device, args.car_following_parameters)
            self.barrier_optimizer_FV2 = barrier_optimizer(args.state_size_nncbf, args.hidden_size_nncbf, args.output_size_nncbf, 10000, args.Lf_FV2, args.Lg_FV2, args.gamma, args.cbf_tau, args.CAV_idx, args.FV2_idx, args.dt, args.lr_cbf, 512, args.device, args.car_following_parameters)
            self.cbf_tau = args.cbf_tau
            if os.path.exists('model_parameters/FW_FV1_episode_500.pth'):
                self.barrier_optimizer_FV1.compensator.load_state_dict(torch.load('model_parameters/FW_FV1_episode_500.pth'))
                print('Load FV1 compensator successfully!')
            if os.path.exists('model_parameters/FW_FV2_episode_500.pth'):
                self.barrier_optimizer_FV2.compensator.load_state_dict(torch.load('model_parameters/FW_FV2_episode_500.pth'))
                print('Load FV2 compensator successfully!')

        self.filt_1 = pa.filters.FilterRLS(3, mu=0.99, w = self.car_following_parameters)
        self.filt_2 = pa.filters.FilterRLS(3, mu=0.99, w = self.car_following_parameters)

        self.SIDE_FV1 = NN_SI_DE_Module(3, 1, args.lr_cf, args.lr_de, args.batch_size_SIDE, args.buffer_size_SIDE, args.device, args.FV1_idx)
        self.SIDE_FV2 = NN_SI_DE_Module(3, 1, args.lr_cf, args.lr_de, args.batch_size_SIDE, args.buffer_size_SIDE, args.device, args.FV2_idx)
        if self.SIDE_enabled:
            self.SIDE_FV1.load_model('model_parameters/SIDE_FV1_')
            #self.SIDE_FV1.load_model('SI_pretrain/')
            self.SIDE_FV2.load_model('model_parameters/SIDE_FV2_')
            self.FW1_parameters = self.SIDE_FV1.car_following_model_parameters()
            self.FW2_parameters = self.SIDE_FV2.car_following_model_parameters()
            self.actor.safeLayer.FW1_parameters = self.FW1_parameters
            self.actor.safeLayer.FW2_parameters = self.FW2_parameters
            self.car_following_parameters = self.FW1_parameters
        self.num_episodes = args.num_episodes
        self.input_blending_weight = np.arange(self.num_episodes) / (self.num_episodes - 1)
        self.episode_cnt = 0

        # Initialize the safety layer
        self.safeLayer = None
        if self.safety_layer_enabled or self.safety_layer_no_grad:
            self.safeLayer = BarrierLayer(None, self.car_following_parameters, self.safety_layer_no_grad, self.SIDE_enabled)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        tau = 0.3
        gamma = 1
        La_FV1 = self.barrier_optimizer_FV1.compensator(s)
        La_FV2 = self.barrier_optimizer_FV2.compensator(s)
        
        a = self.actor(s, La_FV1, La_FV2, self.Learning_CBF).detach().numpy().flatten()
        # if self.safety_layer_enabled:
        #    a_safe = self.safeLayer(a, s, self.cbf_tau, gamma, self.CAV_index, La_FV1, La_FV2)
        #    a = a + a_safe
        return a

    def act(self, s, add_noise=False, evaluate = False, acceleration = None, cf_saturation = None):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        acceleration = torch.unsqueeze(torch.tensor(acceleration, dtype=torch.float), 0).to(self.device)
        # CBF parameters for the safety layer
        tau = 0.3
        gamma = 1 

        # If training or evaluating
        if self.nn_cbf_enabled:
            La_FV1 = self.barrier_optimizer_FV1.cal_La(s)
            La_FV2 = self.barrier_optimizer_FV2.cal_La(s)
        else:
            La_FV1 = None
            La_FV2 = None

        if self.SIDE_enabled:
            cf_saturation_FW1 = self.SIDE_FV1._get_disturbance_estimation(s)
            cf_saturation_FW2 = self.SIDE_FV2._get_disturbance_estimation(s)
        else:
            cf_saturation_FW1 = None
            cf_saturation_FW2 = None
            
        if not evaluate:
            # Get the Gaussian distribution based on the current state
            # dist = self.actor.get_dist(s)
            # Sample the action according to the probability distribution (reparameterization trick)
            # a = dist.rsample() 
            # a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
            # a_logprob = dist.log_prob(a)  # The log probability density of the action

            # Use the safety layer to add the safety control input
            #if self.safety_layer_enabled:
            #    a_safe = self.safeLayer(a, s, tau, gamma, self.CAV_index)
            #    a = a + a_safe
            a, a_logprob = self.actor.get_act_from_dist(s, La_FV1, La_FV2, self.nn_cbf_enabled, acceleration, cf_saturation_FW1, cf_saturation_FW2)
        else:
            # If evaluating, we only use the mean
            a = self.actor(s, La_FV1, La_FV2, self.nn_cbf_enabled,acceleration, cf_saturation_FW1, cf_saturation_FW2)

            # Use the safety layer to add the safety control input
            # if self.safety_layer_enabled:
            #     a_safe = self.safeLayer(a, s, tau, gamma, self.CAV_index)
            #     a = a + a_safe
            return a.cpu().detach().numpy().flatten(), None
        
        # Return the action and the log probability density of the action
        with torch.no_grad():
           return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()
        # return a, a_logprob

    def step(self, s, a, a_logprob, r, s_, done, total_steps, acceleration):
        if self.filter_update or self.SIDE_update:
            self.parameter_estimation(s, s_, acceleration)      
        
        if self.nn_cbf_update:
            
            self.barrier_optimizer_FV1.step_optimize(s, a)
            self.barrier_optimizer_FV2.step_optimize(s, a)

        
        self.replay_buffer.store(s, a, a_logprob, r, s_, done, acceleration)  # Store the transition in the replay buffer
        
        if self.replay_buffer.count == self.batch_size:
            s, a, a_logprob, r, s_, done, acceleration = self.replay_buffer.numpy_to_tensor()  # Get training data
            s, a, a_logprob, r, s_, done, acceleration = s.to(self.device), a.to(self.device), a_logprob.to(self.device), r.to(self.device), s_.to(self.device), done.to(self.device), acceleration.to(self.device)

            if self.SIDE_enabled:
                cf_saturation_FW1 = self.SIDE_FV1._get_disturbance_estimation(s)
                cf_saturation_FW2 = self.SIDE_FV2._get_disturbance_estimation(s)
            else:
                cf_saturation_FW1 = None
                cf_saturation_FW2 = None
            advantages = []
            gae = 0
            with torch.no_grad():  # advantages and v_target have no gradient
                vs = self.critic(s)
                vs_ = self.critic(s_)
                deltas = r.cpu() + self.gamma * vs_.cpu()- vs.cpu()
                for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.cpu().flatten().numpy())):
                    gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                    advantages.insert(0, gae)
                advantages = torch.tensor(advantages, dtype=torch.float).view(-1, 1)
                v_target = advantages + vs.cpu()
                if self.is_adv_norm:  # Advantage normalization
                    advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-5))

            # Optimize policy for K epochs:
            for _ in range(self.K_epochs):
                # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
                for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                    if self.SIDE_enabled:
                        dist_current = self.actor.get_dist(s[index], acceleration[index], cf_saturation_FW1 = cf_saturation_FW1[index], cf_saturation_FW2 = cf_saturation_FW2[index])
                    else:
                        dist_current = self.actor.get_dist(s[index], acceleration[index])
                    dist_entropy = dist_current.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                    a_logprob_current = dist_current.log_prob(a[index])
                    # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                    ratios = torch.exp(a_logprob_current.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                    surr1 = ratios * advantages[index].to(self.device)  # Only calculate the gradient of 'a_logprob_current' in ratios
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages[index].to(self.device)
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Policy entropy
                    # Update actor
                    self.optimizer_actor.zero_grad()
                    actor_loss.mean().backward()
                    if self.is_grad_clip:  # Gradient clip
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.optimizer_actor.step()
                    if self.safety_layer_enabled:
                        self.actor.safeLayer.gamma.data = torch.clamp(self.actor.safeLayer.gamma.data, 0, 10)
                        self.actor.safeLayer.k1.data = torch.clamp(self.actor.safeLayer.k1.data, 0, 10)

                    v_s = self.critic(s[index])
                    critic_loss = F.mse_loss(v_target[index].to(self.device), v_s)
                    # Update critic
                    self.optimizer_critic.zero_grad()
                    critic_loss.backward()
                    if self.is_grad_clip:  # Gradient clip
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.optimizer_critic.step()

            self.replay_buffer.count = 0
            if self.is_lr_decay:  # Learning rate Decay
                self.lr_decay(total_steps)

    def parameter_estimation(self, state, next_state, acceleration):
        state_FW1 = state[[self.FV1_idx, self.FV1_idx + 4, self.FV1_idx + 4 - 1]]
        state_FW2 = state[[self.FV2_idx, self.FV2_idx + 4, self.FV2_idx + 4 - 1]]
        state_FW1[0] = state_FW1[0] - self.s_star
        state_FW2[0] = state_FW2[0] - self.s_star
        state_FW1[1] = - (state_FW1[1] - self.v_star)
        state_FW2[1] = - (state_FW2[1] - self.v_star)
        state_FW1[2] = state_FW1[2] - self.v_star
        state_FW2[2] = state_FW2[2] - self.v_star

        if self.SIDE_update:
            next_state_FW1 = next_state[[self.FV1_idx, self.FV1_idx + 4, self.FV1_idx + 4 - 1]]
            next_state_FW2 = next_state[[self.FV2_idx, self.FV2_idx + 4, self.FV2_idx + 4 - 1]]
            next_state_FW1[0] = next_state_FW1[0] - self.s_star
            next_state_FW2[0] = next_state_FW2[0] - self.s_star
            next_state_FW1[1] = - (next_state_FW1[1] - self.v_star)
            next_state_FW2[1] = - (next_state_FW2[1] - self.v_star)
            next_state_FW1[2] = next_state_FW1[2] - self.v_star
            next_state_FW2[2] = next_state_FW2[2] - self.v_star
            
            
            self.SIDE_FV1.step(state_FW1, acceleration[self.FV1_idx+1], next_state_FW1)
            self.SIDE_FV2.step(state_FW2, acceleration[self.FV2_idx+1], next_state_FW2)

            self.FW1_parameters = self.SIDE_FV1.car_following_model_parameters()
            self.FW2_parameters = self.SIDE_FV2.car_following_model_parameters()
            
        if self.filter_update:
            self.filt_1.adapt(acceleration[self.FV1_idx+1], state_FW1)
            self.filt_2.adapt(acceleration[self.FV2_idx+1], state_FW2)

            if self.filt_enable:
                self.FW1_parameters = self.filt_1.w
                self.FW2_parameters = self.filt_2.w

        self.actor.safeLayer.FW1_parameters = self.FW1_parameters
        self.actor.safeLayer.FW2_parameters = self.FW2_parameters


    def lr_decay(self, total_steps):
        # Learning rate decay
        lr_a_current = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_current = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_current
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_current

        # self.barrier_optimizer_FV1.lr_current = self.barrier_optimizer_FV1.lr_initial * (1 - total_steps / self.max_train_steps)
        # self.barrier_optimizer_FV2.lr_current  = self.barrier_optimizer_FV2.lr_initial * (1 - total_steps / self.max_train_steps)

    def save(self, checkpoint_path, epsilon_number):
        # Save checkpoint
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if self.safety_layer_enabled and not self.nn_cbf_enabled:
            torch.save(self.actor.state_dict(), os.path.join(checkpoint_path, 'ppo_actor_episode_' + str(epsilon_number) + '.pth'))
            torch.save(self.critic.state_dict(), os.path.join(checkpoint_path, 'ppo_critic_episode_' + str(epsilon_number) + '.pth'))
        elif self.safety_layer_enabled and self.nn_cbf_enabled:
            torch.save(self.actor.state_dict(), os.path.join(checkpoint_path, 'ppo_actor_episode_' + str(epsilon_number) + '_nn_cbf.pth'))
            torch.save(self.critic.state_dict(), os.path.join(checkpoint_path, 'ppo_critic_episode_' + str(epsilon_number) + '_nn_cbf.pth'))
        else:
            torch.save(self.actor.state_dict(), os.path.join(checkpoint_path, 'ppo_actor_episode_' + str(epsilon_number) + '_no_safety.pth'))
            torch.save(self.critic.state_dict(), os.path.join(checkpoint_path, 'ppo_critic_episode_' + str(epsilon_number) + '_no_safety.pth'))

    def load(self, checkpoint_path, epsilon_number):
        # Load checkpoint
        if self.safety_layer_enabled  and not self.nn_cbf_enabled:
            self.actor.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ppo_actor_episode_' + str(epsilon_number) + '.pth')))
            self.critic.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ppo_critic_episode_' + str(epsilon_number) + '.pth')))
        elif not self.safety_layer_enabled and self.safety_layer_no_grad:
            self.actor.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ppo_actor_episode_' + str(epsilon_number) + '_no_safety.pth')))
            self.critic.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ppo_critic_episode_' + str(epsilon_number) + '_no_safety.pth')))
        elif self.safety_layer_enabled and self.nn_cbf_enabled:
            self.actor.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ppo_actor_episode_' + str(epsilon_number) + '_nn_cbf.pth')))
            self.critic.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ppo_critic_episode_' + str(epsilon_number) + '_nn_cbf.pth')))
        else:
            self.actor.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ppo_actor_episode_' + str(epsilon_number) + '_no_safety.pth')))
            self.critic.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ppo_critic_episode_' + str(epsilon_number) + '_no_safety.pth')))


if __name__ == '__main__':
    # Create the environment
    env = PlatoonEnv()
    # Set the device
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    # Set the safety layer
    safety_layer_enabled = True
    # Select the agent
    agent_select = 'ppo'
    # Set if train the agent
    agent_train = True
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
    parser.add_argument("--cbf_tau", type=float, default=0.3, help="CAV index in the platoon")
    parser.add_argument("--cbf_gamma", type=float, default=1, help="CAV index in the platoon")
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
    parser.add_argument("--state_size_nncbf", type=float, default=12, help="CAV index in the platoon")
    parser.add_argument("--hidden_size_nncbf", type=float, default=100, help="CAV index in the platoon")
    parser.add_argument("--output_size_nncbf", type=float, default=1, help="CAV index in the platoon")
    args = parser.parse_args()
    args.device = device
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = 5.0
    args.max_episode_steps = env.max_steps
    agent = PPOAgent(args)

    state = env.reset()
    action, action_prob = agent.act(state)
    print(agent.actor)
    #make_dot(action, params=dict(list(agent.actor.named_parameters()))).render("attached", format="png")
