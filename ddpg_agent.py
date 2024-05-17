import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Actor, Critic
from utils import ReplayBuffer, OUNoise
from NeuralBarrier import *
import os


class DDPGAgent:
    def __init__(self, args):
        # Initialize an Agent object.
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.device = args.device
        self.safety_layer_enabled = args.safety_layer_enabled
        self.lr_a = args.lr_actor
        self.lr_c = args.lr_critic
        self.max_train_steps = args.max_train_steps
        self.actor_eval = Actor(args).to(args.device)
        self.actor_target = Actor(args).to(args.device)
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=args.lr_actor)
        self.CAV_index = args.CAV_idx
        self.car_following_parameters = [1.2566, 1.5000, 0.9000]
        self.cbf_tau = args.cbf_tau
        
        self.critic_eval = Critic(args).to(args.device)
        self.critic_target = Critic(args).to(args.device)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=args.lr_critic)

        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.action_noise = OUNoise(args.action_dim)
        # self.barrier_optimizer_CAV = barrier_optimizer(args.state_dim, args.hidden_size_nncbf, args.output_size_nncbf, 60*args.batch_size, args.Lf_CAV, args.Lg_CAV, args.gamma, args.tau, args.CAV_idx, args.CAV_idx, args.dt, args.lr_cbf, 60*args.batch_size, args.device)
        self.barrier_optimizer_FV1 = barrier_optimizer(args.state_dim, args.hidden_size_nncbf, args.output_size_nncbf, 60*args.batch_size, args.Lf_FV1, args.Lg_FV1, args.gamma, args.cbf_tau, args.CAV_idx, args.FV1_idx, args.dt, args.lr_cbf, 60*args.batch_size, args.device)
        self.barrier_optimizer_FV2 = barrier_optimizer(args.state_dim, args.hidden_size_nncbf, args.output_size_nncbf, 60*args.batch_size, args.Lf_FV2, args.Lg_FV2, args.gamma, args.cbf_tau, args.CAV_idx, args.FV2_idx, args.dt, args.lr_cbf, 60*args.batch_size, args.device)

    def step(self, state, action, action_safe, reward, next_state, done, total_steps):
        # Save experience in replay memory
        self.replay_buffer.add(state, action, action_safe, reward, next_state, done)

        s_i, v_i = state[4*self.CAV_index], state[4*self.CAV_index+1]                               # CAV's spacing and velocity
        s_im, v_im = state[4*(self.CAV_index-1)], state[4*(self.CAV_index-1)+1]           # CAV front vehicle's spacing and velocity
        s_f_1, v_f_1 = state[4*(self.CAV_index+1)], state[4*(self.CAV_index+1)+1]                               # following vehicle 1's spacing and velocity
        s_f_im1, v_f_im1 = state[4*self.CAV_index], state[4*self.CAV_index+1]           # front vehicle's spacing and velocity
        s_f_2, v_f_2 = state[4*(self.CAV_index+2)], state[4*(self.CAV_index+2)+1]                               # following vehicle 2's spacing and velocity
        s_f_im2, v_f_im2 = state[4*(self.CAV_index+1)], state[4*(self.CAV_index+1)+1]           # front vehicle's spacing and velocity
            
        s_i_ls = [s_i, s_f_1, s_f_2]
        v_i_ls = [v_i, v_f_1, v_f_2]
        v_im_ls = [v_im, v_f_im1, v_f_im2]
        # self.barrier_optimizer_CAV.Lf = (v_im1 - v_i)
        # self.barrier_optimizer_CAV.Lg = -self.barrier_optimizer_CAV.tau
        # self.barrier_optimizer_CAV.step_optimize(state, action)
        v_star   = 15
        s_star   = 20
        self.barrier_optimizer_FV1.Lg = -self.barrier_optimizer_FV1.tau
        self.barrier_optimizer_FV1.Lf = (v_im_ls[0] +  v_i_ls[1] - 2 * v_i_ls[0] + self.cbf_tau*(self.car_following_parameters[0]*(s_i_ls[1]+s_star) - self.car_following_parameters[1]*(v_i_ls[1]+v_star) + self.car_following_parameters[2]*(v_im_ls[1]+v_star)))
        self.barrier_optimizer_FV2.Lg = -self.barrier_optimizer_FV2.tau

        self.barrier_optimizer_FV1.step_optimize(state, action)
        self.barrier_optimizer_FV2.step_optimize(state, action)

        # Learn every UPDATE_EVERY time steps.
        if len(self.replay_buffer) >= self.batch_size:
            experiences = self.replay_buffer.sample(self.batch_size)
            self.learn(experiences)

    def act(self, state, add_noise=True):
        # Generate an action from the actor
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_eval.eval()
        
        with torch.no_grad():
            # La_CAV, Lb_CAV = self.barrier_optimizer_CAV.compensator(state)
            # La_FV1, Lb_FV1 = self.barrier_optimizer_FV1.compensator(state)
            # La_FV2, Lb_FV2 = self.barrier_optimizer_FV2.compensator(state)
            La_FV1 = self.barrier_optimizer_FV1.compensator(state)
            La_FV2 = self.barrier_optimizer_FV2.compensator(state)
            action, action_safe = self.actor_eval(state, La_FV1, La_FV2)
            action = action.cpu().data.numpy()
            action_safe = action_safe.cpu().data.numpy()
        self.actor_eval.train()
        if add_noise:
            action += self.action_noise.sample()
        return action, action_safe

    def reset_noise(self):
        self.action_noise.reset()

    def learn(self, experiences):
        # Update the critic and actor parameters
        states, actions, actions_safe, rewards, next_states, dones = experiences

        # Set device for training
        states = states.to(self.device)
        actions = actions.to(self.device)
        actions_safe = actions_safe.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update the critic
        # La_CAV, Lb_CAV = self.barrier_optimizer_CAV.compensator(next_states)
        # La_FV1, Lb_FV1 = self.barrier_optimizer_FV1.compensator(next_states)
        # La_FV2, Lb_FV2 = self.barrier_optimizer_FV2.compensator(next_states)
        La_FV1 = self.barrier_optimizer_FV1.compensator(next_states)
        La_FV2 = self.barrier_optimizer_FV2.compensator(next_states)
        next_actions, next_actions_safe = self.actor_target(next_states, La_FV1, La_FV2)
        Q_targets_next = self.critic_target(next_states, next_actions_safe)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_eval(states, actions_safe)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the actor
        # La_CAV, Lb_CAV = self.barrier_optimizer_CAV.compensator(next_states)
        # La_FV1, Lb_FV1 = self.barrier_optimizer_FV1.compensator(next_states)
        # La_FV2, Lb_FV2 = self.barrier_optimizer_FV2.compensator(next_states)
        La_FV1 = self.barrier_optimizer_FV1.compensator(next_states)
        La_FV2 = self.barrier_optimizer_FV2.compensator(next_states)
        actions_pred, actions_pred_safe = self.actor_eval(states, La_FV1, La_FV2)
        actor_loss = -self.critic_eval(states, actions_pred_safe).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft-update the target networks
        self.soft_update(self.actor_eval, self.actor_target, self.tau)
        self.soft_update(self.critic_eval, self.critic_target, self.tau)


    def lr_decay(self, total_steps):
        # Decay learning rate
        lr_a_current = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_current = self.lr_c * (1 - total_steps / self.max_train_steps)
        self.barrier_optimizer_FV1.lr_current = self.barrier_optimizer_FV1.lr_initial * (1 - total_steps / self.max_train_steps)
        self.barrier_optimizer_FV2.lr_current  = self.barrier_optimizer_FV2.lr_initial * (1 - total_steps / self.max_train_steps)

        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_current
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_c_current
        


    def soft_update(self, eval_model, target_model, tau):
        # Soft update model parameters
        for target_param, eval_param in zip(target_model.parameters(), eval_model.parameters()):
            target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

    def save(self, checkpoint_path, epsilon_number):
        # Save the model checkpoint
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if self.safety_layer_enabled:
            torch.save(self.actor_eval.state_dict(), os.path.join(checkpoint_path, 'ddpg_actor_episode_' + str(epsilon_number) + '.pth'))
            torch.save(self.critic_eval.state_dict(), os.path.join(checkpoint_path, 'ddpg_critic_episode_' + str(epsilon_number) + '.pth'))
        else:
            torch.save(self.actor_eval.state_dict(), os.path.join(checkpoint_path, 'ddpg_actor_episode_' + str(epsilon_number) + '_no_safety.pth'))
            torch.save(self.critic_eval.state_dict(), os.path.join(checkpoint_path, 'ddpg_critic_episode_' + str(epsilon_number) + '_no_safety.pth'))

    def load(self, checkpoint_path, epsilon_number):
        # Load the model checkpoint
        if self.safety_layer_enabled:
            self.actor_eval.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ddpg_actor_episode_' + str(epsilon_number) + '.pth')))
            self.critic_eval.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ddpg_critic_episode_' + str(epsilon_number) + '.pth')))
        else:
            self.actor_eval.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ddpg_actor_episode_' + str(epsilon_number) + '_no_safety.pth')))
            self.critic_eval.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ddpg_critic_episode_' + str(epsilon_number) + '_no_safety.pth')))