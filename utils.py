import numpy as np
import random
from collections import deque, namedtuple
import torch

class ReplayBuffer:
    def __init__(self, buffer_size=int(1e5), batch_size=64):
        # Initialize a ReplayBuffer object (for ddpg).
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "action_safe", "reward", "next_state", "done"])

    def add(self, state, action, action_safe, reward, next_state, done):
        # Add a new experience to memory.
        e = self.experience(state, action, action_safe, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=None):
        # Randomly sample a batch of experiences from memory.
        if batch_size is None:
            batch_size = self.batch_size
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float()
        actions_safe = torch.from_numpy(np.vstack([e.action_safe for e in experiences])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        return (states, actions, actions_safe, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class ReplayBuffer_Compensator:
    def __init__(self, buffer_size=int(1e5), batch_size=64):
        # Initialize a ReplayBuffer object (for Compensator).
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "control_input", "h_derivative_true", "Lf"])

    def add(self, state, control_input, h_derivative_true, Lf):
        # Add a new experience to memory.
        e = self.experience(state, control_input, h_derivative_true, Lf)
        self.memory.append(e)

    def sample(self, batch_size=None):
        # Randomly sample a batch of experiences from memory.
        if batch_size is None:
            batch_size = self.batch_size
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        control_input = torch.from_numpy(np.vstack([e.control_input for e in experiences])).float()
        h_derivative_true = torch.from_numpy(np.vstack([e.h_derivative_true for e in experiences])).float()
        Lf = torch.from_numpy(np.vstack([e.Lf for e in experiences])).float()

        return (states, control_input, h_derivative_true, Lf)

    def __len__(self):
        return len(self.memory)


class ReplayBuffer_PPO:
    def __init__(self, args):
        # Initialize a ReplayBuffer_PPO object.
        self.state = np.zeros((args.batch_size, args.state_dim))
        self.action = np.zeros((args.batch_size, args.action_dim))
        self.action_logprob = np.zeros((args.batch_size, args.action_dim))
        self.reward = np.zeros((args.batch_size, 1))
        self.state_ = np.zeros((args.batch_size, args.state_dim))
        self.done = np.zeros((args.batch_size, 1))
        self.acceleration = np.zeros((args.batch_size, args.vehicle_num))
        self.count = 0

    def store(self, state, action, action_logprob, reward, state_, done, acceleration):
        # Store the transition in the replay buffer
        self.state[self.count] = state
        self.action[self.count] = action
        self.action_logprob[self.count] = action_logprob
        self.reward[self.count] = reward
        self.state_[self.count] = state_
        self.done[self.count] = done
        self.acceleration[self.count] = acceleration
        self.count += 1

    def numpy_to_tensor(self):
        # Convert numpy array to torch tensor and return
        state = torch.tensor(self.state, dtype=torch.float)
        action = torch.tensor(self.action, dtype=torch.float)
        action_logprob = torch.tensor(self.action_logprob, dtype=torch.float)
        reward = torch.tensor(self.reward, dtype=torch.float)
        state_ = torch.tensor(self.state_, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        acceleration = torch.tensor(self.acceleration, dtype=torch.float)
        return state, action, action_logprob, reward, state_, done, acceleration
    
class ReplayBuffer_SIDE:
    def __init__(self, buffer_size=int(1e5), batch_size=64):
        # Initialize a ReplayBuffer object (for SIDE).
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "next_state"])
        self.count = 0

    def store(self, state, action, next_state):
        # Add a new experience to memory.
        e = self.experience(state, action, next_state)
        self.memory.append(e)
        self.count += 1

    def sample(self, batch_size=None):
        # Randomly sample a batch of experiences from memory.
        if batch_size is None:
            batch_size = self.batch_size
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        return (states, actions, next_states)

class OUNoise:
    def __init__(self, action_size, mu=0.0, theta=0.15, sigma=0.2):
        # Initialize parameters and noise process.
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_size) * self.mu
        self.reset()

    def sample(self):
        # Update internal state and return it as a noise sample.
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def reset(self):
        # Reset the internal state (= noise) to mean (mu).
        self.state = np.ones(self.action_size) * self.mu

def to_numpy(x):
    # convert torch tensor to numpy array
    return x.cpu().detach().double().numpy()

def to_tensor(x, dtype, device, requires_grad=False):
    # convert numpy array to torch tensor
    return torch.from_numpy(x).type(dtype).to(device).requires_grad_(requires_grad)
