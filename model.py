import torch
import torch.nn as nn
import torch.nn.functional as F
from BarrierNet import BarrierLayer


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()

        # Initialize the actor network
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dims[0])
        self.fc2 = nn.Linear(args.hidden_dims[0], args.hidden_dims[1])
        self.fc3 = nn.Linear(args.hidden_dims[1], args.action_dim)
        
        # Initialize the safety layer
        self.safeLayer = None
        if args.safety_layer_enabled:
            self.safeLayer = BarrierLayer(args.state_dim)
        self.CAV_index = args.CAV_idx
        self.cbf_tau = args.cbf_tau
        self.cbf_gamma = args.cbf_gamma

    def forward(self, state, La_FV1, La_FV2):
        # Generate an action from the actor
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        u = torch.tanh(self.fc3(x))

        # Generate a safe action from the safety layer
        if self.safeLayer is not None:
            u_safe = self.safeLayer(u, state, self.cbf_tau, self.cbf_gamma, self.CAV_index, La_FV1, La_FV2)
            return u, u_safe+u
        else:
            return u, u


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()

        # Initialize the critic network
        self.fc1 = nn.Linear(args.state_dim + args.action_dim, args.hidden_dims[0])
        self.fc2 = nn.Linear(args.hidden_dims[0], args.hidden_dims[1])
        self.fc3 = nn.Linear(args.hidden_dims[1], 1)

    def forward(self, state, action):
        # Generate a Q-value from the critic
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
