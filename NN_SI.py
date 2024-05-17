import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from utils import ReplayBuffer_SIDE

class OVM_Estimator(nn.Module):
    def __init__(self):
        super(OVM_Estimator,self).__init__()
        self.alpha1 = nn.Parameter(torch.tensor(1., requires_grad=True)) #0.5
        self.alpha2 = nn.Parameter(torch.tensor(1., requires_grad=True)) #0.5
        self.alpha3 = nn.Parameter(torch.tensor(1., requires_grad=True))
        
    def forward(self,x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        #a = self.k*(x[:,2]-0.5*self.vmax*self.tanh(x[:,0]-self.hc)+self.tanh(self.hc))
        a = self.alpha1*x[:,0] + self.alpha2*x[:,1] + self.alpha3*x[:,2]
        return a.unsqueeze(1)
    

class Disturbance_Estimator(nn.Module):
    def __init__(self, state_num, action_num):
        super(Disturbance_Estimator,self).__init__()
        self.fc1 = nn.Linear(state_num + action_num, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)
        self.tanh = nn.Tanh()
        
    def forward(self,x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Disturbance_Estimator_LSTM(nn.Module):

    def __init__(self, num_classes=1, input_size = 3, hidden_size = 2, num_layers = 1):
        super(Disturbance_Estimator_LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

        self.apply(self.weight_init)
        
    def weight_init(self,m):
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias,0)
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=x.device))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=x.device))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out
    
class NN_SI_DE_Module():
    def __init__(self, state_num, action_num, lr_cf, lr_de, batch_size, buffer_size, device, veh_idx):
        self.state_num = state_num
        self.action_num = action_num
        self.lr_cf = lr_cf
        self.lr_de = lr_de
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.veh_idx = veh_idx

        self.car_following_estimator = OVM_Estimator().to(self.device)
        self.disturbance_estimator = Disturbance_Estimator(self.state_num, self.action_num).to(self.device)

        self.optimizer_cf = torch.optim.Adam(self.car_following_estimator.parameters(), lr = self.lr_cf)
        self.optimizer_de = torch.optim.Adam(self.disturbance_estimator.parameters(), lr = self.lr_de)

        self.replay_buffer = ReplayBuffer_SIDE(self.buffer_size, self.batch_size)

        self.loss_de_lst = []
        self.loss_cf_lst = []
        self.dt = 0.1

        self.s_star = 20
        self.v_star = 15

    def step(self, state, action, next_state):
        self.replay_buffer.store(state, action, next_state)
        if self.replay_buffer.count > self.batch_size:
            self.learn()

    def learn(self):
        state, action, next_state = self.replay_buffer.sample()
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        self.optimizer_cf.zero_grad()
        self.optimizer_de.zero_grad()

        action_pred = self.car_following_estimator(state)
        action_disturbance = self.disturbance_estimator(torch.cat((state, action_pred.detach().cpu()), 1))

        next_state_pred_wo_de = self._get_next_state(state, action_pred)
        next_state_pred_w_de = self._get_next_state(state, action_pred.detach().cpu() + action_disturbance)

        # loss_cf = F.mse_loss(action_pred, action)
        loss_cf = F.mse_loss(next_state_pred_wo_de, -next_state[:,1])
        loss_cf.backward()
        self.optimizer_cf.step()

        loss_de = F.mse_loss(next_state_pred_w_de, -next_state[:,1])
        loss_de.backward()
        self.optimizer_de.step()

        self.loss_cf_lst.append(loss_cf.item())
        self.loss_de_lst.append(loss_de.item())


    def _get_disturbance_estimation(self, state):
        state_FW = state[:,[self.veh_idx, self.veh_idx + 4, self.veh_idx + 4 - 1]]
        
        state_FW[:,0] = state_FW[:,0] - self.s_star
        state_FW[:,1] = - (state_FW[:,1] - self.v_star)
        state_FW[:,2] = state_FW[:,2] - self.v_star

        state = torch.tensor(state_FW, dtype=torch.float).to(self.device)
        action_pred = self.car_following_estimator(state)
        action_pred = torch.tensor(action_pred, dtype=torch.float).to(self.device)
        disturbance = self.disturbance_estimator(torch.cat((state, action_pred), 1))
        return disturbance.detach().cpu().squeeze()
    
    def _get_disturbance_estimation_2(self, new_input):
        new_input = torch.tensor(new_input, dtype=torch.float).to(self.device)
        disturbance = self.disturbance_estimator(new_input)

        return disturbance.detach().cpu().squeeze()
    
    def _get_car_following_estimation(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action_pred = self.car_following_estimator(state)
        return action_pred.detach().cpu().numpy()
    
    def _get_next_state(self,state, action):
        # state[:, 0] = state[:, 0] + self.dt * (state[:, 2] - state[:, 1])

        next_state = -state[:, 1] + self.dt * (action.squeeze(1))
        return next_state
    
    def get_next_state_with_disturbance_estimation(self, state):
        action_pred = self._get_car_following_estimation(state)
        disturbance = self._get_disturbance_estimation(state, action_pred)
        action_pred = action_pred + disturbance
        next_state = self._get_next_state(state, action_pred)
        return next_state
    
    def car_following_model_parameters(self):
        return [self.car_following_estimator.alpha1.cpu().detach().numpy().tolist(), self.car_following_estimator.alpha2.cpu().detach().numpy().tolist(), self.car_following_estimator.alpha3.cpu().detach().numpy().tolist()]
    
    def save_model(self, path):
        torch.save(self.car_following_estimator.state_dict(), path + 'car_following_estimator.pth')
        torch.save(self.disturbance_estimator.state_dict(), path + 'disturbance_estimator.pth')

    def load_model(self, path):
        try:
            self.car_following_estimator.load_state_dict(torch.load(path + 'car_following_estimator.pth'))
            self.disturbance_estimator.load_state_dict(torch.load(path + 'disturbance_estimator.pth'))
        except:
            print('error loading')