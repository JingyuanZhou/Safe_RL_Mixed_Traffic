import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import *
from qpth.qp import QPFunction, QPSolvers

eps = 1e-7

class BarrierLayer(nn.Module):
    def __init__(self, states_size, car_following_parameters = [1.2566, 1.5000, 0.9000], safety_layer_no_grad = False, SIDE_enabled = False):
        super(BarrierLayer, self).__init__()
        # Initialize the unused parameters
        self.e = Variable(torch.Tensor())
        self.states_size = states_size
        self.following_veh = 2
        if safety_layer_no_grad:
            self.gamma = torch.tensor([1.0,1.0,1.0])#2*torch.ones(self.following_veh + 1)
            self.k1 = torch.tensor([1.0])
        else:
            self.gamma = nn.Parameter(torch.tensor([1.0,1.0,1.0]), requires_grad=True)
            self.k1 = nn.Parameter(torch.tensor([10.0]), requires_grad=True)
        # self.car_following_parameters = car_following_parameters

        self.FW1_parameters = car_following_parameters
        self.FW2_parameters = car_following_parameters

        self.SIDE_enabled = SIDE_enabled

    def forward(self, u_nominal, states, tau, gamma, CAV_index, La_FV1 = None, La_FV2 = None, Learning_CBF = False, acceleration = None, cf_saturation_FW1 = None, cf_saturation_FW2 = None):
        
        # Safety ahead constraint
        
        v_star   = 15
        s_star   = 20
        bias = 0.
        if_batch = states.dim() > 1 # check if batch

        if if_batch:
            
            batch_size = states.size(dim=0)
            # define the parameters of the CBF constraint

            # bacth Q matrix
            self.Q = torch.zeros(batch_size, self.following_veh + 1 , self.following_veh + 1)
            Q_weight = [1, 1, 1]
            for i in range(0, self.following_veh + 1):
                self.Q [:,i,i] = Q_weight[i]

            # batch p vector
            self.p = torch.zeros(batch_size, self.following_veh + 1)
            #self.p [:,0] = -u_nominal.squeeze(1)

            # batch states
            s_i, v_i = states[:, CAV_index], states[:, CAV_index+4]                               # CAV's spacing and velocity
            s_im, v_im = states[:, CAV_index-1], states[:, CAV_index+4-1]           # CAV front vehicle's spacing and velocity
            s_f_1, v_f_1 = states[:, CAV_index + 1], states[:, CAV_index+4+1]                               # following vehicle 1's spacing and velocity
            s_f_im1, v_f_im1 = states[:, CAV_index], states[:, CAV_index+4]           # front vehicle's spacing and velocity
            s_f_2, v_f_2 = states[:, CAV_index + 2], states[:, CAV_index+4+2]                               # following vehicle 2's spacing and velocity
            s_f_im2, v_f_im2 = states[:, CAV_index + 1], states[:, CAV_index+4+1]           # front vehicle's spacing and velocity
            
            s_i_ls = [s_i, s_f_1, s_f_2]
            v_i_ls = [v_i, v_f_1, v_f_2]
            v_im_ls = [v_im, v_f_im1, v_f_im2]

            # batch lie derivatives
            if Learning_CBF == False:
                La_FV1 = torch.zeros(batch_size, 1)
                La_FV2 = torch.zeros(batch_size, 1)
            
            if self.SIDE_enabled == False:
                cf_saturation_FW1 = torch.zeros(batch_size)
                cf_saturation_FW2 = torch.zeros(batch_size)

            Lfh1 = (v_im_ls[0] - v_i_ls[0]).unsqueeze(1) #+ La_CAV.detach()
            Lfh2 =(- v_im_ls[0] +  v_i_ls[0] + v_im_ls[1] - v_i_ls[1] - tau*(self.FW1_parameters[0]*(s_i_ls[1]-s_star) - self.FW1_parameters[1]*(v_i_ls[1]-v_star) + self.FW1_parameters[2]*(v_im_ls[1]-v_star) + cf_saturation_FW1)).unsqueeze(1) + La_FV1.detach()
            Lfh3 =(- v_im_ls[0] +  v_i_ls[0] + v_im_ls[2] - v_i_ls[2] - tau*(self.FW2_parameters[0]*(s_i_ls[2]-s_star) - self.FW2_parameters[1]*(v_i_ls[2]-v_star) + self.FW2_parameters[2]*(v_im_ls[2]-v_star) + cf_saturation_FW2)).unsqueeze(1) + La_FV2.detach()
            Lfh_ls = torch.hstack([Lfh1, Lfh2, Lfh3])
            Lgh_ls = torch.hstack([-tau * torch.ones(batch_size,1), tau * torch.ones(batch_size,1), tau * torch.ones(batch_size,1)]) #Lb_CAV.detach()
            #Lfh = (v_im - v_i).unsqueeze(1) #+ La.detach()
            #Lgh = -tau #+ Lb.detach()

            alpha_h_1 = (self.gamma[0]*(s_i - tau * v_i).pow(1) - bias).unsqueeze(1)
            alpha_h_2 = (self.gamma[1]*(s_f_1 - s_i - tau * (v_f_1 - v_i)).pow(1)).unsqueeze(1)
            alpha_h_3 = (self.gamma[2]*(s_f_2 - s_i - tau * (v_f_2 - v_i)).pow(1)).unsqueeze(1)
            alpha_h_ls = torch.hstack([alpha_h_1, alpha_h_2, alpha_h_3])
            nominal_part = u_nominal.mul(Lgh_ls)

            # Feasible acceleration constraint
            if acceleration is not None:
                u_min = -5
                control_bound = acceleration[:, CAV_index - 1 + 1] + self.k1*(v_im_ls[0] - v_i_ls[0] - tau*u_min) - u_nominal.squeeze()
                # control_bound = 10000*torch.ones(batch_size)

            # batch G matrix
            self.G = torch.zeros(batch_size, self.following_veh+2, self.following_veh+1) # batch size, number of constraints, number of variables
            #for i in range(0, self.following_veh+1):

            self.G [:,0:self.following_veh+1,0] = - Lgh_ls
            self.G [:,1,1] = - 1
            self.G [:,2,2] = - 1
            # feasibility
            self.G [:,self.following_veh+1,0] = 1
            
            # batch h vector
            self.h = torch.zeros(batch_size, self.following_veh+2)
            self.cbf_h = (alpha_h_ls+Lfh_ls+nominal_part)
            self.h[:,0:self.following_veh+1] = self.cbf_h
            self.h[:,self.following_veh+1] = control_bound
            
            # print(self.Q, self.p, self.G, self.h)
            # calculate the CBF constraint with QP solvers in batch
            u_ = QPFunction()(self.Q.float(), self.p.float(),
                            self.G.float(), self.h.float(), self.e, self.e).float()

            # return the first column of the solution
            return u_[:,0].unsqueeze(1)
        
        else:
            # define the parameters of the CBF constraint

            # Q matrix
            self.Q = torch.zeros(self.following_veh + 2 , self.following_veh + 2)
            Q_weight = [1, 1, 1]
            for i in range(0, self.following_veh + 1):
                self.Q [i,i] = Q_weight[i]
            
            # p vector
            self.p = torch.zeros(self.following_veh + 1)
            # self.p [0] = -u_nominal[0]

            # states
            s_i, v_i = states[CAV_index], states[CAV_index+4]                               # CAV's spacing and velocity
            s_im, v_im = states[CAV_index-1], states[CAV_index+4-1]           # CAV front vehicle's spacing and velocity
            s_f_1, v_f_1 = states[CAV_index + 1], states[CAV_index+4+1]                               # following vehicle 1's spacing and velocity
            s_f_im1, v_f_im1 = states[CAV_index], states[CAV_index+4]           # front vehicle's spacing and velocity
            s_f_2, v_f_2 = states[CAV_index + 2], states[CAV_index+4+2]                               # following vehicle 2's spacing and velocity
            s_f_im2, v_f_im2 = states[CAV_index + 1], states[CAV_index+4+1]           # front vehicle's spacing and velocity

            s_i_ls = [s_i, s_f_1, s_f_2]
            v_i_ls = [v_i, v_f_1, v_f_2]
            v_im_ls = [v_im, v_f_im1, v_f_im2]

            if Learning_CBF == False:
                La_FV1 = torch.tensor(0)
                La_FV2 = torch.tensor(0)

            if self.SIDE_enabled == False:
                cf_saturation_FW1 = torch.zeros(0)
                cf_saturation_FW2 = torch.zeros(0)

            # lie derivatives
            Lfh1 = (v_im_ls[0] - v_i_ls[0])
            Lfh2 =(- v_im_ls[0] +  v_i_ls[0] + v_im_ls[1] - v_i_ls[1] - tau*(self.car_following_parameters[0]*(s_i_ls[1]-s_star) - self.car_following_parameters[1]*(v_i_ls[1]-v_star) + self.car_following_parameters[2]*(v_im_ls[1]-v_star) + cf_saturation_FW1)) + La_FV1.detach()
            Lfh3 =(- v_im_ls[0] +  v_i_ls[0] + v_im_ls[2] - v_i_ls[2] - tau*(self.car_following_parameters[0]*(s_i_ls[2]-s_star) - self.car_following_parameters[1]*(v_i_ls[2]-v_star) + self.car_following_parameters[2]*(v_im_ls[2]-v_star) + cf_saturation_FW2)) + La_FV2.detach()
            Lfh_ls = torch.hstack([Lfh1, Lfh2, Lfh3])
            Lgh_ls = torch.tensor([-tau, tau, tau])

            # Feasible acceleration constraint
            if acceleration is not None:
                u_min = -5
                control_bound = acceleration[CAV_index - 1 + 1] + self.k1*(v_im_ls[0] - v_i_ls[0] - tau*u_min) - u_nominal
                # control_bound = 10000*torch.ones(1)

            alpha_h_1 = (self.gamma[0]*(s_i - tau * v_i).pow(1) - bias)
            alpha_h_2 = (self.gamma[1]*(s_f_1 - s_i - tau * (v_f_1 - v_i)).pow(1))
            alpha_h_3 = (self.gamma[2]*(s_f_2 - s_i - tau * (v_f_2 - v_i)).pow(1))
            alpha_h_ls = torch.hstack([alpha_h_1, alpha_h_2, alpha_h_3])
            nominal_part = u_nominal.mul(Lgh_ls)
            # batch G matrix
            self.G = torch.zeros(self.following_veh+2, self.following_veh+1) # number of constraints, number of variables
            self.G [0:self.following_veh+1,0] = - Lgh_ls
            self.G [1,1] = - 1
            self.G [2,2] = - 1
            # feasibility
            self.G [self.following_veh+1,0] = 1
            
            # h vector
            self.h = torch.zeros(self.following_veh+2)
            self.cbf_h = (alpha_h_ls+Lfh_ls+nominal_part)
            self.h[0:self.following_veh+1] = self.cbf_h
            self.h[self.following_veh+1] = control_bound

            # calculate the CBF constraint with QP solvers
            u_ = QPFunction()(self.Q.float(), self.p.float(),
                            self.G.float(), self.h.float(), self.e, self.e).float()
            # return the first column of the solution
            return u_[:,0]

if __name__ == "__main__":
    # test
    x = torch.tensor([[0, 10, 0, 0, 0, 10, 0, 0, 5, 10, 0, 0, 15, 20, 0, 0],[0, 10, 0, 0, 0, 10, 0, 0, 5, 10, 0, 0, 15, 20 ,0 ,0]])
    x = torch.tensor([0, 10, 0, 0, 0, 10, 0, 0, 5, 10, 0, 0, 15, 20, 0, 0])
    CAV_index = 1
    u_nominal = torch.tensor([[3],[4]])
    u_nominal = torch.tensor([4])
    tau = 0.3
    gamma = 1
    BN_model = BarrierLayer(8)
    output = BN_model(u_nominal, x, tau, gamma, CAV_index)
    print(output.size())
    print(u_nominal.size())
    print(output + u_nominal)
