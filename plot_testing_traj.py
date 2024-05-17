# plot the trajectory of the vehicles in different scenarios
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys
plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['mathtext.default'] = 'regular'

# read the data from the file
def read_data(filename):
    data = np.load(filename)
    return data

vis_hx = True
tau = 0.3

for scenario in [2,3,4]:
    data_w_safety_layer = read_data('results\ppo_test_rl_traj_no_nn_cbf_scenario_'+str(scenario)+'_SI_enabled_True.npy')
    data_w_safety_layer_wo_si = read_data('results\ppo_test_rl_traj_no_nn_cbf_scenario_'+str(scenario)+'_SI_enabled_False.npy')
    data_wo_safety_layer = read_data('results\ppo_test_rl_traj_no_safe_scenario_'+str(scenario)+'.npy')
    data_pure_car_following = read_data('results\\test_cf_traj_scenario_'+str(scenario)+'.npy')
    # results\test_cf_traj_scenario_4.npy
    if scenario == 1 or scenario == 4:
        spacing_w_safety_layer = data_w_safety_layer[:,5+2]
        # spacing_w_safety_layer_wo_si = data_w_safety_layer_wo_si[:,5+2]
        spacing_wo_safety_layer = data_wo_safety_layer[:,5+2]
        spacing_pure_car_following = data_pure_car_following[:,5+2]

        velocity_w_safety_layer = data_w_safety_layer[:,2]
        velocity_wo_safety_layer = data_wo_safety_layer[:,2]
        velocity_pure_car_following = data_pure_car_following[:,2]

        hx_w_safety_layer = spacing_w_safety_layer - velocity_w_safety_layer*tau
        hx_wo_safety_layer = spacing_wo_safety_layer - velocity_wo_safety_layer*tau
        hx_pure_car_following = spacing_pure_car_following - velocity_pure_car_following*tau

    elif scenario == 2:
        spacing_w_safety_layer = data_w_safety_layer[:,5+3]
        spacing_w_safety_layer_wo_si = data_w_safety_layer_wo_si[:,5+3]
        spacing_wo_safety_layer = data_wo_safety_layer[:,5+3]
        spacing_pure_car_following = data_pure_car_following[:,5+3]

        velocity_w_safety_layer = data_w_safety_layer[:,3]
        velocity_w_safety_layer_wo_si = data_w_safety_layer_wo_si[:,3]
        velocity_wo_safety_layer = data_wo_safety_layer[:,3]
        velocity_pure_car_following = data_pure_car_following[:,3]

        hx_w_safety_layer = spacing_w_safety_layer - velocity_w_safety_layer*tau
        hx_w_safety_layer_wo_si = spacing_w_safety_layer_wo_si - velocity_w_safety_layer_wo_si*tau
        hx_wo_safety_layer = spacing_wo_safety_layer - velocity_wo_safety_layer*tau
        hx_pure_car_following = spacing_pure_car_following - velocity_pure_car_following*tau

    elif scenario == 3:
        spacing_w_safety_layer = data_w_safety_layer[:,5+4]
        spacing_w_safety_layer_wo_si = data_w_safety_layer_wo_si[:,5+4]
        spacing_wo_safety_layer = data_wo_safety_layer[:,5+4]
        spacing_pure_car_following = data_pure_car_following[:,5+4]

        velocity_w_safety_layer = data_w_safety_layer[:,4]
        velocity_w_safety_layer_wo_si = data_w_safety_layer_wo_si[:,4]
        velocity_wo_safety_layer = data_wo_safety_layer[:,4]
        velocity_pure_car_following = data_pure_car_following[:,4]

        hx_w_safety_layer = spacing_w_safety_layer - velocity_w_safety_layer*tau
        hx_w_safety_layer_wo_si = spacing_w_safety_layer_wo_si - velocity_w_safety_layer_wo_si*tau
        hx_wo_safety_layer = spacing_wo_safety_layer - velocity_wo_safety_layer*tau
        hx_pure_car_following = spacing_pure_car_following - velocity_pure_car_following*tau

    time_step = np.arange(0,len(spacing_w_safety_layer)*0.1,0.1)

    plt.figure(scenario, figsize=(10, 8))
    plt.plot(time_step,spacing_w_safety_layer, label='with safety layer',alpha=0.5)
    if scenario == 2 or scenario == 3:
        plt.plot(time_step,spacing_w_safety_layer_wo_si, label='with safety layer w/o SI',alpha=0.5)
    plt.plot(time_step,spacing_wo_safety_layer, label='w/o safety layer',alpha=0.5)
    plt.plot(time_step,spacing_pure_car_following, label='pure car following',alpha=0.5)

    #plt.plot(data_pure_car_following[:,5+3], label='pure car following')
    plt.xlabel('Time step', fontdict={'family' : 'Times New Roman', 'size'   : 27})
    plt.ylabel('Spacing', fontdict={'family' : 'Times New Roman', 'size'   : 2})
    # plt.legend(['with safety layer', 'with safety layer w\o SI', 'w\o safety layer', 'pure car following'])
    plt.legend(frameon=False, prop={'family' : 'Times New Roman', 'size'   : 25})
    plt.grid(True)
    plt.axhline(y=0,ls=":",c="black")
    plt.yticks(fontproperties = 'Times New Roman', size = 25)
    plt.xticks(fontproperties = 'Times New Roman', size = 25)

    plt.savefig('results\spacing_scenario_'+str(scenario)+'.pdf', dpi = 300)
    
    if vis_hx:
        plt.figure(scenario+10, figsize=(10, 8))
        plt.plot(time_step,hx_w_safety_layer, label='safe-RL with SI',alpha=0.5)
        if scenario == 2 or scenario == 3:
            plt.plot(time_step,hx_w_safety_layer_wo_si, label='safe-RL w/o SI',alpha=0.5)
        plt.plot(time_step,hx_wo_safety_layer, label='PPO w/o safety guarantee',alpha=0.5)
        plt.plot(time_step,hx_pure_car_following, label='pure car following',alpha=0.5)
        plt.xlabel('Time step', fontdict={'family' : 'Times New Roman', 'size'   : 27})
        if scenario == 2:
            plt.ylabel('h$_{th,'+str(3)+'}(x)$', fontdict={'family' : 'Times New Roman', 'size'   : 27})
        elif scenario == 3:
            plt.ylabel('h$_{th,'+str(4)+'}(x)$', fontdict={'family' : 'Times New Roman', 'size'   : 27})
        elif scenario == 4:
            plt.ylabel('h$_{th,'+str(2)+'}(x)$', fontdict={'family' : 'Times New Roman', 'size'   : 27})
        # plt.legend(['with safety layer', 'with safety layer w\o SI', 'w\o safety layer', 'pure car following'])
        plt.legend(frameon=False, prop={'family' : 'Times New Roman', 'size'   : 25})
        plt.grid(True)
        plt.axhline(y=0,ls=":",c="black")
        plt.yticks(fontproperties = 'Times New Roman', size = 25)
        plt.xticks(fontproperties = 'Times New Roman', size = 25)

        plt.savefig('results\hx_scenario_'+str(scenario)+'.pdf', dpi = 300)