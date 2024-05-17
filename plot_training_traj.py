from visualize import plot_rewards, plot_velocity_and_spacing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['mathtext.default'] = 'regular'

def plot_training_data_seperate(agent_select, safety_layer_enabled):
    '''
    Plot the training data for each vehicle and the total reward for each episode separately
    '''
    # Load the data from csv
    if safety_layer_enabled:
        episode_rewards_pd = pd.read_csv('training_traj/episode_rewards_' + agent_select + '.csv', index_col=False)
        velocity_data_pd = pd.read_csv('training_traj/velocity_data_' + agent_select + '.csv', index_col=False)
        spacing_data_pd = pd.read_csv('training_traj/spacing_data_' + agent_select + '.csv', index_col=False)
    else:
        episode_rewards_pd = pd.read_csv('training_traj/episode_rewards_no_safety_' + agent_select + '.csv', index_col=False)
        velocity_data_pd = pd.read_csv('training_traj/velocity_data_no_safety_' + agent_select + '.csv', index_col=False)
        spacing_data_pd = pd.read_csv('training_traj/spacing_data_no_safety_' + agent_select + '.csv', index_col=False)

    # Convert the data to numpy array
    velocity_data = velocity_data_pd[['0','1','2','3','4']].to_numpy()
    spacing_data = spacing_data_pd[['0','1','2','3','4']].to_numpy()

    # Plot the data
    if safety_layer_enabled:
        path = 'training_traj/episode_rewards_' + agent_select + '.png'
        plot_rewards(episode_rewards_pd, window_size=1, save_path=path)
        for veh_id in range(5):
            path = 'training_traj/vehicl_' + agent_select + '_'+str(veh_id)+'.png'
            plot_velocity_and_spacing(velocity_data[:,veh_id], spacing_data[:,veh_id ], veh_id = veh_id, save_path=path)
    else:
        path = 'training_traj/episode_rewards_' + agent_select + '_no_safety.png'
        plot_rewards(episode_rewards_pd, window_size=1, save_path=path)
        for veh_id in range(5):
            path = 'training_traj/vehicle_' + agent_select + '_'+str(veh_id)+'_no_safety.png'
            plot_velocity_and_spacing(velocity_data[:,veh_id], spacing_data[:,veh_id ], veh_id = veh_id, save_path=path)


def plot_training_data_join(agent_list):
    '''
    Plot the total reward for each episode together
    '''
    for agent_select in agent_list:
        # Load the data from csv
        episode_rewards_pd = pd.read_csv('training_traj/episode_rewards_' + agent_select + '.csv', index_col=False)
        episode_rewards_pd_no_safety = pd.read_csv('training_traj/episode_rewards_no_safety_' + agent_select + '.csv', index_col=False)

        # Saving path
        path = 'training_traj/episode_rewards_join_' + agent_select + '.pdf'

        # Plot the data
        fig, ax = plt.subplots(1,1,figsize=(10, 8.2))
        plot_rewards(episode_rewards_pd.to_numpy()[:,1], window_size=1, label = agent_select + ' with safety layer')
        plot_rewards(episode_rewards_pd_no_safety.to_numpy()[:,1], window_size=1, label = agent_select + ' without safety layer')
        plt.grid(True)
        ax.ticklabel_format(style='scientific', scilimits=(-1,2), axis='y') # set y axis to scientific notation
        ax.yaxis.get_offset_text().set(fontproperties = 'Times New Roman',size=25)  # 左上角
        plt.yticks(fontproperties = 'Times New Roman', size = 28)
        plt.xticks(fontproperties = 'Times New Roman', size = 28)
        plt.legend(['safe-RL with SI', 'PPO without safety guarantee'], frameon=False, prop={'family' : 'Times New Roman', 'size'   : 25})
        #plt.legend([agent_select + ' with safety layer', agent_select +' without safety layer'], frameon=False, prop={'family' : 'Times New Roman', 'size'   : 25})
        

        # draw inset figure
        axins = inset_axes(ax, width="45%", height="35%", loc='lower left',
                   bbox_to_anchor=(0.35, 0.2, 1, 1),
                   bbox_transform=ax.transAxes)
        axins.plot(episode_rewards_pd.to_numpy()[:,1], label = agent_select.upper() + ' with safety layer')
        axins.plot(episode_rewards_pd_no_safety.to_numpy()[:,1], label = agent_select.upper() + ' without safety layer')

        axins.set_xlim(0, 100)
        if agent_select == 'ppo':
            axins.set_ylim(-10000, 0)
        else:
            axins.set_ylim(-200000, 0)
        axins.xaxis.set_visible(False)
        axins.yaxis.set_visible(False)
        
        # draw triangle for original data
        tx0 = 0
        tx1 = 100
        if agent_select == 'ppo':
            ty0 = -10000
        else:
            ty0 = -200000

        ty1 = 0
        sx = [tx0,tx1,tx1,tx0,tx0]
        sy = [ty0,ty0,ty1,ty1,ty0]
        ax.plot(sx,sy,"black",lw=0.8)

        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=0.8)

        plt.savefig(path, dpi = 300)
    plt.show()


if __name__ == "__main__":

    # choose to plot the training data for each vehicle and the total reward for each episode separately or together
    if_seperate = False

    if if_seperate:
        plot_training_data_seperate('ddpg', True)
        plot_training_data_seperate('ddpg', False)
        plot_training_data_seperate('ppo', True)
        plot_training_data_seperate('ppo', False)
        plt.show()

    else:
        agent_list = ['ppo']
        plot_training_data_join(agent_list)
    