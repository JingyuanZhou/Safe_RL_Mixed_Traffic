import gym
import numpy as np
from gym import spaces
from ovm import OptimalVelocityModel

class PlatoonEnv(gym.Env):
    def __init__(self, num_vehicles=5, ovm_params=None, dt=0.1, max_steps=400, select_scenario=0, pure_car_following=False, set_disturbance = False):
        super(PlatoonEnv, self).__init__()

        # Set the default parameters
        self.num_vehicles = num_vehicles
        self.dt = dt
        self.max_steps = max_steps
        self.steps = 0
        # 0: random head vehicle velocity, 1: emergency braking
        self.select_scenario = select_scenario 
        self.pure_car_following = pure_car_following

        # Set up action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 * (num_vehicles - 1),), dtype=np.float32)

        # Initialize the OptimalVelocityModel
        self.cav_index = [2]
        self.ovm = OptimalVelocityModel(num_vehicles=num_vehicles, params=ovm_params, cav_index= self.cav_index)
        self.hx_ls = []

        self.set_disturbance = set_disturbance

    def reset(self, disturbance = None):
        self.steps = 0
        if self.set_disturbance:
            self.ovm.reset(disturbance)
        else:
            self.ovm.reset()

        return self._get_obs(), self.get_acceleration()

    def step(self, action):
        # Apply action to the AV (vehicle 2)
        self.ovm.set_control_input(2, action[0])

        # Update the OVM for all vehicles
        self.ovm.update(self.dt, self.select_scenario, self.pure_car_following)

        obs = self._get_obs()
        reward = self._get_reward()
        acceleration = self.get_acceleration()
        done = self.steps >= self.max_steps
        self.steps += 1

        return obs, reward, acceleration, done, {}

    def _get_obs(self):
        # The observation is the spacing, velocity, differential spacing, and differential velocity of all vehicles
        spacing_diff = self.ovm.spacing[:-1] - self.ovm.spacing[1:]
        velocity_diff = self.ovm.velocity[:-1] - self.ovm.velocity[1:]
        obs = np.concatenate((self.ovm.spacing[1:], self.ovm.velocity[1:], spacing_diff, velocity_diff))
        return obs

    def get_velocity(self):
        # Return the velocity of all vehicles
        return self.ovm.velocity.copy()

    def get_spacing(self):
        # Return the spacing of all vehicles
        return self.ovm.spacing.copy()
    
    def get_position(self):
        # Return the position of all vehicles
        return self.ovm.position.copy()
    
    def get_acceleration(self):
        # Return the acceleration of all vehicles
        return self.ovm.acceleration.copy()

    def _get_reward(self):
        # Energy consumption (not considered in this work)
        energy_consumption = 0 

        eps = 1e-6
        # Safety
        ttc = - self.ovm.spacing[2] / (self.ovm.velocity[1] - self.ovm.velocity[2] + eps)
        if ttc >= 0 and ttc <= 4:
            safety = np.log(ttc / 4)
        else:
            safety = 0

        ttc_FW1 = - self.ovm.spacing[3] / (self.ovm.velocity[2] - self.ovm.velocity[3] + eps)
        if ttc_FW1 >= 0 and ttc_FW1 <= 4:
            safety_FW1 = 0.5*np.log(ttc_FW1 / 4)
        else:
            safety_FW1 = 0

        ttc_FW2 = - self.ovm.spacing[4] / (self.ovm.velocity[3] - self.ovm.velocity[4] + eps)
        if ttc_FW2 >= 0 and ttc_FW2 <= 4:
            safety_FW2 = 0.1*np.log(ttc_FW2 / 4)
        else:
            safety_FW2 = 0

        # Traffic Efficiency
        TG = self.ovm.spacing[2] / (self.ovm.velocity[2] + eps)
        if TG >= 2.5:
            R_eff = -1
        else:
            R_eff = 0

        # Stability
        v_star = self.ovm.v0
        s_star = self.ovm.s0
        stability = -np.sum(np.square((self.ovm.velocity[2:4] - self.ovm.velocity[1])))

        # CBF candidate Lie derivatives difference
        # dot_h = 
        

        # Return the sum of the rewards
        return energy_consumption + safety + stability + R_eff + safety_FW1 + safety_FW2

