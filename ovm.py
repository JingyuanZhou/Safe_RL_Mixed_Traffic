import numpy as np

class OptimalVelocityModel:
    def __init__(self, num_vehicles=5, params=None, cav_index = None):
        # Set the default parameters
        if params is None:
            params = {
                'alpha': 0.6,
                'beta': 0.9,
                'tau': 1.5,
                'v0': 15, # Equilibrium velocity
                's0': 20, # Equilibrium spacing
                's_st': 5,
                's_go': 35,
                'v_max': 30,
                'velocity_noise': 2, # Velocity fluctuation amplitude for the leading vehicle
                'a_max': 5, # 5
                'a_min': -5 # 5
            }
        self.num_vehicles = num_vehicles
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.tau = params['tau']
        self.v0 = params['v0']
        self.s0 = params['s0']
        self.s_st = params['s_st']
        self.s_go = params['s_go']
        self.v_max = params['v_max']
        self.velocity_noise = params['velocity_noise']
        self.a_max = params['a_max']
        self.a_min = params['a_min']
        self.num_vehicles = num_vehicles
        self.spacing = np.zeros(num_vehicles)
        self.velocity = np.zeros(num_vehicles)
        self.position = np.zeros(num_vehicles)
        self.control_input = np.zeros(num_vehicles)
        self.cav_index = cav_index
        self.acceleration = np.zeros(num_vehicles)
        self.t = 0

        self.disturbance = None

    def reset(self, disturbance = None):
        # The spacing between the vehicles
        self.spacing.fill(self.s0) 
        # The velocity of the vehicles
        self.velocity.fill(self.v0) 
        self.t = 0
        # The position of the vehicles
        for i in range(self.num_vehicles): 
            self.position[i] = (self.num_vehicles - i - 1) * self.s0
        # The control input of the vehicles
        self.control_input.fill(0)
        # The acceleration of the vehicles
        self.acceleration.fill(0)

        self.disturbance = disturbance

    def set_control_input(self, vehicle_idx, control_input):
        # Set the control input of the vehicle
        self.control_input[vehicle_idx] = control_input

    def update(self, dt, select_scenario, pure_car_following):
        # Update the time
        self.t += dt
        if select_scenario == 0 or select_scenario == 1:
            scenario_id = [0]
            duration = [0,5]
        elif select_scenario == 2:
            scenario_id = [3]
            duration = [0,8]
        elif select_scenario == 3:
            scenario_id = [4]
            duration = [0,8]
        elif select_scenario == 4:
            scenario_id = [1]
            duration = [0,5]
        if self.disturbance is not None:
            duration = [0, self.disturbance[0]]

        # Update all the vehicles
        for i in range(self.num_vehicles - 1, 0, -1):
            # Update the velocity
            if i in self.cav_index and not pure_car_following:
                # The autonomous vehicle uses the provided control input
                dv = self.velocity[i-1] - self.velocity[i]
                if self.control_input[i] > self.a_max:
                    self.control_input[i] = self.a_max
                elif self.control_input[i] < self.a_min:
                    self.control_input[i] = self.a_min

                self.velocity[i] += self.control_input[i] * dt
                # Update the spacing
                self.spacing[i] += dv * dt
                self.position[i] += self.velocity[i] * dt

                self.acceleration[i] = self.control_input[i]

            elif i in scenario_id and self.t >= duration[0] and self.t < duration[1]:# and not pure_car_following
                # Update the lead vehicle (the first vehicle) in different scenarios
                dv = self.velocity[i-1] - self.velocity[i]
                if select_scenario == 2:
                    if self.disturbance is not None:
                        emergent_acc = self.disturbance[1]
                    else:
                        emergent_acc = 1
                    if self.t >= duration[0] and self.t < duration[1]/2:
                        self.velocity[i] += emergent_acc * dt
                        self.acceleration[i] = emergent_acc
                    if self.t >= duration[1]/2 and self.t < duration[1]:
                        self.velocity[i] += 0
                        self.acceleration[i] = 0
                    self.position[i] += self.velocity[i] * dt
                    self.spacing[i] += dv * dt

                elif select_scenario == 3:
                    if self.disturbance is not None:
                        emergent_acc = self.disturbance[1]
                    else:
                        emergent_acc = 1

                    if self.t >= duration[0] and self.t < duration[1]/2:
                        self.velocity[i] += emergent_acc * dt
                        self.acceleration[i] = emergent_acc
                    if self.t >= duration[1]/2 and self.t < duration[1]:
                        self.velocity[i] += 0
                        self.acceleration[i] = 0
                    self.position[i] += self.velocity[i] * dt
                    self.spacing[i] += dv * dt
                    
                elif select_scenario == 4:
                    if self.disturbance is not None:
                        braking_acc = self.disturbance[1]
                    else:
                        braking_acc = -4

                    if self.t >= duration[0] and self.t < duration[1]/2:
                        self.velocity[i] += braking_acc * dt
                        self.acceleration[i] = braking_acc
                    if self.t >= duration[1]/2 and self.t < duration[1]:
                        self.velocity[i] += 0
                        self.acceleration[i] = 0
                    self.position[i] += self.velocity[i] * dt
                    self.spacing[i] += dv * dt
                    
            else:
                # The other vehicles use the OVM model
                dv = self.velocity[i-1] - self.velocity[i]
                ds = dv
                
                # calculate the desired velocity and spacing
                s_star = self.s0
                v_star = self.alpha*self.v_max/2*np.pi/(self.s_go-self.s_st)*np.sin(np.pi*(s_star-self.s_st)/(self.s_go-self.s_st))

                # calculate the desired acceleration
                cal_D = self.spacing[i]
                if cal_D > self.s_go:
                    cal_D = self.s_go
                elif cal_D < self.s_st:
                    cal_D = self.s_st

                acceleration = self.alpha * (self.v_max/2*(1-np.cos(np.pi*(cal_D-self.s_st)/(self.s_go-self.s_st))) - self.velocity[i]) + self.beta * dv
                if acceleration > self.a_max:
                    acceleration = self.a_max
                elif acceleration < self.a_min:
                    acceleration = self.a_min

                self.acceleration[i] = acceleration

                self.velocity[i] += acceleration * dt
            
                # Update the spacing
                self.spacing[i] += dv * dt
                self.position[i] += self.velocity[i] * dt

        if select_scenario == 0:
            # Random noise scenario
            self.velocity[0] += (self.control_input[0] + np.random.normal(0, self.velocity_noise)) * dt
            self.spacing[0] += 0
            self.position[0] += self.velocity[0] * dt
            self.acceleration[0] = (self.control_input[0] + np.random.normal(0, self.velocity_noise))
        elif select_scenario == 1:
            # Emergency braking scenario
            braking_acc = -5
            self.acceleration[0] = 0
            if self.t >= duration[0] and self.t < duration[1]/2:
                self.velocity[0] += braking_acc * dt
                self.acceleration[0] = braking_acc
            elif self.t >= duration[1]/2 and self.t < duration[1]:
                self.velocity[0] += -braking_acc * dt
                self.acceleration[0] = -braking_acc
                
            self.position[0] += self.velocity[0] * dt
            


        
