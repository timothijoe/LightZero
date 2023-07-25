import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal, Independent

class WpDecoder(nn.Module):
    def __init__(self,
        control_num = 2,
        seq_len = 30,
        use_relative_pos = True,
        dt = 0.03,
        traj_control_mode = 'jerk',
        ):
        super(WpDecoder, self).__init__()
        self.control_num = control_num
        self.seq_len = seq_len 
        self.use_relative_pos = use_relative_pos
        self.dt = dt
        self.traj_control_mode = traj_control_mode

    def plant_model_acc(self, prev_state_batch, pedal_batch, steering_batch, dt = 0.03):
        # we assume the pedal batch and steer batch belongs to  [-1, 1]
        prev_state = prev_state_batch
        x_t = prev_state[:,0]
        y_t = prev_state[:,1]
        psi_t = prev_state[:,2]
        v_t = prev_state[:,3]
        # Here we scale them to the fitable value
        steering_batch = steering_batch * 0.5
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        pedal_batch = pedal_batch * 5
        pedal_batch = torch.clamp(pedal_batch, -5, 5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt 
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_t_1 = psi_dot*dt + psi_t 
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t 
        y_t_1 = y_dot * dt + y_t
        
        #psi_t = self.wrap_angle_rad(psi_t)
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim = 1)
        #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
        return current_state

    def plant_model_jerk(self, prev_state_batch, jerk_batch, steering_rate_batch, dt = 0.03):
        # x, y, theta, v, acc, steer, 
        # control, jerk
        #import copy
        prev_state = prev_state_batch
        x_t = prev_state[:,0]
        y_t = prev_state[:,1]
        psi_t = prev_state[:,2]
        v_t = prev_state[:,3]
        pedal_t = prev_state[:,4]
        steering_t = prev_state[:, 5]
        jerk_batch = jerk_batch * 4
        steering_rate_batch = steering_rate_batch * 0.5
        jerk_batch = torch.clamp(jerk_batch, -4, 4)
        steering_rate_batch = torch.clamp(steering_rate_batch, -0.5, 0.5)
        pedal_batch = pedal_t + jerk_batch * dt 
        pedal_batch = torch.clamp(pedal_batch, -5, 5)
        steering_batch = steering_t + steering_rate_batch * dt
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt 
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_dot = torch.clamp(psi_dot, -3.14 /2,3.14 /2)
        psi_t_1 = psi_dot*dt + psi_t 
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t 
        y_t_1 = y_dot * dt + y_t
        
        #psi_t = self.wrap_angle_rad(psi_t)
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1, pedal_batch, steering_batch], dim = 1)
        #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
        return current_state        

    def decode(self, z, init_state):
        generated_traj = []
        prev_state = init_state 
        assert z.shape[1] == self.seq_len * 2
        for i in range(self.seq_len):
            control_1 = z[:, 2*i]
            control_2 = z[:, 2*i +1]
            #curr_state = self.plant_model_batch(prev_state, jerk_batch, steer_rate_batch, self.dt)
            if self.traj_control_mode == 'jerk':
                curr_state = self.plant_model_jerk(prev_state, control_1, control_2, self.dt)
            elif self.traj_control_mode == 'acc':
                curr_state = self.plant_model_acc(prev_state, control_1, control_2, self.dt)

            generated_traj.append(curr_state)
            prev_state = curr_state 
        generated_traj = torch.stack(generated_traj, dim = 1)
        return generated_traj
    
    def forward(self, z, init_state):
        return self.decode(z, init_state)



class CCDecoder(nn.Module):
    def __init__(self,
        control_num = 2,
        seq_len = 30,
        use_relative_pos = True,
        dt = 0.03,
        traj_control_mode = 'jerk',
        steer_rate_constrain_value=0.4,
        ):
        super(CCDecoder, self).__init__()
        self.control_num = control_num
        self.seq_len = seq_len 
        self.use_relative_pos = use_relative_pos
        self.dt = dt
        self.traj_control_mode = traj_control_mode
        self.steer_rate_constrain_value=steer_rate_constrain_value


    # def plant_model_batch(self, prev_state_batch, pedal_batch, steering_batch, dt = 0.03):
    #     #import copy
    #     prev_state = prev_state_batch
    #     x_t = prev_state[:,0]
    #     y_t = prev_state[:,1]
    #     psi_t = prev_state[:,2]
    #     v_t = prev_state[:,3]
    #     steering_batch = steering_batch * 0.4
    #     #pedal_batch = torch.clamp(pedal_batch, -5, 5)
    #     steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
    #     beta = steering_batch
    #     a_t = pedal_batch * 4
    #     v_t_1 = v_t + a_t * dt 
    #     v_t_1 = torch.clamp(v_t_1, 0, 10)
    #     psi_dot = v_t * torch.tan(beta) / 2.5
    #     psi_t_1 = psi_dot*dt + psi_t 
    #     x_dot = v_t_1 * torch.cos(psi_t_1)
    #     y_dot = v_t_1 * torch.sin(psi_t_1)
    #     x_t_1 = x_dot * dt + x_t 
    #     y_t_1 = y_dot * dt + y_t
        
    #     #psi_t = self.wrap_angle_rad(psi_t)
    #     current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim = 1)
    #     #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
    #     return current_state

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        t_min = t_min.float()
        t_max = t_max.float()
        result = (t > t_min).float() * t + (t < t_min).float() * t_min 
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max 
        return result 
    def plant_model_acc(self, prev_state_batch, pedal_batch, steering_batch, dt = 0.03, last_st = None,st_rate_constrain=0.5):
        # we assume the pedal batch and steer batch belongs to  [-1, 1]
        prev_state = prev_state_batch
        x_t = prev_state[:,0]
        y_t = prev_state[:,1]
        psi_t = prev_state[:,2]
        v_t = prev_state[:,3]



        # Here we scale them to the fitable value
        # steering_batch = steering_batch * 0.5

        # if last_st is None: 
        #     last_st = torch.zeros_like(steering_batch)
        # d_steer = st_rate_constrain * dt 
        # min_st = last_st - d_steer 
        # max_st = last_st + d_steer 
        # steering_batch = self.clip_by_tensor(steering_batch, min_st, max_st)
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)



        # print('steering batch')
        # print(steering_batch)
        # pedal_batch = pedal_batch * 2.5
        pedal_batch = torch.clamp(pedal_batch, -2.5, 2.5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt 
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_t_1 = psi_dot*dt + psi_t 
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t 
        y_t_1 = y_dot * dt + y_t
        
        #psi_t = self.wrap_angle_rad(psi_t)
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim = 1)
        #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
        return current_state, beta

    def plant_model_jerk(self, prev_state_batch, jerk_batch, steering_rate_batch, dt = 0.03):
        # x, y, theta, v, acc, steer, 
        # control, jerk
        #import copy
        prev_state = prev_state_batch
        x_t = prev_state[:,0]
        y_t = prev_state[:,1]
        psi_t = prev_state[:,2]
        v_t = prev_state[:,3]
        pedal_t = prev_state[:,4]
        steering_t = prev_state[:, 5]
        jerk_batch = jerk_batch * 4
        steering_rate_batch = steering_rate_batch * 0.5
        jerk_batch = torch.clamp(jerk_batch, -4, 4)
        steering_rate_batch = torch.clamp(steering_rate_batch, -0.5, 0.5)
        pedal_batch = pedal_t + jerk_batch * dt 
        pedal_batch = torch.clamp(pedal_batch, -5, 5)
        steering_batch = steering_t + steering_rate_batch * dt
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt 
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_dot = torch.clamp(psi_dot, -3.14 /2,3.14 /2)
        psi_t_1 = psi_dot*dt + psi_t 
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t 
        y_t_1 = y_dot * dt + y_t
        
        #psi_t = self.wrap_angle_rad(psi_t)
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1, pedal_batch, steering_batch], dim = 1)
        #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
        return current_state 

    # def decode(self, z, init_state):
    #     generated_traj = []
    #     state_shape = init_state.shape  
    #     last_st = None
    #     if state_shape[1] == 6:
    #         last_st = init_state[:, 5]
    #         init_state = init_state[:, :4]
            
    #     prev_state = init_state 
    #     st_rate_constrain = self.steer_rate_constrain_value
    #     assert z.shape[1] == 2

    #     steering_batch = z[:, 1] * 0.5

    #     if last_st is None: 
    #         last_st = torch.zeros_like(steering_batch)
    #     # d_steer = st_rate_constrain * self.dt 
    #     # min_st = last_st - d_steer 
    #     # max_st = last_st + d_steer 
    #     # # steering_batch = self.clip_by_tensor(steering_batch, min_st, max_st)
    #     # steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
    #     # pedal_batch = z[:, 0] * 2.5

    #     for i in range(self.seq_len):
    #         control_1 = z[:, 0]
    #         control_2 = z[:, 1]
    #         # acc: from -1 to 1  -> -2.5 to 2.5
    #         # steer: from -1 to 1 -> -0.5 to 0.5
    #         pedal_batch = control_1 * 2.5 
    #         steering_batch = control_2 * 0.5
    #         if self.traj_control_mode == 'jerk':
    #             curr_state = self.plant_model_jerk(prev_state, control_1, control_2, self.dt)
    #         elif self.traj_control_mode == 'acc':
    #             curr_state, beta = self.plant_model_acc(prev_state, pedal_batch, steering_batch, self.dt, last_st, st_rate_constrain)
    #             # print('steering batch: {}'.format(steering_batch))
    #             #curr_state, beta = self.plant_model_acc(prev_state, control_1, control_2, self.dt, last_st, 0.3)
    #         #curr_state = self.plant_model_batch(prev_state, pedal_batch, steer_batch, self.dt)
    #         generated_traj.append(curr_state)
    #         prev_state = curr_state 
    #         last_st = beta 
    #     generated_traj = torch.stack(generated_traj, dim = 1)
    #     return generated_traj


    def decode(self, z, init_state):
        generated_traj = []
        state_shape = init_state.shape  
        last_st = None
        if state_shape[1] == 6:
            last_st = init_state[:, 5]
            init_state = init_state[:, :4]
            
        prev_state = init_state 
        st_rate_constrain = self.steer_rate_constrain_value
        assert z.shape[1] == 2

        if last_st is None: 
            last_st = torch.zeros_like(steering_batch)
        steering_batch = z[:, 1] * 0.5
        d_steer = st_rate_constrain * self.dt 
        min_st = last_st - d_steer 
        max_st = last_st + d_steer 
        steering_batch = self.clip_by_tensor(steering_batch, min_st, max_st)
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        pedal_batch = z[:, 0] * 2.5

        for i in range(self.seq_len):
            # control_1 = z[:, 0]
            # control_2 = z[:, 1]
            # # acc: from -1 to 1  -> -2.5 to 2.5
            # # steer: from -1 to 1 -> -0.5 to 0.5
            # pedal_batch = control_1 * 2.5 
            # steering_batch = control_2 * 0.5
            if self.traj_control_mode == 'jerk':
                curr_state = self.plant_model_jerk(prev_state, control_1, control_2, self.dt)
            elif self.traj_control_mode == 'acc':
                # curr_state, beta = self.plant_model_acc(prev_state, pedal_batch, steering_batch, self.dt, last_st, st_rate_constrain)
                curr_state, beta = self.plant_model_acc(prev_state, pedal_batch, steering_batch, self.dt)
                # print('steering batch: {}'.format(steering_batch))
                #curr_state, beta = self.plant_model_acc(prev_state, control_1, control_2, self.dt, last_st, 0.3)
            #curr_state = self.plant_model_batch(prev_state, pedal_batch, steer_batch, self.dt)
            generated_traj.append(curr_state)
            prev_state = curr_state 
            last_st = beta 
        generated_traj = torch.stack(generated_traj, dim = 1)
        return generated_traj

    def forward(self, z, init_state):
        return self.decode(z, init_state)