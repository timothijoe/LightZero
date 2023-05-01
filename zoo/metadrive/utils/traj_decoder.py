import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal, Independent

class VaeDecoder(nn.Module):
    def __init__(self,
        embedding_dim = 64,
        h_dim = 64,
        latent_dim = 3,
        seq_len = 20,
        use_relative_pos = True,
        dt = 0.1,
        traj_control_mode = 'acc',
        one_side_class_vae = False,
        steer_rate_constrain_value = 0.4,
        ):
        super(VaeDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim 
        self.num_layers = 1
        self.latent_dim = latent_dim
        self.label_dim = 2
        self.seq_len = seq_len 
        self.use_relative_pos = use_relative_pos
        self.dt = dt
        self.traj_control_mode = traj_control_mode
        self.steer_rate_constrain_value = steer_rate_constrain_value
        # input: x, y, theta, v,   output: embedding
        if self.traj_control_mode == 'jerk':
            self.spatial_embedding = nn.Linear(6, self.embedding_dim)
        elif self.traj_control_mode == 'acc':
            self.spatial_embedding = nn.Linear(4, self.embedding_dim)
        self.hidden2control = nn.Linear(self.h_dim, 2)
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, self.num_layers)
        self.one_side_class_vae = one_side_class_vae
        if self.one_side_class_vae:
            self.init_hidden_decoder = torch.nn.Linear(in_features = self.latent_dim-1, out_features = self.h_dim * self.num_layers)
        else:
            self.init_hidden_decoder = torch.nn.Linear(in_features = self.latent_dim, out_features = self.h_dim * self.num_layers)
        label_dims = [self.h_dim, self.h_dim, self.h_dim, self.label_dim]
        label_modules = []
        in_channels = 1 if self.one_side_class_vae else self.latent_dim
        for m_dim in label_dims:
            label_modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    nn.LeakyReLU())
            )
            in_channels = m_dim  
        self.label_classification = nn.Sequential(*label_modules) 

    def plant_model_jerk(self, prev_state_batch, jerk_batch, steering_rate_batch, dt = 0.03):
        prev_state = prev_state_batch
        x_t = prev_state[:,0]
        y_t = prev_state[:,1]
        psi_t = prev_state[:,2]
        v_t = prev_state[:,3]
        pedal_t = prev_state[:,4]
        steering_t = prev_state[:, 5]
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

    def plant_model_acc(self, prev_state_batch, pedal_batch, steering_batch, dt = 0.03):
        #import copy
        prev_state = prev_state_batch
        x_t = prev_state[:,0]
        y_t = prev_state[:,1]
        psi_t = prev_state[:,2]
        v_t = prev_state[:,3]
        pedal_batch = torch.clamp(pedal_batch, -5, 5)
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
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim = 1)
        #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
        return current_state



    def decode1(self, z, init_state):
        generated_traj = []
        prev_state = init_state 
        # output_label = self.label_classification(z)
        if self.one_side_class_vae:
            output_label = self.label_classification(z[:,:1])
        else:
            output_label = self.label_classification(z[:,:])
        # decoder_input shape: batch_size x 4
        decoder_input = self.spatial_embedding(prev_state)
        decoder_input = decoder_input.view(1, -1 , self.embedding_dim)
        decoder_h = self.init_hidden_decoder(z)
        if len(decoder_h.shape) == 2:
            decoder_h = torch.unsqueeze(decoder_h, 0)
            #decoder_h.unsqueeze(0)
        decoder_h = (decoder_h, decoder_h)
        for _ in range(self.seq_len):
            # output shape: 1 x batch x h_dim
            output, decoder_h = self.decoder(decoder_input, decoder_h)
            control = self.hidden2control(output.view(-1, self.h_dim))
            if self.traj_control_mode == 'jerk':
                curr_state = self.plant_model_jerk(prev_state, control[:,0], control[:,1], self.dt)
            elif self.traj_control_mode == 'acc':
                curr_state = self.plant_model_acc(prev_state, control[:,0], control[:,1], self.dt)
            #curr_state = self.plant_model_jerk(prev_state, control[:,0], control[:,1], self.dt)
            generated_traj.append(curr_state)
            decoder_input = self.spatial_embedding(curr_state)
            decoder_input = decoder_input.view(1, -1, self.embedding_dim)
            prev_state = curr_state 
        generated_traj = torch.stack(generated_traj, dim = 1)
        return generated_traj, output_label

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        t_min = t_min.float()
        t_max = t_max.float()
        result = (t > t_min).float() * t + (t < t_min).float() * t_min 
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max 
        return result 

    def plant_model_batch(self, prev_state_batch, pedal_batch, steering_batch, dt = 0.03, last_st = None, st_rate_constrain=0.5):
        #import copy
        prev_state = prev_state_batch
        x_t = prev_state[:,0]
        y_t = prev_state[:,1]
        psi_t = prev_state[:,2]
        v_t = prev_state[:,3]
        #pedal_batch = torch.clamp(pedal_batch, -5, 5)
        
        # if last_st is not None: 
        #     d_steer = st_rate_constrain * dt 
        #     min_st = last_st - d_steer 
        #     max_st = last_st + d_steer 
        #     steering_batch = self.clip_by_tensor(steering_batch, min_st, max_st)
        if last_st is None: 
            last_st = torch.zeros_like(steering_batch)
        d_steer = st_rate_constrain * dt 
        min_st = last_st - d_steer 
        max_st = last_st + d_steer 
        steering_batch = self.clip_by_tensor(steering_batch, min_st, max_st)
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
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim = 1)
        #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
        return current_state, steering_batch


    def decode(self, z, init_state):
        if self.one_side_class_vae:
            # output_label = self.label_classification(z[:,:1])
            generated_traj = self.decode_len20(z[:,1:], init_state)
            # generated_traj_len10 = self.decode_len10(z[:,1:], init_state)

        else:
            # output_label = self.label_classification(z[:,:])
            generated_traj = self.decode_len20(z, init_state)
            # generated_traj_len10 = self.decode_len10(z, init_state)
        # generated_traj = self.decode_len20(z, init_state)
        # generated_traj_len10 = self.decode_len10(z, init_state)
        return generated_traj
        # return generated_traj, generated_traj_len10, output_label

    def decode_len10(self, z, init_state):
        generated_traj = []
        prev_state = init_state
        # decoder_input shape: batch_size x 4
        decoder_input = self.spatial_embedding(prev_state)
        decoder_input = decoder_input.view(1, -1 , self.embedding_dim)
        decoder_h = self.init_hidden_decoder(z)
        if len(decoder_h.shape) == 2:
            decoder_h = torch.unsqueeze(decoder_h, 0)
            #decoder_h.unsqueeze(0)
        decoder_h = (decoder_h, decoder_h)
        last_st = None
        for _ in range(10):
            # output shape: 1 x batch x h_dim
            output, decoder_h = self.decoder_len10(decoder_input, decoder_h)
            control = self.hidden2control(output.view(-1, self.h_dim))
            #last_st = None 
            curr_state, steering_batch = self.plant_model_batch(prev_state, control[:,0], control[:,1], self.dt, last_st, 0.1)
            generated_traj.append(curr_state)
            decoder_input = self.spatial_embedding(curr_state)
            decoder_input = decoder_input.view(1, -1, self.embedding_dim)
            prev_state = curr_state 
            last_st = steering_batch
        generated_traj = torch.stack(generated_traj, dim = 1)
        return generated_traj


    def decode_len20(self, z, init_state):
        generated_traj = []
        last_st = None
        state_shape = init_state.shape
        if state_shape[1] == 6:
            last_st = init_state[:,5]
            init_state = init_state[:, :4]
        prev_state = init_state
        max_steer_rate = self.steer_rate_constrain_value
        decoder_input = self.spatial_embedding(prev_state)
        decoder_input = decoder_input.view(1, -1 , self.embedding_dim)
        decoder_h = self.init_hidden_decoder(z)
        if len(decoder_h.shape) == 2:
            decoder_h = torch.unsqueeze(decoder_h, 0)
        decoder_h = (decoder_h, decoder_h)
        for _ in range(self.seq_len):
            # output shape: 1 x batch x h_dim
            output, decoder_h = self.decoder(decoder_input, decoder_h)
            control = self.hidden2control(output.view(-1, self.h_dim))
            #last_st = None
            curr_state, steering_batch = self.plant_model_batch(prev_state, control[:,0], control[:,1], self.dt, last_st,max_steer_rate)
            generated_traj.append(curr_state)
            decoder_input = self.spatial_embedding(curr_state)
            decoder_input = decoder_input.view(1, -1, self.embedding_dim)
            prev_state = curr_state 
            last_st = steering_batch
        generated_traj = torch.stack(generated_traj, dim = 1)
        return generated_traj

    def forward(self, z, init_state):
        return self.decode(z, init_state)
