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
        steer_rate_constrain_value = 0.4,
        ):
        super(VaeDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim 
        self.num_layers = 1
        self.latent_dim = latent_dim
        self.label_dim = 1#1
        self.seq_len = seq_len 
        self.use_relative_pos = use_relative_pos
        self.dt = dt
        # input: x, y, theta, v,   output: embedding
        self.spatial_embedding = nn.Linear(4, self.embedding_dim)
        # input: h_dim, output: throttle, steer
        self.hidden2control = nn.Linear(self.h_dim, 2)
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, self.num_layers)
        self.steer_rate_constrain_value = steer_rate_constrain_value
        self.init_hidden_decoder = torch.nn.Linear(in_features = self.latent_dim, out_features = self.h_dim * self.num_layers)
        label_dims = [self.h_dim, self.h_dim, self.h_dim, self.label_dim]
        label_modules = []
        in_channels = self.latent_dim -0
        for m_dim in label_dims:
            label_modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    nn.LeakyReLU())
            )
            in_channels = m_dim  
        self.label_classification = nn.Sequential(*label_modules) 

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
        
        if last_st is not None: 
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
        generated_traj = self.decode_len20(z, init_state)
        return generated_traj

    def decode_len20(self, z, init_state):
        # generated_traj = []
        # prev_state = init_state
        # max_steer_rate = self.steer_rate_constrain_value
        # # decoder_input shape: batch_size x 4
        # decoder_input = self.spatial_embedding(prev_state)
        # decoder_input = decoder_input.view(1, -1 , self.embedding_dim)
        # decoder_h = self.init_hidden_decoder(z)
        # if len(decoder_h.shape) == 2:
        #     decoder_h = torch.unsqueeze(decoder_h, 0)
        #     #decoder_h.unsqueeze(0)
        # decoder_h = (decoder_h, decoder_h)
        # last_st = None
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

    # def forward(self, z, init_state):
    #     return self.decode(z, init_state)


    def is_straight_trajectory(self, states, y_variation_threshold=1.2, xy_ratio_threshold=6, theta_variation_threshold=0.3, theta_diff_threshold=0.1):
        # 提取 x, y, 和 theta 列
        xs = states[:, 0]
        ys = states[:, 1]
        thetas = states[:, 2]

        # 检查Y轴波动是否过大
        if torch.max(ys) - torch.min(ys) > y_variation_threshold:
            return False

        # 检查X轴是否有足够的变化
        if (torch.max(xs) - torch.min(xs)) < xy_ratio_threshold * (torch.max(ys) - torch.min(ys)):
            return False

        # 检查theta的累计变化量
        cumulative_theta_variation = torch.sum(torch.abs(torch.diff(thetas)))
        if cumulative_theta_variation > theta_variation_threshold:
            return False

        # 检查起始和结束点的theta差
        if torch.abs(thetas[-1] - thetas[0]) > theta_diff_threshold:
            return False
        # 如果所有的检查都通过了，那么轨迹是直线
        return True



    def forward(self, z, init_state):
        zt = self.decode(z,init_state)
        # print('hidden state: {}'.format(z))
        # print('trajecotry: {}'.format(zt[0,:,1]))
        return zt
        return self.decode(z, init_state)

    def recalculate_theta_v(self, traj):
        # 假设 traj 是一个 N x 4 的 numpy 数组，每行表示一个状态 [x, y, theta, v]
        # 假设时间间隔 dt 是恒定的

        # 计算 theta
        # theta 是相邻点与x轴的夹角，用 arctan2 计算 y 分量和 x 分量的角度
        for i in range(len(traj) - 1):
            dx = traj[i+1, 0] - traj[i, 0]
            dy = traj[i+1, 1] - traj[i, 1]
            traj[i, 2] = torch.atan2(dy, dx)  # 更新 theta

        # 最后一个点的 theta 用倒数第二个点的 theta
        traj[-1, 2] = traj[-2, 2]

        # 计算 v
        # v 是两个相邻点之间的距离
        # 这里我们可以直接用 numpy 的 diff 函数来计算相邻两点间的差异
        # dx = torch.diff(traj[:, 0])
        # dy = torch.diff(traj[:, 1])
        # distances = torch.sqrt(dx**2 + dy**2)
        # speeds = np.append(distances, 0)  # 最后一个点的速度设为 0 或者用前一个点的速度
        # traj[:, 3] = speeds  # 更新 v

        return traj

    # def forward(self, z, init_state):
    #     # return self.decode(z, init_state)
    #     traj =  self.decode(z, init_state)
    #     is_straight = self.is_straight_trajectory(traj[0])
    #     if(is_straight):
    #         traj[0, :, 1] *= 0.1
    #         traj_new = self.recalculate_theta_v(traj[0])
    #         traj[0] = traj_new
    #     # print('x: {}'.format(traj[0,:,0]))
    #     # print('y: {}'.format(traj[0,:,1]))
    #     # print('theta: {}'.format(traj[0,:,2]))
    #     return traj