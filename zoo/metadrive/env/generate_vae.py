import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import math 
from torch.distributions import Normal, Independent


def compute_theta_change(traj1):
    # traj = np.transpose(traj1)
    traj = traj1
    thetas = traj[:, 2]  # 提取轨迹中的角度信息
    theta_change = np.diff(thetas)  # 计算每个时刻的角度变化量
    theta_list = list(thetas)
    max_theta = max(theta_list)
    min_theta = min(theta_list)
    mean_theta = np.mean(theta_list)
    mid_theta = thetas[10]
    last_theta = thetas[-1]
    theta_info = np.array([max_theta, min_theta, mean_theta, mid_theta, last_theta])
    # 返回直方图信息
    hist, bins = np.histogram(theta_change, bins=30, range=(-15, 15))
    lhist = list(hist)
    lhist_sum = np.sum(lhist)+1
    normalized_hist = hist / lhist_sum 
    return normalized_hist, theta_info

class VaeEncoder2(nn.Module):
    def __init__(self,
        embedding_dim = 64,
        h_dim = 64,
        latent_dim = 100,
        seq_len = 30,
        use_relative_pos = True,
        dt = 0.03,
        ):
        super(VaeEncoder2, self).__init__()
        self.encoding_len = seq_len if seq_len == 20 else seq_len * 2
        self.embedding_dim = embedding_dim
        self.label_dim = 2
        self.h_dim = h_dim 
        self.num_layers = 1
        self.latent_dim = latent_dim
        self.seq_len = seq_len 
        self.use_relative_pos = use_relative_pos
        self.dt = dt
        self.device = torch.device('cpu') #torch.device('cuda:0')

        # input: x, y, theta, v,   output: embedding
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)

        enc_mid_dims = [self.h_dim, self.h_dim, self.h_dim, self.latent_dim]
        enc_mid_dims2 = [self.h_dim, self.h_dim, self.h_dim, int(self.embedding_dim / 4)]
        mu_modules = []
        sigma_modules = []
        cur_hist_list = []
        theta_info_list = []
        in_channels = self.h_dim
        for m_dim in enc_mid_dims:
            mu_modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    #nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            sigma_modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    #nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = m_dim  
        in_channels = 30
        for m_dim in enc_mid_dims2:
            cur_hist_list.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    #nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            theta_info_list.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    #nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = m_dim  

        self.mean = nn.Sequential(*mu_modules) 
        self.log_var = nn.Sequential(*sigma_modules)
        self.cur_net = nn.Sequential(*cur_hist_list)
        self.theta_net = nn.Sequential(*theta_info_list)
        self.encoder = nn.LSTM(self.embedding_dim + int(0.5 * self.embedding_dim), self.h_dim, self.num_layers)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.h_dim).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.h_dim).to(self.device)
        )
    def get_relative_position(self, abs_traj):
        # abs_traj shape: batch_size x seq_len x 4
        # rel traj shape: batch_size x seq_len -1 x 2
        rel_traj = abs_traj[:, 1:, :2] - abs_traj[:, :-1, :2]
        rel_traj = torch.cat([abs_traj[:, 0, :2].unsqueeze(1), rel_traj], dim = 1)
        rel_traj = torch.cat([rel_traj, abs_traj[:,:,2:]],dim=2)
        #rel_traj = torch.cat([rel_traj, abs_traj[:,:,2:].unsqueeze(2)],dim=2)
        # rel_traj shape: batch_size x seq_len x 4
        return rel_traj

    def encode(self, input, curve_hist, theta_info):
        # input meaning: a trajectory len 25 and contains x, y , theta, v
        # input shape: batch x seq_len x 4
        #data_traj shape: seq_len x batch x 4
        if self.use_relative_pos:
            input = self.get_relative_position(input)
            input = input[:,:,:2]

        curve_feature = self.cur_net(curve_hist)
        theta_feature = self.theta_net(theta_info)
        curve_feature = curve_feature.repeat(self.seq_len, 1, 1)
        theta_feature = theta_feature.repeat(self.seq_len, 1, 1)
        # traj_label_onehot = one_hot(traj_label.long(),num=2).unsqueeze(0)
        # traj_label_onehot = traj_label_onehot.repeat(self.seq_len, 1, 1)
        data_traj = input.permute(1, 0, 2).contiguous()
        traj_embedding = self.spatial_embedding(data_traj.view(-1, 2))
        # traj_embedding = traj_embedding.view(self.seq_len, -1, self.embedding_dim)
        traj_embedding = traj_embedding.view(self.encoding_len, -1, self.embedding_dim)
        traj_embedding = traj_embedding[:self.seq_len]
        # Here we do not specify batch_size to self.batch_size because when testing maybe batch will vary
        batch_size = traj_embedding.shape[1]
        hidden_tuple = self.init_hidden(batch_size)
        traj_embedding = torch.cat([theta_feature, theta_feature , traj_embedding], 2)
        output, encoder_h = self.encoder(traj_embedding, hidden_tuple)
        mu = self.mean(encoder_h[0])
        log_var = self.log_var(encoder_h[0])
        #mu, log_var = torch.tanh(mu), torch.tanh(log_var)
        return mu, log_var

    def forward(self, input, curve_hist, theta_info):
        return self.encode(input, curve_hist, theta_info)


class VaeDecoder2(nn.Module):
    def __init__(self,
        embedding_dim = 64,
        h_dim = 64,
        latent_dim = 3,
        seq_len = 20,
        use_relative_pos = True,
        dt = 0.1,
        steer_rate_constrain_value = 0.4,
        ):
        super(VaeDecoder2, self).__init__()
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
        traj =  self.decode(z, init_state)
        return traj


def get_auto_encoder2(vae_encoder, vae_decoder, expert_traj, init_state, curve_hist, theta_info):
    mu, logvar = vae_encoder(expert_traj, curve_hist, theta_info)
    z = torch.tanh(mu)
    recons_traj = vae_decoder(z, init_state)

def get_auto_encoder(vae_encoder, vae_decoder, expert_traj):
    init_state = expert_traj[0,:]
    curve_hist, theta_info = compute_theta_change(expert_traj)
    theta_info = np.tile(theta_info, 6)
    expert_traj = expert_traj[1:, :]
    init_state = np.expand_dims(init_state, axis=0)
    curve_hist = np.expand_dims(curve_hist, axis=0)
    theta_info = np.expand_dims(theta_info, axis=0)
    expert_traj = np.expand_dims(expert_traj, axis=0)
    init_state = torch.from_numpy(init_state).float()
    curve_hist = torch.from_numpy(curve_hist).float()
    theta_info = torch.from_numpy(theta_info).float()
    expert_traj = torch.from_numpy(expert_traj).float()
    

    with torch.no_grad():
        mu, logvar = vae_encoder(expert_traj, curve_hist, theta_info)
        z = torch.tanh(mu)
        recons_traj = vae_decoder(z, init_state)
    recons_traj = recons_traj.numpy()
    init_state_cpu = init_state.numpy()
    init_state_cpu = np.expand_dims(init_state_cpu, axis=1)
    traj = np.concatenate((init_state_cpu, recons_traj), axis=1)
    traj = traj[0]
    return traj