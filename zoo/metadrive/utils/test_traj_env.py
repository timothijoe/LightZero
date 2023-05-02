import metadrive  # Import this package to register the environment!
import gym
import torch

from metadrive.envs.metadrive_env import MetaDriveEnv
import sys 
sys.path.append('/home/zhoutong/osiris/LightZero')
from zoo.metadrive.utils.traj_decoder import VaeDecoder
latent_action = torch.zeros((1, 3), device='cuda')

# 创建一个形状为[1,6]的张量并将其移到CUDA设备上
init_state = torch.zeros((1, 6), device='cuda')
init_state[0][4] = 3

_traj_decoder = VaeDecoder(
    embedding_dim = 64,
    h_dim = 64,
    latent_dim = 3,
    seq_len = 10,
    dt = 0.1,
    traj_control_mode = 'acc',
    one_side_class_vae=False,
    steer_rate_constrain_value=0.5,
)
_traj_decoder = _traj_decoder.to('cuda')
vae_load_dir = '/home/SENSETIME/zhoutong/osiris/LightZero/zoo/metadrive/model/nov02_len10_dim3_v1_ckpt'
_traj_decoder.load_state_dict(torch.load(vae_load_dir))
traj = _traj_decoder(latent_action, init_state)
init_state = init_state[:,:4]
traj = torch.cat([init_state.unsqueeze(1), traj], dim = 1)
traj = traj[0,:,:2]
traj_cpu = traj.detach().to('cpu').numpy()





TRAJ_CONTROL_MODE = 'acc'
SEQ_TRAJ_LEN = 10
metadrive_config=dict(use_render=True,
    show_seq_traj = True,
    traffic_density = 0.30, #0.20
    # need_inverse_traffic=True, #True

    seq_traj_len = SEQ_TRAJ_LEN,
    traj_control_mode = TRAJ_CONTROL_MODE,
    
    # map='SOSO', 
    #map='OSOS',
    # map='SXSX',
    # map = 'XSXS',
    # map='SXSX',
    # enable_u_turn = True,
    # traffic_mode=TrafficMode.Trigger,
    avg_speed = 6.0,


    #show_interface=False,
    use_lateral=True,
    use_speed_reward = True,
    use_heading_reward = True,
    use_jerk_reward = True,
    heading_reward=0.15,
    speed_reward = 0.05,
    driving_reward = 0.2,
    jerk_bias = 10,

    crash_vehicle_penalty = 4.0,
    out_of_road_penalty = 5.0,
    debug_info=True,
    ignore_first_steer = False,
    zt_mcts = True,
)



# import zoo.metadrive.env.traj_env
from zoo.metadrive.env.traj_env import MetaDriveTrajEnv
# env = MetaDriveEnv(config=dict(use_render=True))
env = MetaDriveTrajEnv(config=metadrive_config)
# env = gym.make("MetaDrive-10env-v0", config=dict(use_render=True))
env.reset()
latent_action = latent_action.cpu().numpy()
for i in range(100):
    #obs, reward, done, info = env.step(traj_cpu)
    obs, reward, done, info = env.step(latent_action)
    env.render()
    if done:
        env.reset()
env.close()