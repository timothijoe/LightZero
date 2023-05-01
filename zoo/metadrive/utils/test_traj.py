import torch
from zoo.metadrive.utils.traj_decoder import VaeDecoder
# 创建一个形状为[1,3]的张量并将其移到CUDA设备上
latent_action = torch.zeros((1, 3), device='cuda')

# 创建一个形状为[1,6]的张量并将其移到CUDA设备上
init_state = torch.zeros((1, 6), device='cuda')

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
# traj.shape 1, 11, 4
print('zt') 