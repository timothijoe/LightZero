import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from easydict import EasyDict
from tqdm import tqdm
from spirl_model import SpirlEncoder
#from spirl_dataset import SPIRLDataset
from spirl_traj_dataset import SPIRLDataset
from utils import mk_logdir, loss_function
from zoo.metadrive.utils.traj_decoder import VaeDecoder

expert_dir = '/home/PJLAB/puyuan/hoffung/taecrl_data/straight'
expert_dir = '/home/zhoutong/hoffung/expert_data_collection/straight'
expert_dir = '/home/zhoutong/hoffung/expert_data_collection/expcc_straight'
expert_dir = '/home/hunter/hoffung/expert_data_collection/straight_aggresive/'
expert_dir = '/home/hunter/hoffung/expert_data_collection/straight_wild/'
# expert_dir = '/home/zhoutong/hoffung/expert_data_collection/round'
# expert_dir = '/home/zhoutong/hoffung/expert_data_collection/inter'
metadrive_basic_config = dict(
    #exp_name = 'metadrive_train_expert_straight_aggresive',
    exp_name = 'metadrive_train_expert_straight_wild',
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=3,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            batch_size=64,
            learning_rate=3e-4,
            lr=1e-4,
            epoches=200,
            epoch_per_save = 10,
        ),
    ),
)
main_config = EasyDict(metadrive_basic_config)

zt_traj_decoder = VaeDecoder(
    embedding_dim = 64,
    h_dim = 64,
    latent_dim = 3,
    seq_len = 10,
    dt = 0.1,
    traj_control_mode = 'acc',
    one_side_class_vae=False,
    steer_rate_constrain_value=0.5,
)
vae_load_dir = '/home/hunter/hoffung/LightZero/zoo/metadrive/model/nov02_len10_dim3_v1_ckpt'
zt_traj_decoder.load_state_dict(torch.load(vae_load_dir,map_location=torch.device('cpu')))



def main(cfg):
    mk_logdir(cfg.exp_name)
    tb_logger = SummaryWriter('result/{}/log/'.format(cfg.exp_name))
    for param in zt_traj_decoder.parameters():
        param.requires_grad = False
    model = SpirlEncoder(**cfg.policy.model)
    train_dataset = SPIRLDataset(expert_dir)
    train_loader = DataLoader(train_dataset, cfg.policy.learn.batch_size, shuffle=True, num_workers=8)
    optimizer = Adam(model.parameters(), lr=cfg.policy.learn.lr)
    iter_num = 0

    for epoch in range(cfg.policy.learn.epoches):
        model.train()
        decs = 'Train - epoch-{}'.format(epoch)
        sub_iter = 0
        model.train()
        epoch_loss = 0
        for data_state, data_vehicle_state, gt_trajs in tqdm(train_loader):
            data_state = data_state.to(torch.float32)
            data_vehicle_state = data_vehicle_state.to(torch.float32)
            gt_trajs = gt_trajs.to(torch.float32)

            pred_action = model(data_state)
            init_state = data_vehicle_state
            traj = zt_traj_decoder(pred_action, init_state)
            init_state = init_state[:,:4]
            traj = torch.cat([init_state.unsqueeze(1), traj], dim = 1)
            valid_traj = traj 
            traj = traj[:,:,:2]


            loss = loss_function(traj, gt_trajs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss 
            tb_logger.add_scalar("train_iter/{}".format('MSE loss'), loss.item(), iter_num)
            iter_num += 1
        tb_logger.add_scalar("train_epoch/{}".format('MSE loss'), epoch_loss.item(), epoch)
        if(epoch % cfg.policy.learn.epoch_per_save == 0):
            state_dict = model.state_dict()
            torch.save(state_dict, "result/{}/ckpt/{}_ckpt".format(cfg.exp_name, epoch))


if __name__ == '__main__':
    main(main_config)