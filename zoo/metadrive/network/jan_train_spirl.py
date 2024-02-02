import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from easydict import EasyDict
from tqdm import tqdm
from spirl_model import SpirlEncoder
from spirl_dataset import SPIRLDataset
from utils import mk_logdir, loss_function

expert_dir = '/home/PJLAB/puyuan/hoffung/taecrl_data/straight'
expert_dir = '/home/zhoutong/hoffung/expert_data_collection/straight'
expert_dir = '/home/hunter/hoffung/mask_folder/'
expert_dir = '/home/rpai_lab_server_1/dec_jan/data_related/xad_expert_data/'
metadrive_basic_config = dict(
    exp_name = 'metadrive_train_expert',
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




def main(cfg):
    mk_logdir(cfg.exp_name)
    tb_logger = SummaryWriter('result/{}/log/'.format(cfg.exp_name))
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
        for data_state, latent_action in tqdm(train_loader):
            pred_action = model(data_state)
            loss = loss_function(pred_action, latent_action)
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