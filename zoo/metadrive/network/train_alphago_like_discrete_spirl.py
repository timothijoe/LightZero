import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from easydict import EasyDict
from tqdm import tqdm
from spirl_model import SpirlEncoder
#from spirl_dataset import SPIRLDataset
from spirl_discrete_traj_dataset import SPIRLDataset
from utils import mk_logdir, loss_function, loss_function_class
from zoo.metadrive.utils.traj_decoder import VaeDecoder
from zoo.metadrive.utils.control_decoder import CCDecoder
from lzero.model.common import EZNetworkOutput, RepresentationNetwork
#from lzero.model.sampled_efficientzero_model import PredictionNetwork
from lzero.model.muzero_model import PredictionNetwork
from torch import nn


expert_dir = '/home/PJLAB/puyuan/hoffung/taecrl_data/straight'
expert_dir = '/home/zhoutong/hoffung/expert_data_collection/straight'
expert_dir = '/home/zhoutong/hoffung/expert_data_collection/expcc_straight'
expert_dir = '/home/hunter/hoffung/expert_data_collection/straight_aggresive/'
expert_dir = '/home/hunter/hoffung/expert_data_collection/straight_wild/'
expert_dir = '/home/hunter/hoffung/expert_data_collection/inter_wild/'
expert_dir = '/home/hunter/hoffung/expert_data_collection/inter_agressive/'
expert_dir = '/home/hunter/hoffung/expert_data_collection/compare_straight_aggresive/'
# expert_dir = '/home/zhoutong/hoffung/expert_data_collection/round'
# expert_dir = '/home/zhoutong/hoffung/expert_data_collection/inter'
metadrive_basic_config = dict(
    #exp_name = 'metadrive_train_expert_straight_aggresive',
    exp_name = 'metadrive_zt3_discrete',
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            batch_size=64,
            learning_rate=3e-5,
            lr=1e-4,
            epoches=200,
            epoch_per_save = 10,
        ),
    ),
)
main_config = EasyDict(metadrive_basic_config)


class ContinousEncoder(nn.Module):
    r"""
    Overview:
        The ``Convolution Encoder`` used in models. Used to encoder raw 2-dim observation.
    Interfaces:
        ``__init__``, ``forward``
    """
    def __init__(
            self,
    ) -> None:
        super(ContinousEncoder, self).__init__()
        self.representation_network = RepresentationNetwork(
            [5, 200, 200],
            1,
            64,
            True,
            norm_type='BN',
        )
        self.prediction_network = PredictionNetwork(
            [5,200,200],
            49,
            1,
            64,
            16,
            16,
            [32],
            [32],
            601,
            2704,
            2704,
            True,
            last_linear_layer_init_zero=True,
            activation = nn.ReLU(inplace=True),
            norm_type='BN',
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_state = self.representation_network(x)
        policy_logits, value = self.prediction_network(latent_state)
        return policy_logits


zt_traj_decoder = CCDecoder(
    control_num = 2,
    seq_len = 1,
    use_relative_pos = True,
    dt = 0.1,
    traj_control_mode = 'acc',
    steer_rate_constrain_value = 0.5,
)

def main(cfg):
    mk_logdir(cfg.exp_name)
    tb_logger = SummaryWriter('result/{}/log/'.format(cfg.exp_name))
    for param in zt_traj_decoder.parameters():
        param.requires_grad = False
    # model = SpirlEncoder(**cfg.policy.model)
    model = ContinousEncoder()
    train_dataset = SPIRLDataset(expert_dir)
    train_loader = DataLoader(train_dataset, cfg.policy.learn.batch_size, shuffle=True, num_workers=8)
    optimizer = Adam(model.parameters(), lr=cfg.policy.learn.lr)
    iter_num = 0
    represent_state_dict = model.representation_network.state_dict()
    pred_state_dict = model.prediction_network.state_dict()
    torch.save(represent_state_dict, "result/{}/ckpt/represent_{}_ckpt".format(cfg.exp_name, 0))
    torch.save(pred_state_dict, "result/{}/ckpt/pred_{}_ckpt".format(cfg.exp_name, 0))

    for epoch in range(cfg.policy.learn.epoches):
        model.train()
        decs = 'Train - epoch-{}'.format(epoch)
        sub_iter = 0
        model.train()
        epoch_loss = 0
        for data_state, gt_label in tqdm(train_loader):
            data_state = data_state.to(torch.float32)
            gt_label = gt_label.to(torch.long)
            pred_action = model(data_state)
            loss = loss_function_class(pred_action, gt_label)
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