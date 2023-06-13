from spirl_model import SpirlEncoder
from spirl_dataset import SPIRLDataset
from torch.utils.data import DataLoader
from collections import defaultdict
import os
import numpy as np
from ding.utils.data.collate_fn import default_collate, default_decollate
from easydict import EasyDict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim import Adam
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import SACPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, NaiveReplayBuffer


metadrive_basic_config = dict(
    exp_name = 'metadrive_basic_sac',
    policy=dict(
        cuda=False,
        on_policy=False,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=3,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            multi_gpu=False,
            init_w = False,
            lr=1e-4,
            epoches=200,
        ),
        collect=dict(
            n_sample=50,
            unroll_len = 1,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=10,
                )
            ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=100000,
            ),
        ), 
        priority= False,
        priority_IS_weight=False,
    )
)

def train(model, optimizer, loader, tb_logger=None, start_iter=0):
    loss_epoch = defaultdict(list)
    iter_num = start_iter

    for data_state, latent_action in tqdm(loader):
        log_vars = model(data_state)
    #     print(log_vars)
    #     optimizer.zero_grad()
    #     total_loss = log_vars['total_loss']
    #     total_loss.backward()
    #     optimizer.step()
    #     log_vars['cur_lr'] = optimizer.defaults['lr']
    #     for k, v in log_vars.items():
    #         loss_epoch[k] += [log_vars[k].item()]
    #         if iter_num % 50 == 0 and tb_logger is not None:
    #             tb_logger.add_scalar("train_iter/" + k, v, iter_num)
    #     iter_num += 1
    # loss_epoch = {k: np.mean(v) for k, v in loss_epoch.items()}
    return iter_num, loss_epoch


main_config = EasyDict(metadrive_basic_config)
cfg = main_config
model = SpirlEncoder(**cfg.policy.model)
expert_dir = '/home/PJLAB/puyuan/hoffung/taecrl_data/straight'
train_dataset = SPIRLDataset(expert_dir)



train_loader = DataLoader(train_dataset, cfg.policy.learn.batch_size, shuffle=True, num_workers=8)
optimizer = Adam(model.parameters(), cfg.policy.learn.lr)
tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
iterations = 0
best_loss = 1e8
start_epoch = 0



iterations = 0
best_loss = 1e8
start_epoch = 0

# if cfg.policy.resume:
#     start_epoch, iterations, best_loss = load_best_ckpt(
#         cilrs_policy.learn_mode, optimizer, exp_name=cfg.exp_name, ckpt_path=cfg.policy.ckpt_path
#     )

for epoch in range(start_epoch, cfg.policy.learn.epoches):
    iter_num, loss = train(model, optimizer, train_loader, tb_logger, iterations)
    iterations = iter_num
    print('zt1')


print('zt')