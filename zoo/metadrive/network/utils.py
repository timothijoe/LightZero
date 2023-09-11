import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from torch.distributions import Normal, Independent
import os


def mk_logdir(exp_name):
    path1 = 'result'
    path2 = 'result/{}/ckpt'.format(exp_name)
    path3 = 'result/{}/log'.format(exp_name)
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    if not os.path.exists(path3):
        os.makedirs(path3)

def loss_function(pred_action, latent_action):
    # latent_action_predict, latent_action_gt
    pred_action = pred_action
    gt_action = latent_action
    recons_loss = F.mse_loss(pred_action, gt_action)
    loss = recons_loss 
    return loss 

def loss_function_class(pred, gt_class):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, gt_class)
    return loss 