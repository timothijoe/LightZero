import os
import numpy as np
from typing import Any, Dict
import torch
from torch.utils.data import Dataset
import os
import pickle
import copy


class SPIRLDataset(Dataset):

    def __init__(self, expert_dir = None) -> None:
        self.expert_dir = expert_dir 
        self.extract_data = self.read_data()
        self.len = len(self.extract_data)

    def get_file_list(self):
        episode_data_list = []
        file_list = []
        for cur_file in os.listdir(self.expert_dir):
            cur_file_path = os.path.join(self.expert_dir, cur_file)
            file_list.append(cur_file_path)
        for file in file_list:
            with open(file, 'rb') as f:
                episode_data = pickle.load(f)
                episode_data_list.append(episode_data)
        return episode_data_list

    def read_data(self):
        self.training__files = []
        transition_library = {}
        episode_data_list = self.get_file_list()
        for episode_data in episode_data_list:
            self.last_traj = None 
            self.last_obs = None 
            zt_num = 0
            for transition_data in episode_data['transition_list']:
                if zt_num == 0:
                    pass 
                else:
                    new_trans = {}
                    new_trans['observation'] = self.last_obs 
                    previous_traj_fine = copy.deepcopy(self.last_traj)
                    consecutive_traj = transition_data['tftraj']
                    consecutive_traj_fine = consecutive_traj[1:]
                    total_traj = np.concatenate((previous_traj_fine, consecutive_traj_fine), axis=0)
                    new_trans['trajectory'] = total_traj 
                self.last_traj = transition_data['raw_traj']
                self.last_obs = transition_data['observation']
                trans_key = len(transition_library)
                transition_library[str(trans_key)] = transition_data 
                zt_num += 1 
        return transition_library

        

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> Any:
        #return self.extract_data[str(index)]['state'], self.extract_data[str(index)]['action']
        return self.extract_data[str(index)]['observation']['birdview'].transpose((2, 0, 1)), self.extract_data[str(index)]['latent_action']