import os
import numpy as np
from typing import Any, Dict
import torch
from torch.utils.data import Dataset
import os
import pickle

class SPIRLDataset(Dataset):

    def __init__(self, expert_dir = None) -> None:
        self.expert_dir = expert_dir 
        self.extract_data = self.read_data()
        self.len = len(self.extract_data)
        self.pattern = 'continuous'

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
            for transition_data in episode_data['transition_list']:
                trans_key = len(transition_library)
                # transition_library[str(trans_key)] = transition_data 
                v1 = transition_data['trajectory'][1,3]
                v0 = transition_data['trajectory'][0,3]
                acc =  v1 - v0
                acc = acc / 0.1 
                acc_clamp = np.clip(acc, -5, 5)
                dtheta = transition_data['trajectory'][1,2]
                theta_dot = dtheta / 0.1 
                if np.abs(v1) < 0.05:
                    steer = 0.0
                else:
                    steer = np.arctan(2.5 * theta_dot / np.abs(v1))
                steer_clamp = np.clip(steer, -0.5, 0.5)
                acc_unify = acc_clamp / 5.0
                steer_unify = steer_clamp / 0.5
                lst = [-1.0, -0.66, -0.33, 0, 0.33, 0.66, 1]
                nearest_index_acc = min(range(len(lst)), key=lambda i: abs(lst[i] - acc_unify))   
                nearest_index_steer = min(range(len(lst)), key=lambda i: abs(lst[i] - steer_unify))     
                zt_num = 7
                final_index = nearest_index_acc * zt_num + nearest_index_steer
                transition_data['class_label'] = final_index 
                transition_library[str(trans_key)] = transition_data
        return transition_library

        

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> Any:
        return self.extract_data[str(index)]['observation']['birdview'].transpose((2, 0, 1)), self.extract_data[str(index)]['class_label']
        return self.extract_data[str(index)]['observation']['birdview'].transpose((2, 0, 1)), self.extract_data[str(index)]['trajectory'][0] + zt, self.extract_data[str(index)]['trajectory'][:,:2]