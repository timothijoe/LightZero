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
                transition_library[str(trans_key)] = transition_data 
        return transition_library

        

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> Any:
        #return self.extract_data[str(index)]['state'], self.extract_data[str(index)]['action']
        #return self.extract_data[str(index)]['observation']['birdview'].transpose((2, 0, 1)), self.extract_data[str(index)]['latent_action']
        return self.extract_data[str(index)]['observation']['birdview'].transpose((2, 0, 1)), self.extract_data[str(index)]['trajectory'][0], self.extract_data[str(index)]['trajectory'][:,:2] 
        #return self.extract_data[str(index)]['observation']['birdview'].transpose((2, 0, 1)), self.extract_data[str(index)]['observation']['vehicle_state'], self.extract_data[str(index)]['trajectory'][1:] 