import os, numpy as np, copy
import sys 
sys.path.append("..")
import torch
import pandas as pd
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import random
import os
import torch
from torch.utils.data import Dataset

class Supermarket(Dataset):
    def __init__(self, subset, past_length, future_length, phase="train"):
        self.subset = subset
        self.dataset_directory = f"supermarket/dataset/preprocessed_dataset/{self.subset}"
        self.data_file = self.dataset_directory + f'/{phase}_data.npy'
        self.num_file = self.dataset_directory + f'/{phase}_num.npy'
        self.past_length = past_length
        self.future_length = future_length

        # Check if the file exists
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"No file found at {self.data_file} \n Consider Preprocessing")
        if not os.path.exists(self.num_file):
            raise FileNotFoundError(f"No file found at {self.num_file} \n Consider Preprocessing")

        # Load the data from file
        self.dataset = np.load(self.data_file)
        self.num_agent = np.load(self.num_file)
        self.trajectory_batches = self.dataset[:,:,:20,:]
        self.initial_pos_batches = self.dataset[:, :, self.past_length-1, :].copy() / 1000
        self.mask_batches = np.ones((self.dataset.shape[0], 1))

    # def __getitem__(self, index):
    #     trajectory = self.dataset[index]
    #     # print("inside getitem ", trajectory.shape, self.dataset.shape)
    #     past = trajectory[:, :self.past_length]
    #     future = trajectory[:, self.past_length:self.past_length+self.future_length]
    #     num_agent = self.num_agent[index]

    #     # Convert to tensor and return
    #     return past, future, num_agent

    # def __len__(self):
    #     return len(self.dataset)
    

class Data2Numpy:
    def __init__(self, subset, past_length, future_length, split):
        self.subset = subset
        self.past_length = past_length
        self.future_length = future_length
        self.split = split
        self.processed_data_folder_path = "./preprocessed_dataset/" + self.subset
        self.data_path = f"./supermarket/{self.subset}"

        assert subset in ['german_1', 'german_2','german_3', 'german_4']


    def generate_data(self):
        # data is of the format [tag_id', 'time', 'x', 'y', 'description', 'trajectory_name', 'point_type']
        # extract only tag_id, x, y, trajectory_name from self.data
        directory_path = self.processed_data_folder_path
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")

        for split in list(self.split.keys()):
            data = self.split[split][['tag_id', 'x', 'y', 'trajectory_name']].copy()
            grouped_data = data.groupby(['tag_id', 'trajectory_name'])[['x', 'y']].apply(lambda x: x.values.tolist()).to_dict()
            data_len = len(grouped_data)
            # considering that this dataset has only one agent
            num_data = np.ones((data_len))
            # shape of dataset = length of data, number of agents(here one), total trajectory length, 2 coordinates
            dataset = np.zeros((data_len, 1,  self.past_length + self.future_length, 2))
            for i, key in enumerate(grouped_data.keys()):
                trajectory = grouped_data[key]
                if len(trajectory) < self.past_length + self.future_length:
                    continue
                else:
                    dataset[i] = trajectory[:self.past_length + self.future_length]
            np.save(f"{self.processed_data_folder_path}/{split}_data.npy", dataset)
            np.save(f"{self.processed_data_folder_path}/{split}_num.npy", num_data)




