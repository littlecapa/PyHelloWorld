import logging
from torch.utils.data import Dataset
import torch

class HW_Data_Loader(Dataset):

    def __init__(self, data_len = 1000, noise = 0.0):
        self.load_data(data_len, noise)

    def __len__(self):
        return len(self.y_value)

    def __getitem__(self, idx):
        return self.X_value[idx], self.y_value[idx]
    
    def load_data(self, data_len = 1000, noise = 0.0):
        self.X_value = torch.rand((data_len,2))
        x1 = self.X_value[:,0]
        x2 = self.X_value[:,1]
        self.y_value = x1 * x2

    def get_baseline_test_data(self):
        X_list = torch.tensor([[0.3651, 0.6404], [0.1234, 0.5678], [0.9876, 0.5432]])
        x1 = X_list[:,0]
        x2 = X_list[:,1]
        y_list = x1*x2
        return X_list, y_list
