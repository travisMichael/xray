import os
import numpy as np
import torch

class XrayLoader():
    def __init__(self, path):
        self.path = path
        self.directory_list = os.listdir(path)
        self.size = len(self.directory_list)
        self.index = 0


    def get_next_batch(self):
        np_data = np.load(self.path + self.directory_list[self.index] + "/data.npy")
        np_labels = np.load(self.path + self.directory_list[self.index] + "/labels.npy")
        self.index += 1

        data = torch.tensor(np_data, dtype=torch.float)
        labels = torch.tensor(np_labels, dtype=torch.long)
        return (data, labels)

    def reset(self):
        self.index = 0