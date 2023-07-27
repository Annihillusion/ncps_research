import numpy as np
import os
from torch.utils.data import Dataset

class NeuronDataset(Dataset):
    def __init__(self, root_dir="E:/Celegans-ForwardCrawling-RNNs/Dataset1"):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        if len(self.file_list) == 0:
            raise RuntimeError("Could not find data")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.root_dir + '/' + self.file_list[idx]
        data = np.genfromtxt(path, delimiter=' ')
        # nin = ['time', 'PLML2', 'PLMR', 'AVBL', 'AVBR']
        # nout = ['time', 'DB1', 'LUAL', 'PVR', 'VB1']
        # neurons = ['time', 'DB1', 'LUAL', 'PVR', 'VB1', 'PLML2', 'PLMR', 'AVBL', 'AVBR']
        x = data[:, -4:]
        y = data[:, 1:5]
        return x, y



if __name__ == '__main__':
    a = np.genfromtxt('E:\Celegans-ForwardCrawling-RNNs\Dataset1\Sequence_01.dat', delimiter=' ')
    print(0)