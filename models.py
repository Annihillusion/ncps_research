import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ncps.torch import LTC
from ncps.wirings import AutoNCP, FullyConnected


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 64, 5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 5, padding=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.mean((-1, -2))  # Global average pooling
        return x


class ConvLTC(nn.Module):
    def __init__(self, n_neurons=64, n_actions=4, connect_policy='ncp'):
        super().__init__()
        self.conv_block = ConvBlock()
        if connect_policy == 'ncp':
            self.rnn = LTC(256, AutoNCP(n_neurons, n_actions), batch_first=True)
        elif connect_policy == 'fc':
            self.rnn = LTC(256, FullyConnected(n_neurons, n_actions), batch_first=True)
        else:
            raise ValueError("Choose from 'fc' or 'ncp'!")

    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # Merge time and batch dimension into a single one (because the Conv layers require this)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)  # apply conv block to merged data
        # Separate time and batch dimension again
        x = x.view(batch_size, seq_len, *x.shape[1:])
        x, hx = self.rnn(x, hx)  # hx is the hidden state of the RNN
        return x, hx