import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class convnet1(nn.Module):
    def __init__(self):

        super(convnet1, self).__init__()
        # model
        self.conv1 = nn.Conv2d(1, 10, (15, 3))
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.bn2 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(400, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 1)


    def forward(self, x, istraining=False, minibatch=1):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (3, 1))
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(self.bn2(x)), (3, 1))
        x = F.dropout(x.view(minibatch, -1), training=istraining)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=istraining)
        x = F.dropout(F.relu(self.fc2(x)), training=istraining)

        return F.sigmoid(self.fc3(x))

class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(128, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        fc_out = self.fc(r_out[:, -1, :]).squeeze()
        return self.out(fc_out)