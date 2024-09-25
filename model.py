import torch
import torch.nn as nn



class MMDNet(nn.Module):
    def __init__(self):
        super(MMDNet, self).__init__()
        self.fc1 = nn.Linear(in_features=128*100, out_features=100)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu1(self.fc1(x))
        return x


class PredictNet(nn.Module):
    # predictions
    def __init__(self):
        super(PredictNet, self).__init__()
        self.fc1 = nn.Linear(in_features=100, out_features=1)
        # self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = x.view(x.shape[0], -1)
        return x
