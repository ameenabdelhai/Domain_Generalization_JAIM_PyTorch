import torch
import torch.nn as nn

class BaseNet_clstm(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BaseNet_clstm, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.4)


    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to( torch.device("cpu") )
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to( torch.device("cpu") )

        # forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        ) # out: tensor of shape (batch_size, sequence length, hidden_size)

        # decode the hidden state of the last hidden state
        out = out.reshape(out.shape[0], -1)
        out = self.relu(out)
        return out








