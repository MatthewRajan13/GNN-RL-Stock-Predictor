import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GNNAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GNNAutoEncoder, self).__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, latent_dim)
        self.conv3 = SAGEConv(latent_dim, hidden_dim)
        self.conv4 = SAGEConv(hidden_dim, input_dim)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        z = F.relu(self.conv3(z, edge_index))
        return self.conv4(z, edge_index)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_hat = self.decode(z, edge_index)
        return x_hat