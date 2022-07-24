import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.parameter import Parameter
from torch_geometric.nn import GAE, SAGEConv

GRAPH_ATTRIBUTES = 6
HIDDEN_DIM = 64

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, 2 * in_channels)
        self.conv2 = SAGEConv(2 * in_channels, 4 * in_channels)
        self.conv3 = SAGEConv(4 * in_channels, out_channels)

    def forward(self, x, edge_index):
        if edge_index.size()[1] == 0:  # Empty edge index
            edge_index = torch.tensor([[0], [0]])
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        return x


class GNNDecoder(nn.Module):
    def __intit__(self, in_channels, out_channels):
        super(GNNDecoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, 2 * in_channels)
        self.conv2 = SAGEConv(2 * in_channels, 4 * in_channels)
        self.conv3 = SAGEConv(4 * in_channels, out_channels)

    def forward(self, x, edge_index):
        if edge_index.size()[1] == 0:  # Empty edge index
            edge_index = torch.tensor([[0], [0]])
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        return x


class GNNAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GAE(GNNEncoder(GRAPH_ATTRIBUTES, GRAPH_ATTRIBUTES), decoder=GNNDecoder(GRAPH_ATTRIBUTES,
                                                                                            GRAPH_ATTRIBUTES))
        self.model.eval()

    def init_hidden(self):
        return Parameter(torch.zeros(1, HIDDEN_DIM))

    def forward(self, data):
        a = self.mode.encode(data.x, data.edge_index)
        return a[0], a[1]