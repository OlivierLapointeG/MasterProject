import torch
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from utils import fast_extract_control_nodes_features, dirichlet_energy

class EdgeControl(nn.Module):
    def __init__(self, conv, conv_type='GCN', activation=nn.ReLU()):
        super(EdgeControl, self).__init__()
        self.conv = conv
        self.activation = activation
        self.conv_type = conv_type

    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        if self.conv_type == 'GAT':
            X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = self.activation(self.conv(X, edge_index))
        gg = torch.tanh(scatter((torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
                                 edge_index[0], 0,dim_size=X.size(0), reduce='mean'))

        return gg