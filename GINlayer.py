import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class CustomGINLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, weighted, mlp_depth):
        super(CustomGINLayer, self).__init__(aggr="add")  # 'add' aggregation method
        # Learnable parameters: weights for message passing
        self.in_channels = in_channels
        self.weighted = weighted
        mlp = [nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()]
        for i in range(mlp_depth-1):
            mlp.append(nn.Linear(out_channels, out_channels))
            mlp.append(nn.BatchNorm1d(out_channels))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x, edge_index, weightedEdges,*kargs, residual=True, self_loop=True):
        # x: Node features, shape (N, in_channels)
        # edge_index: Edge index, shape (2, E) for undirected graph

        # Add self-loops for the graph (optional but common in GIN)
        if self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        skip = x
        # Node self-information (initial embedding)
        out = self.mlp(self.propagate(edge_index, x=x, edge_weights=weightedEdges))

        # Message passing step
        return out + residual*skip

    def message(self, x_j, edge_weights):
        # x_j: Node features of neighboring nodes
        # Message passing: Aggregate neighbor node features with weights
        
        if self.weighted:
            if int(x_j.shape[0]) != int(edge_weights.shape[0]):
                return x_j    
            print('Multiplying edge weights with node features !')
            return edge_weights.view(-1, 1) * x_j
        return  x_j

    def update(self, aggr_out):
        # aggr_out: Aggregated messages
        # Combine aggregated messages with node self-information
        return aggr_out

