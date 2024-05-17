import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import MessagePassing,global_mean_pool, global_add_pool, global_max_pool, GINConv, GCNConv, GATConv, ResGatedGraphConv, SGConv
from torch_geometric.utils import add_self_loops, degree
from utils import customSigmoid, dirichlet_energy
from GINlayer import CustomGINLayer
from GCNlayer import CustomGCNLayer
import time


class TestingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, weighted):
        super(TestingLayer, self).__init__(aggr="add")  # 'add' aggregation method
        # Learnable parameters: weights for message passing
        self.in_channels = in_channels
        self.weighted = weighted
        self.edge_index = None
        self.noweighted = []

    def forward(self, x, edge_index, weightedEdges, *kargs):
        # x: Node features, shape (N, in_channels)
        # edge_index: Edge index, shape (2, E) for undirected graph

        self.edge_index = edge_index
        # Add self-loops for the graph (optional but common in GIN)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]


        # Message passing step
        return self.propagate(edge_index, x=x, edge_weights=weightedEdges, norm=norm)

    def message(self, x_j, edge_weights, norm):
        # x_j: Node features of neighboring nodes
        # weights: Normalized adjacency coefficients
        # Message passing: Aggregate neighbor node features with weights
        print("x_j", x_j.shape[0], 'weights',edge_weights.shape[0], 'norm', norm.shape[0])
        if int(x_j.shape[0]) != int(edge_weights.shape[0]):
            print('Noweight')
            info = {"x_j": x_j.shape[0], "edge_weights": edge_weights.shape[0], "edge_index":self.edge_index, "norm": norm.shape[0]}
            self.noweighted.append(info)
        return x_j

    def update(self, aggr_out, x):
        # aggr_out: Aggregated messages
        # x: Node features (self-information)

        # Combine aggregated messages with node self-information
        return aggr_out
    
def extract_control_nodes_features(node_numbers, batch, x):
    # Determine the total number of graphs in the batch
    batch_size = batch.max().item() + 1
    
    # Prepare an empty tensor for the output
    control_nodes_features = torch.empty((batch_size, x.size(1)), device=x.device)
    
    for i in range(batch_size):
        # Find the index of the control node for the current graph
        control_node_idx = (batch == i).nonzero(as_tuple=True)[0][node_numbers[i]]
        
        # Extract the features of the control node and assign them to the output tensor
        control_nodes_features[i] = x[control_node_idx]
    
    return control_nodes_features    

def fast_extract_control_nodes_features(node_numbers, batch, x):
    # Determine the total number of graphs in the batch
    batch_size = batch.max().item() + 1
    num_nodes_per_graph = torch.bincount(batch, minlength=batch_size)
    num_nodes_before = torch.cumsum(num_nodes_per_graph, 0) - num_nodes_per_graph
    idx = num_nodes_before + node_numbers
    return x[idx]

class GNN(torch.nn.Module):
    "The model"
    def __init__(self, dim_in, dim_h, depth, layersType = 'GIN', residual=True,
                  prelinear=1, init_drop =0.05,final_drop=0.25,readout = "add",
                   task='regression',num_classes=10, weighted=False, double_mlp = False,
                   self_loop=True, mlp_depth=2):
        super(GNN, self).__init__()
        # Create a list to store the GIN layers
        self.gnnLayers = nn.ModuleList()
        self.preLinear = nn.ModuleList()
        self.architecture = layersType
        self.dim_h = dim_h
        self.dim_in = dim_in
        self.alphas = None
        self.init_drop = init_drop
        self.final_drop = final_drop
        self.num_classes = num_classes
        self.dirichlet = []
        self.residual = residual
        self.double_mlp = double_mlp
        self.self_loop = self_loop
        if weighted == "NW":
            weighted = False
        else:
            weighted = True

        #Instantiate initial linear layer(s)
        if prelinear:
            for i in range(prelinear):
                self.preLinear.append(Linear(self.dim_in, self.dim_h))
                self.dim_in = dim_h



        # Instantiate and add GIN layers to the list
        for _ in range(depth):
            if layersType == 'torchGIN':
                #create the mlp with the number of layers specified by mlp_depth
                starter_block = [Linear(self.dim_in if _ == 0 else dim_h, dim_h), BatchNorm1d(dim_h), ReLU()]
                mlp_block = [Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU()]
                mlp = starter_block + mlp_block*(mlp_depth-1)
                gin_layer = GINConv(Sequential(*mlp))
                self.gnnLayers.append(gin_layer)
            if layersType == 'customGIN':
                gin_layer = CustomGINLayer(self.dim_in if _ == 0 else self.dim_h, self.dim_h, weighted=weighted, mlp_depth=mlp_depth)
                self.gnnLayers.append(gin_layer)
            if layersType == 'torchGCN':
                gin_layer = GCNConv(self.dim_in if _ == 0 else dim_h, dim_h)
                self.gnnLayers.append(gin_layer)
            if layersType == 'customGCN':
                gin_layer = CustomGCNLayer(self.dim_in if _ == 0 else self.dim_h, self.dim_h, weighted=weighted)
                self.gnnLayers.append(gin_layer)
            if layersType == 'torchGAT':
                gin_layer = GATConv(self.dim_in if _ == 0 else dim_h, dim_h)
                self.gnnLayers.append(gin_layer)
            if layersType == 'torchResGated':
                gin_layer = ResGatedGraphConv(self.dim_in if _ == 0 else dim_h, dim_h)
                self.gnnLayers.append(gin_layer)
            if layersType == 'torchSGCN':
                gin_layer = SGConv(self.dim_in if _ == 0 else dim_h, dim_h)
                self.gnnLayers.append(gin_layer)

            # if layersType == "ScalarAlphaGIN":
            #     gin_layer = ScalarAlphaGINLayer(dim_in if _ == 0 else dim_h, dim_h, weighted=weighted)
            #     self.gnnLayers.append(gin_layer)

            # if layersType == "VectorAlphaGIN":
            #     gin_layer = VectorAlphaGINLayer(dim_in if _ == 0 else dim_h, dim_h, weighted=weighted)
            #     self.gnnLayers.append(gin_layer)

            # if layersType == "VerticalGIN":
            #     gin_layer = VerticalGINLayer(dim_in if _ == 0 else dim_h, dim_h, weighted=weighted)
            #     self.gnnLayers.append(gin_layer)
            
                
        # if readout == "add":
        #     self.readout = global_add_pool
        # elif readout == "mean":
        #     self.readout = global_mean_pool
        # elif readout == "max":
        #     self.readout = global_max_pool

        
        self.lin1 = Linear(dim_h, dim_h)
        self.lin2 = Linear(dim_h, dim_h)
        if layersType == "torchSGCN":
            self.lin1 = Sequential(Linear(dim_h, 2*dim_h), BatchNorm1d(2*dim_h), ReLU(), Linear(2*dim_h, 2*dim_h), BatchNorm1d(2*dim_h), ReLU(),
                                   Linear(2*dim_h, dim_h), BatchNorm1d(dim_h), ReLU())
            self.initBN = BatchNorm1d(dim_h)
        if task == "classification":
            self.lin3 = Linear(dim_h, self.num_classes)
        else:
            self.lin3a = Linear(dim_h, dim_h)
            self.lin4a = Linear(dim_h, 1)
            self.lin3b = Linear(dim_h, dim_h)
            self.lin4b = Linear(dim_h, 1)
        self.BN = BatchNorm1d(dim_h)
    

    def forward(self, x, edge_index, edge_weights=None, degrees=None ,batch=1, dirichlet = False, dropout=True, ctrl=False):
        #reset the dirichlet energy
        self.dirichlet = []
        # Node embeddings
        for i, layer in enumerate(self.preLinear):
            x = layer(x)
            if self.architecture == "torchSGCN":
                x = self.initBN(x)
            x = nn.functional.relu(x)
            if dropout:
                x = F.dropout(x, p=self.init_drop, training=self.training)
        x = x.to(torch.float32)

        # Message passing
        for layer in self.gnnLayers:
            if self.architecture[0:5] == "torch":
                if self.architecture != "torchResGated":
                    x = x + layer(x, edge_index)
                else:
                    x = layer(x, edge_index)
            else:
                x = layer(x, edge_index, edge_weights,degrees, residual=self.residual, self_loop=self.self_loop)
            x = x.to(torch.float32)
            if dirichlet:
                if batch[0] == 0:
                    self.dirichlet.append(dirichlet_energy(x, edge_index).item())

        if self.architecture == "ScalarAlphaGIN":
            self.alphas = [layer.alpha.item() for layer in self.gnnLayers]
        if self.architecture == "VectorAlphaGIN":
            self.alphas = [torch.sigmoid(layer.alpha) for layer in self.gnnLayers]

        
        # Readout layer
        h = fast_extract_control_nodes_features(ctrl, batch, x)
        
        out = self.lin1(h)
        out = self.BN(out)
        out = nn.functional.relu(out)
        out = self.lin2(out)
        out = self.BN(out)
        out = nn.functional.relu(out)
        if dropout:
            out = F.dropout(out, p=self.final_drop, training=self.training)
        if self.double_mlp == False:
            # Classifier
            out = self.lin3a(out)
            out = self.BN(out)
            out = nn.functional.relu(out)
            out = self.lin4a(out)
        else: 
            out1 = self.lin3a(out)
            out1 = self.BN(out1)
            out1 = nn.functional.relu(out1)
            out1 = self.lin4a(out1)
            out2 = self.lin3b(out)
            out2 = self.BN(out2)
            out2 = nn.functional.relu(out2)
            out2 = self.lin4b(out2)
            return out1, out2, dirichlet_energy(x, edge_index).item()
        
        #return the vector out for regression and the dirichlet energy
        return out, out, dirichlet_energy(x, edge_index).item()
