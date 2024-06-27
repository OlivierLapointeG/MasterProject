import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from utils import fast_extract_control_nodes_features, dirichlet_energy

class EdgeControl(nn.Module):
    def __init__(self, conv, activation=nn.Sigmoid()):
        super(EdgeControl, self).__init__()
        self.conv = conv
        self.activation = activation

    def forward(self, X, edge_index, edge_attr):
        n_edges = edge_index.size(1)
        edge_feature = edge_attr.view(n_edges, 1)
        edge_features = torch.cat([X[edge_index[0]], X[edge_index[1]], edge_feature], dim=-1)
        skip = edge_features.clone()
        features = self.activation(self.conv(edge_features))

        return features
    



class TOGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, mlp_depth):
        super(TOGNNLayer, self).__init__(aggr="add")  # 'add' aggregation method
        # Learnable parameters: weights for message passing
        self.in_channels = in_channels
        mlp = [nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()]
        for i in range(mlp_depth-1):
            mlp.append(nn.Linear(out_channels, out_channels))
            mlp.append(nn.BatchNorm1d(out_channels))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)
        self.EdgeControl = EdgeControl(nn.Linear(1+2*in_channels, 2))
        self.control = None 

    def forward(self, x, edge_index, weightedEdges,residual=True, self_loop=True):
        # x: Node features, shape (N, in_channels)
        # edge_index: Edge index, shape (2, E) for undirected graph

        # # Add self-loops for the graph (optional but common in GIN)
        # if self_loop:
        #     edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        skip = x
        # Node self-information (initial embedding)


        out = self.mlp(self.propagate(edge_index, x=x, edge_weights=weightedEdges))

        # Message passing step
        return out + residual*skip

    def message(self, x_j, edge_weights):
        # x_j: Node features of neighboring nodes
        # Message passing: Aggregate neighbor node features with weights
        return edge_weights.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out: Aggregated messages
        # Combine aggregated messages with node self-information
        return aggr_out

def create_weightedEdges(logits):
    """
    Create the edge weights from the probabilities, where the weight is 0 of the maximum probability
    belongs to the first class and 1 if it belongs to the second class
    """
    edge_weights = logits[:,0].view(-1,1)
    return edge_weights


class GNN(torch.nn.Module):
    "The model"
    def __init__(self, dim_in, dim_h, depth, layersType = 'GIN', residual=True,
                  prelinear=1, init_drop =0.05,final_drop=0.25,readout = "add",
                   task='regression',num_classes=10, weighted=False, double_mlp = False,
                   self_loop=True, mlp_depth=2):
        super(GNN, self).__init__()
        # Create a list to store the GIN layers
        self.gnnLayers = nn.ModuleList()
        self.actionlayers = nn.ModuleList()
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



        # Instantiate and add TOGNN layers to the list
        for _ in range(depth):
            action = EdgeControl(nn.Linear(1+2*dim_h, 2))
            conv = TOGNNLayer(dim_h, dim_h, mlp_depth)
            self.gnnLayers.append(conv)
            self.actionlayers.append(action)

           
        
        self.lin1 = Linear(dim_h, dim_h)
        self.lin2 = Linear(dim_h, dim_h)
        
        if task == "classification":
            self.lin3 = Linear(dim_h, self.num_classes)
        else:
            self.lin3a = Linear(dim_h, dim_h)
            self.lin4a = Linear(dim_h, 1)
            self.lin3b = Linear(dim_h, dim_h)
            self.lin4b = Linear(dim_h, 1)
        self.BN = BatchNorm1d(dim_h)
    

    def forward(self, x, edge_index, edge_weights=None, degrees=None ,batch=1, dirichlet = False, dropout=True, ctrl=False, edge_attr=None, visualize_weights=False):
        #reset the dirichlet energy
        self.dirichlet = []
        self.newgraph = []

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
        for conv_layer, action_layer in zip(self.gnnLayers, self.actionlayers):
            logits = action_layer(x, edge_index, edge_attr)
            #create the edge weights from the logits
            probs = F.gumbel_softmax(logits, tau=1, hard=True)
            edge_weights = create_weightedEdges(probs)
            if visualize_weights:
                self.newgraph.append(edge_weights.squeeze().detach().cpu().numpy())
            x = conv_layer(x, edge_index, edge_weights, residual=self.residual, self_loop=self.self_loop)
            x = x.to(torch.float32)
            if dirichlet:
                if batch[0] == 0:
                    self.dirichlet.append(dirichlet_energy(x, edge_index).item())

        if self.architecture == "ScalarAlphaGIN":
            self.alphas = [layer.alpha.item() for layer in self.gnnLayers]
        if self.architecture == "VectorAlphaGIN":
            self.alphas = [torch.sigmoid(layer.alpha) for layer in self.gnnLayers]

        if visualize_weights:
            return None, None, None
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


