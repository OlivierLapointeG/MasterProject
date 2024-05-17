import torch
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from utils import fast_extract_control_nodes_features, dirichlet_energy

class EdgeControl(nn.Module):
    def __init__(self, conv, p=2., conv_type='GCN', activation=nn.ReLU()):
        super(EdgeControl, self).__init__()
        self.conv = conv
        self.p = p
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

class TO_GNN(nn.Module):
    def __init__(self, dim_in, dim_h, depth, layersType = 'GIN', residual=True,
                  prelinear=1, init_drop =0.05,final_drop=0.25,readout = "add",
                   task='regression',num_classes=10, weighted=False, double_mlp = False,
                   self_loop=True, mlp_depth=2, use_gg_conv = True, conv_type='GCN'):
        super(TO_GNN, self).__init__()
        self.gnnLayers = nn.ModuleList()
        self.g2Layers = nn.ModuleList()
        self.preLinear = nn.ModuleList()
        self.architecture = layersType
        self.dim_h = dim_h
        self.dim_in = dim_in
        self.conv_type = conv_type
        self.nlayers = depth
        self.tau = []
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

        # Instantiate and add G2 layers to the list
        for _ in range(depth):
            if conv_type == 'GCN':
                self.conv = GCNConv(dim_h, dim_h)
                if use_gg_conv == True:
                    self.conv_gg = GCNConv(dim_h, dim_h)
            elif conv_type == 'GAT':
                self.conv = GATConv(dim_h,dim_h,heads=4,concat=True)
                if use_gg_conv == True:
                    self.conv_gg = GATConv(dim_h,dim_h,heads=4,concat=True)
            else:
                print('specified graph conv not implemented')

            if use_gg_conv == True:
                self.G2 = G2(self.conv_gg,2,conv_type,activation=nn.ReLU())
            else:
                self.G2 = G2(self.conv,2,conv_type,activation=nn.ReLU())
            self.gnnLayers.append(self.conv)
            self.g2Layers.append(self.G2)

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




        

    def forward(self, x, edge_index, edge_weights=None, degrees=None ,batch=1, dirichlet = False, dropout=True, ctrl=False):
        #reset the dirichlet energy
        self.dirichlet = []
        self.tau = []

        for i, layer in enumerate(self.preLinear):
            x = layer(x)
            x = nn.functional.relu(x)
            if dropout:
                x = F.dropout(x, p=self.init_drop, training=self.training)
        x = x.to(torch.float32)


        # Message passing
        for i,layer in enumerate(self.gnnLayers):
            x_ = torch.relu(layer(x, edge_index))
            if dirichlet:
                if batch[0] == 0:
                    self.dirichlet.append(dirichlet_energy(x, edge_index).item())
            tau = self.g2Layers[i](x, edge_index)
            self.tau.append(tau.detach().cpu().numpy())
            x = (1 - tau) * x + tau * x_

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
