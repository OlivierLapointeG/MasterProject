#make necessary imports for a machine learning project on LRGB dataset, aka peptides-funcs dataset
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset, ZINC
import matplotlib.pyplot as plt
from utils import *
import wandb
import random
import pickle
from tqdm import tqdm
import sys
import GNN 
import G2GCN
import TOGNNdev
from torchmetrics.regression import R2Score
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = str(sys.argv[1]) #torchGIN, torchGCN, customGIN, customGCN, torchGAT, torchSGCN, torchResGated
weighted = str(sys.argv[2]) #W for weighted, NW for not weighted
layer = int(sys.argv[3]) #number of layers
if sys.argv[4] == "True":
    residual = True
else:
    residual = False
test = str(sys.argv[5])
if sys.argv[6] == "True":
    self_loop = True
else:
    self_loop = False
dist = str(sys.argv[7])
name = str(sys.argv[8]) #dataset name
nb_seeds = int(sys.argv[9])
model_type = str(sys.argv[10]) # GNN, G2GNN, SRDF, TOGNN

"""Initialise the dataset"""

if name == "LRGB":
    dataset = LRGBDataset(name="Peptides-func",root='/home/students/oliver/MasterProject/LRGBdataset')
if name == "ZINC":
    dataset = ZINC(subset=True,root='/home/students/oliver/MasterProject/ZINCdataset')
if name == "ZINC_SRDF":
    dataset = ZINC(subset=True,root='/home/students/oliver/MasterProject/ZINCdataset')

data = dataset[0]

"""Customize the dataset to create the synthetic task"""

#open the dataset from the pickle file
with open(f"ToyDataset_vec_{dist}_{name}_.pkl", "rb") as f:
    new_dataset_ = pickle.load(f)

new_features = new_dataset_[0]
new_targets = new_dataset_[1]
new_ctrl = new_dataset_[3]

new_targets = torch.tensor(new_targets, dtype=torch.float32)
skip_list = new_dataset_[2]
if name == 'ZINC_SRDF':
    new_edges = new_dataset_[4]

new_dataset = []
correction = 0
for i, data in tqdm(enumerate(dataset)):
    if i in skip_list:
        correction +=1
        continue
    #change the dataset features to custom features
    data.x = new_features[i-correction]
    data.y = new_targets[i-correction][0]
    data.y2 = new_targets[i-correction][1]
    data.ctrl = new_ctrl[i-correction]
    if name == 'ZINC_SRDF':
        data.edge_index = new_edges[i-correction]
    new_dataset.append(data)

"""Add the weights to the dataset if needed"""
if weighted == "W":
    weightedpath = 'ZINClistOfWeightedAdjFull.pkl'
    disconnectedpath = "ZINCdisconnectedFull.pkl"
    new_dataset = apply_weights_on_dataset(weightedpath, disconnectedpath, new_dataset)


#Take the training set, pick a random data point, and draw the graph to check if the features are correct
data = random.choice(new_dataset)
print(data)
adj = edge_index_to_adjacency_matrix(data.edge_index)
adj = adj.cpu().numpy()
#use networkx to draw the graph from the adjacency matrix
G = nx.from_numpy_array(adj)

pos = nx.spring_layout(G)
colors = []
for i in range(len(G.nodes())):
    #if feature 0 i 0, then node black
    if data.x[i][0] == 0:
        colors.append("black")
    #if feature 0 is 1, then node red
    elif data.x[i][0] == 1:
        colors.append("red")
        fortitle1 = data.x[i][1]
    #if feature 0 is -1, then node blue
    elif data.x[i][0] == -1:
        colors.append("blue")
        fortitle2 = data.x[i][1]
    else:
        colors.append("black")
nx.draw(G, pos, node_color=colors, with_labels=True)
print(data.y)
plt.title(f"Control node: {fortitle1} and source node {fortitle2}, target: {data.y}")
#save the graph as png
plt.savefig('Arandomgraph.png',bbox_inches='tight')
