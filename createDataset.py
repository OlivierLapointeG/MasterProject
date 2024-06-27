#make necessary imports for a machine learning project on LRGB dataset, aka peptides-funcs dataset
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset, QM9, ZINC
import matplotlib.pyplot as plt
import networkx as nx
from utils import *
import time
import wandb
import random
import pickle
import networkx as nx
from torchmetrics.regression import MeanSquaredError
from tqdm import tqdm
import sys
from cudaSDRF import *
from torchmetrics.regression import R2Score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
test = str(sys.argv[1])
dist = int(sys.argv[2])
name = str(sys.argv[3])
if name == "ZINC_mul":
    test = "vec_mul"
mode = str(sys.argv[4]) #visualise, save or histogram 
if mode == "visualise":
    VISUALIZE = True
    HISTOGRAM = False
elif mode == "save":
    VISUALIZE = False
    HISTOGRAM = False
elif mode == "histogram":
    VISUALIZE = False
    HISTOGRAM = True
if name == "QM9":
    dataset = QM9(root='/home/students/oliver/MasterProject/QM9dataset')
elif name == "LRGB":
    dataset = LRGBDataset(name="Peptides-func",root='/home/students/oliver/MasterProject/LRGBdataset')
elif name == "ZINC":
    dataset = ZINC(subset=True,root='/home/students/oliver/MasterProject/ZINCdataset')
elif name =='ZINC_SRDF':
    dataset = ZINC(subset=True,root='/home/students/oliver/MasterProjectZINCdataset')
elif name == "ZINC_mul":
    dataset = ZINC(subset=True,root='/home/students/oliver/MasterProject/ZINCdataset')





#make the dataset iterable
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# data_iterator = iter(dataloader)

#change the dataset features to custom features
def changeFeatures(data, dist=10, test="both"):
    edge = data.edge_index
    adj = edge_index_to_adjacency_matrix(edge).numpy()
    graph = nx.from_numpy_array(adj)

    # Find all pairs of nodes that are at a distance of 'dist'
    shuffled_nodes = list(graph.nodes())
    random.shuffle(shuffled_nodes)

    pairs = []
    found_pair = False
    while found_pair == False:
        for i in range(len(shuffled_nodes)):
            for j in range(len(shuffled_nodes)):
                try:
                    sp = nx.shortest_path_length(graph, shuffled_nodes[i], shuffled_nodes[j])
                except:
                    continue
                if sp == dist:
                    pairs.append((shuffled_nodes[i], shuffled_nodes[j]))
                    found_pair = True
                    break
            if found_pair:
                break
        dist -= 1
        if dist < 1:
            break
    
    #if no pair is found, return false
    if not found_pair:
        return False, False, None

    
    #initialize all features to a 2d tensor. Zero for the first dimension and uniform random for the second dimension
    data.x = torch.zeros((len(shuffled_nodes), 2))
    data.x[:, 1] = torch.FloatTensor(len(shuffled_nodes)).uniform_(-2, 2)

    data.x[pairs[0][0], 0] = 1
    data.x[pairs[0][1], 0] = -1

    #find the neighbors of the first node in the pair using the adjacency matrix
    neighbors = []
    for i in range(len(adj)):
        if adj[pairs[0][0]][i] == 1:
            neighbors.append(i)
    
    #Compute the dirichlet energy of the 1 hop neighborhood of node 1
    energy = 0
    for j in range(len(neighbors)):
        energy += (data.x[pairs[0][0], 1] - data.x[neighbors[j], 1])**2
    
    #change the targets to the addition of the two nodes
    if test=="both":
        data.y = [torch.pow(torch.tensor([data.x[pairs[0][0],1] + data.x[pairs[0][1],1]]), 2), energy]
    elif test=="smooth":
        data.y = energy 
    elif test=="squash":
        data.y = torch.pow(torch.tensor([data.x[pairs[0][0],1] + data.x[pairs[0][1],1]]), 2)
    elif test=="vec":
        data.y = [torch.pow(torch.tensor([data.x[pairs[0][0],1] + data.x[pairs[0][1],1]]), 2), energy]
    elif test=="vec_mul":
        data.y = [torch.tensor([data.x[pairs[0][0],1] * data.x[pairs[0][1],1]]), energy]
    return data.x, data.y, pairs[0][0]



new_dataset_features_list = []
new_dataset_target_list = []
new_dataset_control_nb = []
new_dataset_edge_index = []
skip_list = []
#For testing purposes, we will only use the first 1000 graphs

for i in tqdm(range(len(dataset))):
# for i in tqdm(range(len(dataset)//1000)):
    data = dataset[i]
    nb_loops = data.edge_index.shape[1] // 20
    if name == 'ZINC_SRDF':
        new_edge_index = sdrf(data, loops=nb_loops, remove_edges=True, removal_bound=1.7, tau=20, is_undirected=True).edge_index
    x,y, nb = changeFeatures(data, dist, test=test)
    if x is not False:
        new_dataset_features_list.append(x)
        new_dataset_target_list.append(y)
        new_dataset_control_nb.append(nb)
        if name == 'ZINC_SRDF':
            new_dataset_edge_index.append(new_edge_index)
    else:
        skip_list.append(i)

print(len(new_dataset_features_list), len(new_dataset_target_list), len(skip_list), len(new_dataset_control_nb))
new_dataset = [new_dataset_features_list, new_dataset_target_list, skip_list, new_dataset_control_nb, new_dataset_edge_index]



if HISTOGRAM:
    # Make a histogram for all the new targets[0] and new targets[1]
    targets_smooth = []
    targets_squash = []
    targets_vec = []
    for i in range(len(new_dataset[1])):
        targets_smooth.append(new_dataset[1][i][1].item())
        targets_squash.append(new_dataset[1][i][0].item())

        #plot the histograms in 2 different figures
        plt.figure(1)
        plt.hist(targets_smooth, bins=50)
        plt.title("Histogram of the smooth targets")
        #save the figure
        plt.savefig(f"Histogram_smooth_{test}_{dist}_{name}.png",bbox_inches='tight')

        plt.figure(2)
        plt.hist(targets_squash, bins=50)
        plt.title("Histogram of the squash targets")
        plt.savefig(f"Histogram_squash_{test}_{dist}_{name}.png",bbox_inches='tight')



if VISUALIZE:
    #plot a random graph with the control nodes in red and blue
    random_index = np.random.randint(len(new_dataset))
    edge_index = dataset[random_index].edge_index
    adj = edge_index_to_adjacency_matrix(edge_index).numpy()
    graph = nx.from_numpy_array(adj)
    pos = nx.spring_layout(graph)
    #create the colors for the nodes
    colors = []
    for i in range(len(graph.nodes())):
        #if feature 0 i 0, then node black
        if new_dataset[0][random_index][i][0] == 0:
            colors.append("black")
        #if feature 0 is 1, then node red
        elif new_dataset[0][random_index][i][0] == 1:
            colors.append("red")
        #if feature 0 is -1, then node blue
        elif new_dataset[0][random_index][i][0] == -1:
            colors.append("blue")
        else:
            colors.append("black")

    nx.draw(graph, pos, with_labels=True, node_color=colors)
    #On the graph, write each node's feature 1
    for i in range(len(graph.nodes())):
        plt.text(pos[i][0]+0.05, pos[i][1]+0.05, int(new_dataset[0][random_index][i][0].item()))
        plt.text(pos[i][0]+0.04, pos[i][1], round(new_dataset[0][random_index][i][1].item(), 2))
    plt.title(f"Exemple of graph with  LR target equal to {round(new_dataset[1][random_index][0].item(),2)} and SR target {round(new_dataset[1][random_index][1].item(),2)}")
    plt.savefig(f"A_random_graph_d{dist}.png",bbox_inches='tight')

else:
    #save the dataset
    with open(f"ToyDataset_{test}_{dist}_{name}.pkl", "wb") as f:
        pickle.dump(new_dataset, f)


#open the dataset from the pickle file
# with open(f"ToyDataset.pkl", "rb") as f:
#     new_dataset = pickle.load(f)