#make necessary imports for a machine learning project on LRGB dataset, aka peptides-funcs dataset
import numpy 
import torch
from torch_geometric.data import Data, Dataset, batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset, ZINC
import matplotlib.pyplot as plt
import networkx as nx
from utils import *
import time
import random
import pickle

#download LRGB dataset and create a dataset object containing only peptides func
# dataset = LRGBDataset(name="Peptides-func", root='/home/olivier/GraphCurvatureProject/LRGBdataset')
dataset = ZINC(subset=True,root='/home/students/oliver/MasterProject/ZINCdataset')




if __name__ == "__main__":
    #test the dataset object
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # data_iterator = iter(dataloader)
    # data = next(data_iterator)
    edge_index = dataset[1].edge_index
    
    EPSILON = 5e-2
    STOP = 1e-1
    BETA = 1
    TAU = 0.4
    vizualize = False
    if vizualize:
        #minimuze the eff resistance using function from utils.py
        a = time.time()
        wAdj, etas, epsR, sing = interiorPointMethod(edge_index, beta = BETA, epsilon=EPSILON, stop=STOP, tau=TAU)
        b = time.time()
        print(b-a)
        c = time.time()
        wAdj2, _1, _2, sing= interiorPointMethod(edge_index, beta = BETA, epsilon=EPSILON*0.1, stop=STOP, tau=TAU)
        d = time.time()
        print(d-c)

        c = time.time()
        wAdj3, _1, _2, sing = interiorPointMethod(edge_index, beta = BETA, epsilon=EPSILON*0.01, stop=STOP, tau=TAU)
        d = time.time()
        print(d-c)

        plt.plot(etas)
        plt.plot(epsR)
        plt.show()

        #create figure
        fig, ax = plt.subplots(3,1, figsize=(10,10))
        #fix a title for the whole figure
        fig.suptitle('Optimal weights for the minimal effective resistance', fontsize=22)
        adj = edge_index_to_adjacency_matrix(edge_index).numpy()
        W = wAdj



        G = nx.from_numpy_array(adj)
        pos = nx.spring_layout(G)
        # nx.draw(G, pos=pos, with_labels=True, node_size=10,  width=3, ax=ax[0])
        
        G_opti = nx.from_numpy_array(W)
        edge_weights = nx.get_edge_attributes(G_opti, 'weight')
        min_weight = min(edge_weights.values())
        max_weight = max(edge_weights.values())
        print(min_weight, max_weight)
        cmap = plt.cm.get_cmap('viridis')
        norm = plt.Normalize(min_weight, max_weight)
        opti_edge_colors = [cmap(norm(weight)) for weight in edge_weights.values()]
        im = nx.draw(G_opti, pos=pos, with_labels=False, node_size=8, width=3, edge_color=opti_edge_colors, ax=ax[0])
        #create a colorbar that show the real values of the weights and not the normalized ones, for this axis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax[0])
        ax[0].set_title('Optimal weights', fontsize=18)

        #plot the second graph
        G_opti = nx.from_numpy_array(wAdj2)
        edge_weights = nx.get_edge_attributes(G_opti, 'weight')
        min_weight = min(edge_weights.values())
        max_weight = max(edge_weights.values())
        cmap = plt.cm.get_cmap('viridis')
        norm = plt.Normalize(min_weight, max_weight)
        opti_edge_colors = [cmap(norm(weight)) for weight in edge_weights.values()]
        nx.draw(G_opti, pos=pos, with_labels=False, node_size=8, width=3, edge_color=opti_edge_colors, ax=ax[1])
        #create a colorbar that show the real values of the weights and not the normalized ones
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax[1])
        ax[1].set_title('Optimal weights, epsilon = 0.1', fontsize=18)

        #plot the third graph
        G_opti = nx.from_numpy_array(wAdj3)
        edge_weights = nx.get_edge_attributes(G_opti, 'weight')
        min_weight = min(edge_weights.values())
        max_weight = max(edge_weights.values())
        cmap = plt.cm.get_cmap('viridis')
        norm = plt.Normalize(min_weight, max_weight)
        opti_edge_colors = [cmap(norm(weight)) for weight in edge_weights.values()]
        nx.draw(G_opti, pos=pos, with_labels=False, node_size=8, width=3, edge_color=opti_edge_colors, ax=ax[2])
        #create a colorbar that show the real values of the weights and not the normalized ones
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax[2])
        ax[2].set_title('Optimal weights, epsilon = 0.01', fontsize=18)

        
        plt.show()
    

    adjList = []
    disconnected = []

    counter = 0
    for data in tqdm(dataset):
        edge_index = data.edge_index

        #check that the graph is one connected component
        if not isOneConnectedComponent(edge_index):
            disconnected.append(counter)
            counter += 1
            continue

        wAdj, etas, epsR, singular = interiorPointMethod(edge_index, beta = BETA, epsilon=EPSILON, stop=STOP, tau=TAU)
        adjList.append(wAdj)
        counter += 1
        if counter % 2000 == 0:
            with open(f'ZINClistOfWeightedAdjFirst{counter}.pkl', 'wb') as file:
                pickle.dump(adjList, file)
            with open(f'ZINCdisconnectedFirst{counter}.pkl', 'wb') as file:
                pickle.dump(disconnected, file)
    with open(f'ZINClistOfWeightedAdjFull.pkl', 'wb') as file:
        pickle.dump(adjList, file)
    with open(f'ZINCdisconnectedFull.pkl', 'wb') as file:
        pickle.dump(disconnected, file)



