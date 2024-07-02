import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import sys

def create_community_graph(n, p, q):
    """
    Create a community graph with n nodes and two communities with connection probabilities p and q.
    """
    G = nx.random_partition_graph([n//2, n-(n//2)],p_in=p, p_out=q)
    #remove self loops and lone nodes
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def nx_to_edge_index(graph: nx.Graph):
    # Get the edges from the NetworkX graph
    edges = list(graph.edges)
    
    # Create a list of tuples representing the edges in both directions
    bidirectional_edges = [(u, v) for u, v in edges] + [(v, u) for u, v in edges]
    
    # Sort the bidirectional edges by the source node first, then by the target node
    bidirectional_edges = sorted(bidirectional_edges)
    
    # Separate the tuples into source and target lists
    source, target = zip(*bidirectional_edges)
    
    # Create the edge index tensor with shape (2, 2E)
    edge_index = torch.tensor([source, target], dtype=torch.long)
    
    return edge_index

def create_community_dataset(n, p, nb_graphs):
    """
    Create a dataset of community graphs with n nodes and two communities with connection probabilities p and q.
    """
    slices = nb_graphs // 10
    p_out = np.logspace(-3.5, -1.5, 10)
    dataset = []
    for i in tqdm(range(nb_graphs)):
        q = p_out[i//slices]
        G = create_community_graph(n, p=p, q=q)
        #restart if the graph is not one connected component
        while not nx.is_connected(G):
            print('Restarting...')
            G = create_community_graph(n, p=p, q=q)
        #remove self loops and lone nodes
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        #create a Data object with edge_index only
        edge_index = nx_to_edge_index(G)
        data = Data(edge_index=edge_index, prob_out=q)
        #use the partition as node features
        partition = G.graph["partition"]
        x = torch.zeros(n, 1)
        for i in partition[0]:
            x[i] = 1
        for i in partition[1]:
            x[i] = 2
        data.x = x
        dataset.append(data)
    return dataset

nb_nodes = int(sys.argv[1])
nb_graphs = int(sys.argv[2])
dataset = create_community_dataset(nb_nodes, 0.3, nb_graphs)

torch.save(dataset, f'datasets/community_dataset_{nb_nodes}.pt')