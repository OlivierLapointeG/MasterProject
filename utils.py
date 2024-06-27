import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.transforms import BaseTransform
import cvxpy as cp
from tqdm import tqdm
import pickle
import numpy as np
from torch_geometric.loader import DataLoader
import time
import random

def edge_index_to_adjacency_matrix(edge_index, testing=False):
    if edge_index.numel() == 0:
        return torch.zeros((1, 1), dtype=torch.float)
    num_nodes = torch.max(edge_index) +1
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adjacency_matrix[edge_index[0], edge_index[1]] = 1.0
    adjacency_matrix[edge_index[1], edge_index[0]] = 1.0
    return adjacency_matrix

def plot_adjacency_matrix(adjacency_matrix):
    plt.imshow(adjacency_matrix, cmap='binary')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.title('Adjacency Matrix')
    plt.colorbar()
    plt.show()

def plot_graph(data_point, matrix=1, curv_matrix=1, adj=False, curv=False, molecule=True, ax=None):
    atom_type_dict = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

    adjacency_matrix = matrix
    if not adj:
        adjacency_matrix = edge_index_to_adjacency_matrix(data_point.edge_index).numpy()
    if molecule:
        atom_types = data_point.x[:, 5].numpy()  # Consider only the first 6 elements as atom types

        # Create a graph from the adjacency matrix
        graph = nx.from_numpy_array(adjacency_matrix)

        # Get unique atom types
        unique_atom_types = np.unique(atom_types)

        # Define color mapping based on atom types
        num_unique_types = len(unique_atom_types)
        color_map = plt.get_cmap('tab10')  # You can use any desired colormap here
        node_size = 500
        hydrogen_size = 100

        # Assign colors to nodes based on atom types
        node_colors = [color_map(i % num_unique_types) for i in range(num_unique_types)]

        # Draw the graph with node colors
        pos = nx.spring_layout(graph)  # Positions of the nodes
        nx.draw(graph, pos, with_labels=True, node_color=[node_colors[np.where(unique_atom_types == int(atom_type))[0][0]] for atom_type in atom_types],
                node_size=[node_size if int(atom_type) != 1 else hydrogen_size for atom_type in atom_types], edge_color='gray', width=1.0, ax=ax)

        # Create a color legend
        patches = [plt.Line2D([], [], marker='o', markersize=10, color=node_colors[i], linestyle='',
                              label=str(atom_type_dict[unique_atom_types[i]])) for i in range(num_unique_types)]
    else:
        # Create a graph from the adjacency matrix
        graph = nx.from_numpy_array(adjacency_matrix)

        # Draw the graph with node colors
        pos = nx.spring_layout(graph)  # Positions of the nodes
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray', width=1.0, ax=ax)

    if curv:
        curvature_matrix = curv_matrix
        if isinstance(curvature_matrix, torch.Tensor):
            curvature_matrix = curvature_matrix.numpy()
        
        edge_colors = []
        edge_widths = []


        # Define edge colors based on curvature sign
        for (i,j) in graph.edges():
            edge_colors.append("red" if curvature_matrix[i,j]< 0 else 'blue')
            #edge_widths.append(0.3+np.abs(curv_matrix[i,j]))
            edge_widths.append( 0.5 if curvature_matrix[i,j]> 0 else 0.5+np.abs(curvature_matrix[i,j])*4)
        

        edge_colors = np.array(edge_colors)
        edge_widths = np.array(edge_widths)


        # Create a new figure and axis if ax is None
        if ax is None:
            fig, ax = plt.subplots()


        # Draw the edges with modified attributes
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), edge_color=edge_colors,
                               width=edge_widths, ax=ax)

        # Create a colorbar legend for edge curvature
        sm = plt.cm.ScalarMappable(cmap='coolwarm')
        sm.set_array(curvature_matrix)

    # Display the plot
    plt.title('Network')


def upper_triangle_elements(matrix):
    n = len(matrix)
    elements = []

    for i in range(n):
        for j in range(i + 1, n):
            elements.append(matrix[i][j])

    return elements


    
def find_connected_ntuples(data_point, N):

    ### Works up to N=3 because of fork-like structures ###

    
    adjacency_matrix = edge_index_to_adjacency_matrix(data_point.edge_index).numpy()
    num_nodes = adjacency_matrix.shape[0]
    visited = set()
    connected_ntuples = []

    def dfs(node, current_tuple):
        current_tuple.append(node)
        visited.add(node)

        if len(current_tuple) == N:
            connected_ntuples.append(tuple(sorted(current_tuple)))
        else:
            neighbors = np.where(adjacency_matrix[node] > 0)[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs(neighbor, current_tuple)

        current_tuple.pop()
        visited.remove(node)

    for start_node in range(num_nodes):
        current_tuple = []
        dfs(start_node, current_tuple)

    return sorted(set(connected_ntuples))



def find_neighbor_ntuples(data_point, current_tuple):
    """
    Returns a sorted list of neighbor from the tuple, as sorted tuples
    """
    adjacency_matrix = edge_index_to_adjacency_matrix(data_point.edge_index).numpy()
    neighbors = []
    
    # Iterate over the nodes of the current tuple
    for removed_node in current_tuple:
        leftover_tuple = tuple(elem for elem in current_tuple if elem != removed_node)
        if is_tuple_connected(data_point, leftover_tuple):

            # Iterate over all nodes in the graph
            for graph_node in range(adjacency_matrix.shape[0]):
                if graph_node in current_tuple:
                    continue
                # Check if the node is adjacent to any node in the current tuple
                for tuple_node in leftover_tuple:
                    if adjacency_matrix[tuple_node][graph_node] > 0 and tuple_node != graph_node:
                        # Create a new tuple by appending the adjacent node to the current tuple
                        new_tuple = tuple(list(leftover_tuple) + [graph_node])

                        # Exclude the original tuple and avoid permutations
                        if new_tuple != current_tuple:
                            neighbors.append(new_tuple)
    neighbors = sorted([tuple(sorted(i)) for i in neighbors])
    return neighbors



def is_tuple_connected(data_point, nodes_tuple):
    if len(nodes_tuple) <= 1:
        return True
    adjacency_matrix = edge_index_to_adjacency_matrix(data_point.edge_index)
    num_nodes = adjacency_matrix.shape[0]
    visited = set()

    def dfs(node):
        visited.add(node)

        for neighbor in nodes_tuple:
            if adjacency_matrix[node][neighbor] > 0 and neighbor not in visited:
                dfs(neighbor)

    # Perform DFS starting from the first node in the tuple
    dfs(nodes_tuple[0])

    # Check if all nodes in the tuple were visited
    for node in nodes_tuple:
        if node not in visited:
            return False

    return True

def create_new_graph(data_point, n):
    """
    Returns a new adjacency matrix
    """
    # Step 1: Find all connected N-tuples
    all_ntuples = find_connected_ntuples(data_point, n)
    # Step 2: Create an empty adjacency matrix
    num_tuples = len(all_ntuples)
    adjacency_matrix = np.zeros((num_tuples, num_tuples), dtype=int)
    
    # Step 3 and 4: Iterate over each N-tuple and its neighbors
    for i, current_tuple in enumerate(all_ntuples):
        neighbors = find_neighbor_ntuples(data_point, current_tuple)
        
        for neighbor_tuple in neighbors:
            neighbor_index = all_ntuples.index(tuple(sorted(neighbor_tuple)))
            
            # Step 5: Set adjacency matrix entry to 1
            adjacency_matrix[i, neighbor_index] = 1
    
    return adjacency_matrix

def create_new_data(data_point, n):
    """
    Returns a new edge_index tensor
    """
    
    edge_index = [[],[]]
    connected_ntuples = find_connected_ntuples(data_point, n)
    if len(connected_ntuples) < 2:
        return torch.tensor(edge_index, dtype=torch.int64)
    for i, current_tuple in enumerate(connected_ntuples):
        neighbors = sorted(find_neighbor_ntuples(data_point, current_tuple))
        for neighbor_tuple in neighbors:
            neighbor_index = connected_ntuples.index(neighbor_tuple)
            edge_index[0].append(i)
            edge_index[1].append(neighbor_index)

    return torch.tensor(edge_index)



class MyTransform(BaseTransform):
    def __init__(self, n):
        self.n = n

    def __call__(self, data):
        # Modify the data here
        if self.n != 1:
            data.edge_index = create_new_data(data, self.n)
        
        if data.x.shape[0] <= self.n:
            num_nodes = 0
        else:
            num_nodes = torch.max(data.edge_index)+1
        data.x = torch.rand((num_nodes,10))
        data.pos = None
        data.edge_attr = None 
        data.z = None
        return data
    

def adjacency_to_incidence(adjacency_matrix):
    num_vertices = adjacency_matrix.shape[0]
    num_edges = int(np.sum(adjacency_matrix) / 2)  # Assuming an undirected graph.

    incidence_matrix = np.zeros((num_vertices, num_edges), dtype=int)

    edge_index = 0
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if adjacency_matrix[i, j] == 1:
                incidence_matrix[i, edge_index] = 1
                incidence_matrix[j, edge_index] = -1
                edge_index += 1

    return incidence_matrix

class MinimizeRes():
    def __init__(self, data):
        """
        data is an edge_index tensor (or a batch of)
        """
        self.edge_index = data

        

    def minimizeRes(self, testing=False):
        """
        Returns the optimaly weighted adjacency matrix
        """
        #Adjacency matrix and incidence matrix
        adj = edge_index_to_adjacency_matrix(self.edge_index).numpy()
        inc = adjacency_to_incidence(adj)
        num_nodes = adj.shape[0]
        num_edges = inc.shape[1]

        # Necessary objects
        onzeSurN = np.ones((num_nodes, num_nodes))/num_nodes
        weights = np.ones((num_edges,))

        
        
        # Variables as propose in Ghosh et al. 
        Y = cp.Variable((num_nodes,num_nodes), symmetric=True)
        g = cp.Variable(weights.shape, value=weights)


        #create laplacian from incidence matrix and weights
        L = inc @ cp.diag(g) @ inc.T
        Lpp = L+onzeSurN

        #define identity matrix
        I = np.identity(num_nodes)

        #Create block matrix
        X = cp.bmat([[Lpp,I],[I,Y]])

        #Create SDP 
        objective = cp.Minimize(num_nodes*cp.trace(Y))
        constraints = [X>> 0 , g>=1e-6, cp.sum(g)==num_edges]
        prob = cp.Problem(objective, constraints)
        prob.solve()


        #get the solution
        g = g.value
        if testing:
            print(g)

        #construct weighted adjacency matrix
        W = np.zeros((num_nodes, num_nodes))
        counter = 0
        for k in range(2*len(g)):
            i,j = int(self.edge_index[0][k]), int(self.edge_index[1][k])
            if j < i:
                continue
            if testing:
                print(i,j)
                print('weight=', g[counter])
            weight = np.abs(g[counter])
            W[i][j] = weight
            W[j][i] = weight 
            counter += 1
        return W

class weightTransform(BaseTransform):
    def __call__(self, data):
        # Modify the data here
        weightedAdj = MinimizeRes(data.edge_index).minimizeRes()
        weights = []
        for i,j in zip(data.edge_index[0], data.edge_index[1]):
            weights.append(weightedAdj[i,j])
        data.edge_weight = weights 
        return data
    
def adjToNeighbors(adj):
    """
    Returns a list of neighbors for each node
    """
    neighbors = []
    for i in range(adj.shape[0]):
        for neighbor in np.where(adj[i] > 0)[0]:
            neighbors.append(neighbor)
    return torch.tensor(neighbors)

def adjToNeighborsWeight(adj, testing=False):
    '''
    Returns a list of neighbors edge weight for each node
    '''
    #First add self loops to adj
    adj = adj + np.identity(adj.shape[0])
    if testing:
        plt.figure()
        plt.imshow(adj)
    neighbors = []
    for i in range(adj.shape[0]):
        if testing:
            print('row', i, 'cols', np.where(adj[i] > 0)[0])
        for neighbor in np.where(adj[i] > 0)[0]:
            neighbors.append(adj[i,neighbor])
    return torch.tensor(neighbors)


def edgeindexToDegree(edge_index):
    '''
    Returns a list of degrees for each neighbor of the initial node
    '''
    degrees = []
    freq = torch.bincount(edge_index[0])
    for i in freq:
        for _ in range(i+1):
            degrees.append(i.item())

    return torch.tensor(degrees)

def customSigmoid(x, w):
    return torch.heaviside(1/(1+torch.exp(2*(2-w)-4*x)) -0.5, values= torch.tensor(0.0))

def _deltaR_tot(G, incidence, num_nodes, pseudo=False):
    deltaR_tot = np.zeros((incidence.shape[1],))
    for l in range(incidence.shape[1]):
        al = incidence[:,l]
        deltaR_tot[l] = -num_nodes * np.linalg.norm(G @ al)**2
    return deltaR_tot


def hessian(G, incidence, num_nodes):
    term1 = 2 * num_nodes * incidence.T @ G**2 @ incidence
    term2 = incidence.T @ G @ incidence
    return term1 * term2

def _totalRes(g, n , inc, onzeSurN):
    return n * np.trace(np.linalg.inv((inc @ np.diag(g) @ inc.T)+onzeSurN)) - n

def _gradtotalRes(g, n , incidence, onzeSurN):
    deltaR_tot = np.zeros((incidence.shape[1],))
    G = np.linalg.inv((incidence @ np.diag(g) @ incidence.T)+onzeSurN)
    for l in range(incidence.shape[1]):
        al = incidence[:,l]
        deltaR_tot[l] = -n * np.linalg.norm(G @ al)**2
    return deltaR_tot

def interiorPointMethod(edge_index, beta = 1, epsilon=1e-2, stop=1e-1, tau=0.5):
    """
    Returns the optimaly weighted adjacency matrix
    """
    #Adjacency matrix and incidence matrix
    adj = edge_index_to_adjacency_matrix(edge_index).numpy()
    if not isOneConnectedComponent(edge_index):
        return adj, [], [], True
    inc = adjacency_to_incidence(adj)
    n = adj.shape[0]
    m = inc.shape[1]

    #list for plotting
    etas = []
    epsR = []

    # Necessary objects
    onzeSurN = np.ones((n, n))/n
    g = np.ones((m,))/m

    #Loop until convergence, aka eta > epsilon times R_tot
    for _ in tqdm(range(10000)):
        #Laplacian
        L = inc @ np.diag(g) @ inc.T

        #compute (G+onzeSurN)^-1
        G = L+onzeSurN
        G_inv = np.linalg.inv(G)

        # Compute R_tot
        R_tot = n * np.trace(G_inv) - n

        #Compute gradR_tot
        deltaR_tot = _deltaR_tot(G_inv, inc, n)


        #Compute hessian
        hess = hessian(G_inv, inc, n)

        #Compute the gradient of phi
        grad_phi = -1/g

        #Compute the second derivative of phi
        hess_phi = np.diag(1/g**2)

        #eta definition
        eta = - (min(deltaR_tot)+R_tot)
        #1 Set t = βm/η.
        t = beta * m / eta

        #Define f and H as in Ghosh et al.
        H = t*hess + hess_phi
        f= t*deltaR_tot + grad_phi

        #Expand H with a colum and row of ones at the bottom and right
        H = np.vstack((H, np.ones((1, m))))
        H = np.hstack((H, np.ones((m+1, 1))))
        #Make sure the bottom right element is 0
        H[-1,-1] = 0

        #Expand f with a zero at the bottom
        f = np.append(f, 0)

        #2 Compute Newton step δg by solving the linear system Hδg = −f.
        delta_g = np.linalg.solve(H, -f)[:-1]

        #3 Find step length s by backtracking line search
        s = backtrackingLineSearch(lambda x : _totalRes(x, n, inc, onzeSurN), lambda x: _gradtotalRes(x, n , inc , onzeSurN), g, delta_g, tau=tau,stop=stop)
        #4. Set g := g + sδg.
        g = g + s*delta_g

        #check loop condition
        if eta <= epsilon * R_tot:
            break

    #Multiply back with m
    g = g*m

    #Loop on incidence matrix to construct weighted adjacency matrix
    W = np.zeros((n, n))
    for i, col in enumerate(inc.T):
        n = np.where(col == -1)[0][0]
        m = np.where(col == 1)[0][0]
        W[m,n] = g[i]
        W[n,m] = g[i]

    return W, etas, epsR, False

def backtrackingLineSearch(f, grad_f, x, p, c=1, tau=0.9, stop=1e-1):
    """
    Performs a step of backtracking line search

    f : The objective function
    grad_f : The gradient of the objective function
    x: The current point 
    p: The direction of the step (to be found by the algorithm)
    c: Parameter of the algorithm
    tau: Control parameter
    """
    alpha = 1
    m = grad_f(x).T @ p
    #t = - c * m
    counter = 0
    while True:
        if f(x+alpha*p) < f(x) + alpha * c*m:
            break
        alpha = tau*alpha
        counter += 1
        if alpha < stop:
            return alpha
    return alpha

def isOneConnectedComponent(edge_index):
    """
    Returns True if the graph is one connected component, False otherwise
    """
    #create a graph from the edge index
    graph = nx.from_edgelist(edge_index.T.numpy())
    #check if the graph is one connected component
    return nx.is_connected(graph)

def dirichlet_energy(nodes_features, edge_index):
    # Compute the dirichlet energy of a graph
    # nodes_features : torch tensor of shape (num_nodes, num_features)
    # edge_index : torch tensor of shape (2, num_edges)
    # returns a scalar
    num_nodes = nodes_features.shape[0]
    energy = torch.norm(nodes_features[edge_index[0]] - nodes_features[edge_index[1]], dim=1).pow(2).sum()
    energy =  torch.sqrt(energy / num_nodes)
    #normalize the energy by dividing by the norm of the features squared
    energy = energy / torch.norm(nodes_features).pow(2)
    return energy

def fast_extract_control_nodes_features(node_numbers, batch, x):
    # Determine the total number of graphs in the batch
    batch_size = batch.max().item() + 1
    num_nodes_per_graph = torch.bincount(batch, minlength=batch_size)
    num_nodes_before = torch.cumsum(num_nodes_per_graph, 0) - num_nodes_per_graph
    idx = num_nodes_before + node_numbers
    return x[idx]

def apply_weights_on_dataset(weightpath, disconnectedpath, dataset):
    #Initialise weighted adj
    with open(weightpath, 'rb') as file:
        adjList = pickle.load(file)

    with open(disconnectedpath, 'rb') as file:
        disconnected = pickle.load(file)

    new_dataset = []
    counter = 0
    for i, data in tqdm(enumerate(dataset)):
        if i in disconnected:
            counter += 1
            continue
        adj = adjList[i-counter]
        #check that every element in adj are positive
        for _ in range(len(adj)):
            for j in range(len(adj)):
                if adj[_][j] < 0:
                    adj[_][j] = 1e-6
        data.weightedEdges = adjToNeighborsWeight(adj)
        data.degrees = edgeindexToDegree(data.edge_index)
        data.edge_attr
        new_dataset.append(data)
    dataset = new_dataset
    return dataset

def ConstructAdjFromControl(edge_index, edge_control):
    """
    Returns the adjacency matrix from the edge index and the edge control
    """
    # Convert edge_index to numpy array
    edge_index_np = edge_index.cpu()
    edge_index_np = edge_index_np.numpy()

    # Get the maximum node index
    max_node_index = np.max(edge_index_np) + 1

    # Initialize the adjacency matrix
    adj = np.zeros((max_node_index, max_node_index))

    # Iterate over the edges
    for i in range(len(edge_index_np[0])):
        adj[edge_index_np[0][i], edge_index_np[1][i]] = edge_control[i]

    return adj

def compute_diff(wModel, original_loader, epsilon_loader, device, batch_size):
            diff_smooth = 0
            diff_squash = 0

            for data, data2 in zip(original_loader, epsilon_loader):
                data = data.to(device)
                data2 = data2.to(device)
                out_smooth, out_squash, _ = wModel(data.x.to(torch.float32), data.edge_index, batch=data.batch, dirichlet=False, ctrl=data.ctrl)
                out_smooth, out_squash = out_smooth.view(-1), out_squash.view(-1)
                out_smooth2, out_squash2, _ = wModel(data2.x.to(torch.float32), data2.edge_index, batch=data2.batch, dirichlet=False, ctrl=data2.ctrl)
                out_smooth2, out_squash2 = out_smooth2.view(-1), out_squash2.view(-1)

                # compute the difference between the outputs
                diff_smooth += torch.sum(torch.abs(out_smooth - out_smooth2)) / (len(epsilon_loader)*batch_size)
                diff_squash += torch.sum(torch.abs(out_squash - out_squash2)) / (len(epsilon_loader)*batch_size)

            return diff_smooth, diff_squash

def create_epsilon_loaders(new_dataset, epsilon):
    epsilon_datatest = [data.clone() for data in new_dataset[int(len(new_dataset)*0.9):]]
    epsilon_datatrain = [data.clone() for data in new_dataset[:int(len(new_dataset)*0.9)]]
    for data in epsilon_datatest:
        for nodes in data.x:
            if nodes[0] == -1:
                nodes[1] += epsilon
    for data in epsilon_datatrain:
        for nodes in data.x:
            if nodes[0] == -1:
                nodes[1] += epsilon
    epsilon_trainloader = DataLoader(epsilon_datatrain, batch_size=64, shuffle=False)
    epsilon_testloader = DataLoader(epsilon_datatest, batch_size=64, shuffle=False)

    return epsilon_trainloader, epsilon_testloader

def plot_graph_TOGNN(wModel, train_dataset, test_dataset, device, EPOCHS):
    for test in range(10):
    #take a random graph from dataset
        data_ = random.choice(train_dataset).clone()
        data_ = data_.to(device)
        #run the model on the graph
        out_smooth, out_squash, _ = wModel(data_.x.to(torch.float32), data_.edge_index, batch= data_.batch, dirichlet=False, ctrl=data_.ctrl,
                                            edge_weights=data_.weightedEdges.to(torch.float32), edge_attr=data_.edge_attr.to(torch.float32), visualize_weights=True)
        #plot the graph
        weights = wModel.newgraph
        #transfer the weights to an adjacency matrix using constructAdjFromControl
        adjs = []
        for i in range(len(weights)):
            adj = ConstructAdjFromControl(data_.edge_index,weights[i])
            adjs.append(adj)
        
        graphs = []
        #use nx to construct networkx directed graphs from the adjacency matrices
        for adj in adjs:
            G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
            graphs.append(G)
        pos = nx.spring_layout(G)
        colors = []
        for i in range(len(G.nodes())):
            #if feature 0 i 0, then node black
            if data_.x[i][0] == 0:
                colors.append("black")
            #if feature 0 is 1, then node red
            elif data_.x[i][0] == 1:
                colors.append("red")
            #if feature 0 is -1, then node blue
            elif data_.x[i][0] == -1:
                colors.append("blue")
            else:
                colors.append("black")
        #plot the directed graphs
        fig, ax = plt.subplots(1, len(graphs), figsize=(24, 8))
        for i in range(len(graphs)):
            nx.draw(graphs[i], pos=pos, ax=ax[i], with_labels=True, node_color=colors, edge_color='gray', node_size=300, width=2, arrowsize=15)
        #save the figure in folder as png
        #create a timestamp for the figure
    
        timestamp = time.time()
        plt.savefig(f"graphs/testDigraph_{timestamp}_{test}_{EPOCHS}epochs.png")

def reconstruct_graph(random_data, gradients, depth, model, epoch, dist):
    edge_index = random_data.edge_index
    node_features = random_data.x
    adj = edge_index_to_adjacency_matrix(edge_index).numpy()
    # Construct the graph and write gradients on nodes
    G = nx.from_numpy_array(adj)
    pos = nx.spring_layout(G)

    colors = []
    for i in range(len(G.nodes())):
        #if feature 0 i 0, then node grey
        if node_features[i][0] == 0:
            colors.append("grey")
        #if feature 0 is 1, then node red
        elif node_features[i][0] == 1:
            colors.append("red")
        #if feature 0 is -1, then node blue
        elif node_features[i][0] == -1:
            colors.append("blue")
        else:
            colors.append("black")

    #plot the graph
    plt.figure()
    nx.draw(G, pos=pos, with_labels=False, node_color=colors, edge_color='gray', node_size=300, width=2)
    #add the gradients to the nodes

    for i in range(len(G.nodes())):
        plt.text(pos[i][0], pos[i][1], f'{gradients[i]:.2f}', fontsize=10, color='black')

    #save the figure in folder as png
    plt.savefig(f"graphs/reconstructedGraph_{model}_{depth}layer_D{dist}_{epoch}epoch.png")
