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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

"""parse the arguments"""

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
model_type = str(sys.argv[10]) # GNN, G2GNN, SRDF

"""Initialise the dataset"""

if name == "LRGB":
    dataset = LRGBDataset(name="Peptides-func",root='/home/olivier/GraphCurvatureProject/LRGBdataset')
if name == "ZINC":
    dataset = ZINC(subset=True,root='/home/olivier/GraphCurvatureProject/ZINCdataset')
if name == "ZINC_SRDF":
    dataset = ZINC(subset=True,root='/home/olivier/GraphCurvatureProject/ZINCdataset')


"""Customize the dataset to create the synthetic task"""

#open the dataset from the pickle file
with open(f"ToyDataset_vec_{dist}_{name}.pkl", "rb") as f:
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


"""Split the dataset into train and test loaders"""

train_dataset = new_dataset[:int(len(new_dataset)*0.9)]
test_dataset  = new_dataset[int(len(new_dataset)*0.9):]

#create a copy of the test_dataset where each feature has +epsilon
epsilon_dataset = new_dataset[int(len(new_dataset)*0.9):]
for data in epsilon_dataset:
    for nodes in data.x:
        if nodes[0] == -1:
            nodes[1] += 0.01

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
epsilon_loader = DataLoader(epsilon_dataset, batch_size=64, shuffle=False)

#delete dataset to free up memory
del dataset
torch.cuda.empty_cache()



"""Initialise hyperparameters"""

LEARNING_RATE = 0.0005
EPOCHS = 120
layers = np.arange(1, layer+1)
dim_in = 2
if test == "both":
    double_mlp = True
else:
    double_mlp = False

width = 32 #width of the model
if model == "torchGCN":
    width = 48
mlp_depth = 2 #depth of the mlp for GIN

"""Set up the seeds for group testing"""
seeds = [random.randint(1, 1000) for i in range(nb_seeds)]

for seed in seeds:
    #initialise random seed
    torch.manual_seed(seed)

    #Initialise metrics
    num_labels = 10
    train_R2score_smooth = R2Score().to(device)
    train_R2score_squash = R2Score().to(device)
    test_R2score_smooth = R2Score().to(device)
    test_R2score_squash = R2Score().to(device)




    # Initialise model
    if model_type == "GNN":
        modell = GNN.GNN(dim_in, width, layer, layersType=model, task='regression', 
                        num_classes=num_labels, weighted=weighted, residual=residual, 
                        self_loop=self_loop, double_mlp=double_mlp, mlp_depth=mlp_depth)
    if model_type == "G2GNN":
        modell = G2GCN.G2_GNN(dim_in, width, layer, layersType=model, task='regression', 
                        num_classes=num_labels, weighted=weighted, residual=residual, 
                        self_loop=self_loop, double_mlp=double_mlp, mlp_depth=mlp_depth, use_gg_conv = True, conv_type='GCN')
        model = "G2GCN"
    if model_type == "SRDF":
        modell = GNN.GNN(dim_in, width, layer, layersType=model, task='regression', 
                        num_classes=num_labels, weighted=weighted, residual=residual, 
                        self_loop=self_loop, double_mlp=double_mlp, mlp_depth=mlp_depth)
    if model_type == "TOGNN":
        modell = GNN.GNN(dim_in, width, layer, layersType=model, task='regression', 
                        num_classes=num_labels, weighted=weighted, residual=residual, 
                        self_loop=self_loop, double_mlp=double_mlp, mlp_depth=mlp_depth)

        model = 'SRDF'
    wModel = modell.to(device)
    # Initialise optimizer, and classification criterion  for multi-label classification
    criterion = torch.nn.MSELoss().to(device)

    optimizer = torch.optim.AdamW(wModel.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)


    """ Initialise WandB """
    wandb.init(
        # set the wandb project where this run will be logged
        project="SyntheticDataset10",
        
        # set the name of the run
        name=f"V10-{model}-{layer}layers-testing-{test}-D{dist}",

        # tags
        tags=[test],
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": model,
            "dataset": "ToyDataset",
            "epochs": EPOCHS,
            "Parameters": sum(p.numel() for p in wModel.parameters()),
            "layers": layer,
            "residual": residual,
            "testing":test,
            "dist": dist,
            "self_loop": self_loop,
        }
    )
    print(f"Training {model} with {layer} layers on ToyDataset...")


    """Train the model"""
    for epoch in tqdm(range(EPOCHS)):
        # Reset Metrics
        train_squash_loss = 0
        test_squash_loss = 0
        train_full_loss = 0
        train_smooth_loss = 0
        test_smooth_loss = 0
        test_full_loss = 0
        train_diric = 0
        test_diric = 0
        dirichlet = []
        train_outputs = []
        test_outputs = []
        train_R2score_smooth.reset()
        train_R2score_squash.reset()
        test_R2score_smooth.reset()
        test_R2score_squash.reset()
        if EPOCHS-epoch < 10:
            dropout = False
        else:
            dropout = True

        wModel.train()
        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()

            out_smooth, out_squash, diric = wModel(data.x.to(torch.float32), data.edge_index, batch=data.batch, dropout=dropout, ctrl=data.ctrl, edge_weights=data.weightedEdges.to(torch.float32))
            out_smooth, out_squash = out_smooth.view(-1), out_squash.view(-1)
            

            if test == "both":
                loss_smooth = criterion(out_smooth, data.y2) 
                loss_squash = criterion(out_squash, data.y)
                train_full_loss += ((loss_smooth + loss_squash) / len(train_loader))
                train_squash_loss += (loss_squash / len(train_loader)).detach()
                train_smooth_loss += (loss_smooth / len(train_loader)).detach()
                train_R2score_smooth(out_smooth.detach(), data.y2)
                train_R2score_squash(out_squash.detach(), data.y)
                loss = loss_smooth + loss_squash

            elif test == "smooth":
                loss_smooth = criterion(out_smooth, data.y2)
                train_full_loss += (loss_smooth / len(train_loader)).detach()
                train_R2score_smooth(out_smooth.detach(), data.y2)
                loss = loss_smooth

            elif test == "squash":
                loss_squash = criterion(out_squash, data.y)
                train_full_loss += (loss_squash / len(train_loader)).detach()
                train_R2score_squash(out_squash.detach(), data.y)
                loss = loss_squash


            train_diric += diric/len(train_loader)

            # Run the backward pass
            loss.backward()
            optimizer.step()
        
        
        scheduler.step(metrics=train_full_loss)



        """ Log all the training metrics and empty cache """
        if test == "both":
            wandb.log({"train_loss": train_full_loss, "train_smooth_loss": train_smooth_loss, "train_squash_loss": train_squash_loss, "train_dirichlet": train_diric, "train_R2_smooth": train_R2score_smooth.compute(), "train_R2_squash": train_R2score_squash.compute()})
        elif test == "smooth":
            wandb.log({"train_smooth_loss": train_full_loss, "train_R2_smooth": train_R2score_smooth.compute()})
        elif test == "squash":
            wandb.log({"train_squash_loss": train_full_loss, "train_R2_squash": train_R2score_squash.compute()})

        wandb.log({"train_dirichlet": train_diric})
        if model_type == "G2GNN":
            if epoch%20==0:
                for i in range(len(wModel.tau)):
                    wandb.log({f"tau_{i}": wandb.Histogram(wModel.tau[i])})

        wModel.eval()
        if epoch == 10:
            track_dirichlet = True
        else:
            track_dirichlet = False
        if name == "ZINC_SRDF":
            track_dirichlet = False


        for data in test_loader:
            data = data.to(device)
            out_smooth, out_squash, diric = wModel(data.x.to(torch.float32), data.edge_index, batch= data.batch, dirichlet=track_dirichlet, ctrl=data.ctrl, dropout=False, edge_weights=data.weightedEdges.to(torch.float32))
            out_smooth, out_squash = out_smooth.view(-1), out_squash.view(-1)

            
            if test == "both":
                loss_smooth = criterion(out_smooth, data.y2) 
                test_smooth_loss += (loss_smooth / len(test_loader)).detach()
                loss_squash = criterion(out_squash, data.y)
                test_squash_loss += (loss_squash / len(test_loader)).detach()
                test_full_loss += ((loss_smooth + loss_squash) / len(test_loader)).detach()
                test_R2score_smooth(out_smooth.detach(), data.y2)
                test_R2score_squash(out_squash.detach(), data.y)
            elif test == "smooth":
                loss_smooth = criterion(out_smooth, data.y2)
                test_R2score_smooth(out_smooth.detach(), data.y2)
                test_full_loss += (loss_smooth / len(test_loader)).detach()
            elif test == "squash":
                loss_squash = criterion(out_squash, data.y)
                test_R2score_squash(out_squash.detach(), data.y)
                test_full_loss += (loss_squash / len(test_loader)).detach()

            test_diric += diric/len(test_loader)


            if track_dirichlet:
                if data.batch[0] == 0:
                    dirichlet.append(wModel.dirichlet)


        #compute the difference between the outputs of the model on the test set and the epsilon set
        diff_smooth = 0
        diff_squash = 0
        #run the model on the epsilon dataset and test set simultaneously
        for data, data2 in zip(test_loader, epsilon_loader):
            data = data.to(device)
            data2 = data2.to(device)
            out_smooth, out_squash, _ = wModel(data.x.to(torch.float32), data.edge_index, batch= data.batch, dirichlet=False, ctrl=data.ctrl,edge_weights=data.weightedEdges.to(torch.float32))
            out_smooth, out_squash = out_smooth.view(-1), out_squash.view(-1)
            out_smooth2, out_squash2, _ = wModel(data2.x.to(torch.float32), data2.edge_index, batch= data2.batch, dirichlet=False, ctrl=data2.ctrl,edge_weights=data.weightedEdges.to(torch.float32))
            out_smooth2, out_squash2 = out_smooth2.view(-1), out_squash2.view(-1)


            #compute the difference between the outputs
            diff_smooth += torch.sum(torch.abs(out_smooth - out_smooth2))/len(test_loader)
            diff_squash += torch.sum(torch.abs(out_squash - out_squash2))/len(test_loader)

        #log the differences
        wandb.log({"diff_smooth": diff_smooth, "diff_squash": diff_squash})




        if test=="both":
            wandb.log({"test_loss": test_full_loss,
                        "test_R2_smooth": test_R2score_smooth.compute(),
                        "test_R2_squash": test_R2score_squash.compute(),
                        "test_smooth_loss": test_smooth_loss,
                        "test_squash_loss": test_squash_loss, "test_dirichlet": test_diric})
        elif test == "smooth":
            wandb.log({"test_smooth_loss": test_full_loss, "test_R2_smooth": test_R2score_smooth.compute()})
        elif test == "squash":
            wandb.log({"test_squash_loss": test_full_loss, "test_R2_squash": test_R2score_squash.compute()})
        torch.cuda.empty_cache()
        
       
        #log the curve of the dirichlet energy through the layers
        if track_dirichlet:
            print("dirichlet = ", dirichlet[-1])
            fig = plt.figure()
            plt.plot(layers, dirichlet[-1])
            plt.xlabel("Layer")
            plt.ylabel("Dirichlet energy")
            wandb.log({"dirichlet": wandb.Image(fig)})
            plt.close(fig)
    print(f"Epoch {epoch} | Train LOSS: {train_full_loss} | Test LOSS: {test_full_loss}")
    #log an histogram of the targets
    # wandb.log({"train_target_histogram": wandb.Histogram(np.array(new_targets))})

    wandb.finish()

    print("Training finished!")


