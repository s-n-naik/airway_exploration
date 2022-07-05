#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import torch
import scipy
import scipy.sparse
from collections import Counter
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
# https://www.youtube.com/watch?v=QLIkOtKS4os --> creating custom dataset in pytorch geometric
from torch.utils.data import Dataset, random_split
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GlobalAttention
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_networkx, from_networkx
from sklearn.model_selection import StratifiedKFold
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import pickle
import seaborn as sn
import random
import os

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score




def set_device_and_seed(GPU=True, seed=0, gpu_name = "cuda:0"):
    torch.cuda.is_available()
    if GPU:
        device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device("cpu")
    print(f'Using {device}')

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

        
    set_seed(seed)
    
    return device

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Setting torch, cuda, numpy and random seeds to {seed}")


def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

    
def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()
    
def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == True)


def _vis_graph_example(dataloader,model, index, pilot_df_w_labels, visualise_g = False, save_path = None):
    data = dataloader.dataset[index]
    print(data)
    label = data.y
    g = to_networkx(data)
    # get relabelmap
    idno, relabel = dataloader.dataset.node_map[index]
    # map to original df
    vis_graph  =pilot_df_w_labels.loc[pilot_df_w_labels.idno==idno]
    print(f"This graph is for id: {idno}")
    display(vis_graph.head())
    # get the relabelling to match the pytorch graph 
    vis_graph['start_node'] = vis_graph.startbpid.apply(lambda x:relabel[x])
    vis_graph['end_node'] = vis_graph.endbpid.apply(lambda x:relabel[x])
#     display(vis_graph[['startbpid', 'endbpid']+node_features + ['parent_loc_x_norm','parent_loc_y_norm']])
#     print(data.x)
    print("Getting model preds per node (node model needs to be a per node one)")
    model.eval()
    x, weight = model(data.to(device))
    x = x.cpu().detach().numpy() # take off cuda
    weight = weight.cpu().detach().numpy() # take off cuda
    
#     print("x", x.shape[0], 'weight', weight.shape[0], 'label', label.shape[0])
    
    if model.agg == 'none':
#         print('useing node preds')
        # per node rather than per graph
        preds = x.copy().squeeze()
    else:
        # per graph output - use per node (weights )for color
#         print("using weights")
        preds = weight.copy().squeeze()
#     print("preds shape", preds.shape)
    
    
    # drawing graph in networkx & matplotlib
    cmap = mpl.colormaps['spring'].reversed()
    vmin, vmax = 0,1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cmap_nodes = [cmap(norm(preds[node])) for node in g.nodes()]
    edge_labels = {edge:vis_graph.loc[vis_graph.end_node == edge[1]]['anatomicalname'].item() for edge in g.edges()}
    pos = nx.planar_layout(g, scale=1, center=(0,0), dim=2)
    
    
    if visualise_g:
        f, ax = plt.subplots(figsize=(10,10))
        nx.draw(g,pos=pos, with_labels=False,node_color=cmap_nodes, ax=ax)
        nx.draw_networkx_edge_labels(g, pos,
                                  edge_labels,
                                     font_color='k',
                                     font_size='10',
                                  label_pos=0.5,

                                    )
        
        cbar = plt.colorbar(sm)

        plt.title(f'TRAINING GRAPH: {idno}, ANOMALY LABEL: {label.item()}')
        plt.show()


    
    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    for i in range(len(vis_graph)):
        row = vis_graph.iloc[i]
        ax.plot([row.parent_loc_x,row.x], [row.parent_loc_y,row.y], [row.parent_loc_z, row.z], linestyle='-',linewidth=1, color= cmap(norm(preds[row.end_node])), label=row.end_node)
        ax.scatter(row.x, row.y, row.z, marker='o',color= cmap(norm(preds[row.end_node])))
    ax.grid(False)
    ax.set_facecolor(color=(1,1,1))
    cbar = plt.colorbar(sm)
    plt.title(f'TRAINING GRAPH: {idno}, ANOMALY LABEL: {label.item()}')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()




