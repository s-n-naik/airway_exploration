#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# !apt-get install -y xvfb
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

from typing import Optional

from torch_scatter import scatter_add

from torch_geometric.utils import softmax

class CustomGlobalAttention(torch_geometric.nn.GlobalAttention):
    def __init__(self,gate_nn, nn = None, return_attn=True):
        '''
        
        '''
        super(CustomGlobalAttention, self).__init__(gate_nn, nn)        # super(CustomDataset, self).__init__(root, transform, pre_transform)
        self.return_attn = return_attn
        self.gate_nn = gate_nn
        self.nn = nn
        
        
    def forward(self, x, batch = None, size=None):
        r"""
        # All taken from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/glob/attention.html
        Args:
            x (Tensor): The input node features.
            batch (LongTensor, optional): A vector that maps each node to its
                respective graph identifier. (default: :obj:`None`)
            size (int, optional): The number of graphs in the batch.
                (default: :obj:`None`)
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.int64)

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = int(batch.max()) + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)
        if self.return_attn:
            return out, gate
        else:
            return out


# In[50]:


class GAT(torch.nn.Module):
    def __init__(self, 
                 input_dim_node=13,
                 edge_dim=2, 
                 hidden_dim1=8,
                 hidden_dim2 = 8,
                 heads1=8,
                 heads2 = 1,
                 num_classes=1,
                 dropout=0.6,
                 agg='max'
                    
                
                
                ):
        super(GAT, self).__init__()
        self.hid1 = hidden_dim1
        self.hid2 = hidden_dim2
        self.head1 = heads1
        self.head2 = heads2
        self.node_dim = input_dim_node
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.num_classes = num_classes
        assert agg.lower() in ['max', 'mean','attn', 'none']
        self.agg = agg
        
        
        self.conv1 = GATConv(self.node_dim, self.hid1, edge_dim=self.edge_dim, heads=self.head1, dropout=dropout)
        self.conv2 = GATConv(self.hid1*self.head1, self.hid2, edge_dim=self.edge_dim,concat=False,
                             heads=self.head2, dropout=dropout)
        
        
        self.lin1 = nn.Linear(self.hid2*self.head2, 1) # computes attention weights
#         self.lin2 =  nn.Linear(self.hid2*self.head2, 1) # transfomrs features F --> V before you multiply + add w attention weights
        self.att = CustomGlobalAttention(gate_nn=self.lin1, return_attn=True)
        
        # if global mean pool
        self.classifier = nn.Sequential(
                                        nn.Linear(self.hid2*self.head2, self.num_classes),
                                        nn.Sigmoid()
                                    )

    def forward(self, data):
        weights = None
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
                
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
#         print("conv1", x.shape)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
#         print("conv2", x.shape)
        per_node = F.log_softmax(x, dim=1)
#         print("per node", per_node.shape)
        
        # 2. Readout layer
        if self.agg == 'attn':
            x, weights = self.att(x, batch)  # [batch_size, hidden_channels]
        elif self.agg == 'mean':
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.agg == 'max':
            x = global_max_pool(x,batch) # [batch_size, hidden_channels]
        elif self.agg == 'none':
            # [batch_size*num_nodes_per_graph, hidden_channels]
            x = x
#         print("readout",self.agg, x.shape)
            
        
        # Linear classifier to get per graph label (if agg is not none) or per node label
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
#         print("post linear", x.shape)
        
        if weights is None:
            # either returns attn weights if using attn aggregation or uses the softmax of the features of each node 
            weights = per_node
        
        return x, weights
    




