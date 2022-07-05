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



# make training function
def train_model(model, 
                optimizer,
                train_loader, 
                device,
                num_epochs, 
                count_epochs=0,
                verbose=True, 
                scheduler=None, 
                accum_iter=1, 
                class_weight=None):
    
    plotting_dict_train = {"loss":[], "accuracy": []}
    noisy_label=False
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        denominator_loss = 0
        denominator_acc=0
        for batch_idx, data in list(enumerate(train_loader)):
            data = data.to(device)
            batch_size = data.num_graphs
            batch_vector = data.batch
            denominator_loss+=batch_size
            # reset gradients
            if batch_idx % accum_iter ==0:
                optimizer.zero_grad()

            # conduct a forward pass
            out, weight = model(data)
#             print("out", out.shape, "weight", weight.shape)
            y_per_graph = data.y.float()
            
              # Noisy vs per graph labelling depending on model type
            if (out.squeeze()).shape != data.y.shape:
                noisy_label = True
                y = torch.take(data.y.float(), batch_vector) # repeat label shape
#                 print('Noisy labelling', y.shape)
                # denominator for accuracy etc. is y.shape from here (to average out per node accuracy)
            else:
                y = data.y.float()
#                 print('Graph labelling', y.shape)
                # denominatro here is y.shape (per graph to avg out graph accuracy)
            
            denominator_acc +=y.shape[0]
            # calculate loss and metrics
            pred = out > 0.5
            pred = pred.long()
#             print(pred.unique())
            correct += pred.eq(y.view_as(pred)).sum().item()
            
            
            if class_weight is not None:
                weights = torch.take(class_weight.to(device), y.long()) # class_weight has weight for 0 in posnt 0 and 1 in 1
#                 print("weights", weights, y, sep='\n')
                loss = F.binary_cross_entropy(out.squeeze(),y.float(), weight=weights.float())
            else:
                loss = F.binary_cross_entropy(out.squeeze(),y)
            train_loss += loss.item()

            # backward pass, normalising for gradient accumulation
            loss = loss / accum_iter
            loss.backward()
            
            # step
            if (batch_idx % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()

                
            if verbose:
                print('Epoch: {}, Batch: {}, Loss: {:.2f}'.format(epoch+count_epochs, batch_idx, loss.item()))
        # calculate loss and error for epoch
        train_loss /= denominator_loss # loss is already mean over the nodes so just divide by num graphs in batch
        accuracy = correct / denominator_acc # if doing noisy labelling need to do length of vector
        plotting_dict_train["loss"].append(train_loss)
        plotting_dict_train["accuracy"].append(accuracy)
        
        # step at the end of each epoch
        if scheduler is not None:
            scheduler.step()
            
            if scheduler is not None:
                last_lr = scheduler.get_last_lr()[0]
            
        print('Epoch: {}, Train Loss: {:.5f}, Train Accuracy: {:.5f}'.format(epoch+count_epochs, train_loss, accuracy))
        
    return plotting_dict_train


# In[52]:



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


# id_test = binary_label_df_test.iloc[np.argmax((binary_label_df_test.binaryLL_1==1))].idno
# print("test index", id_test)
# index_test = [index for index, (idno, relabel) in test_loader.dataset.node_map.items() if idno==id_test][0]
# _vis_graph_example(test_loader,model, index_test, pilot_df_w_labels)


# In[53]:


def test_model(model, test_loader, device, count_epochs = 0, threshold=0.5):
    print(f"Getting F1 score, Accuracy and Confusion matrix on the Test Set at epoch {count_epochs}")
    predictions = []
    y = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in list(enumerate(test_loader)):
            data = data.to(device)
            label = data.y
            batch_vector = data.batch
            batch_size = data.num_graphs
            outputs, _ = model(data)
            pred = outputs > threshold
            pred = pred.long()
             # Noisy vs per graph labelling depending on model type
            if (pred.squeeze()).shape != label.shape:
                noisy_label = True
                label_f = torch.take(label.float(), batch_vector) # repeat label shape
#                 print('Noisy labelling', y.shape)
                # denominator for accuracy etc. is y.shape from here (to average out per node accuracy)
            else:
                label_f = label.float()
            
            
            
            predictions.append(pred)
            y.append(label_f)

    predictions_tensor = torch.cat(predictions).cpu().numpy().squeeze()
    labels_tensor = torch.cat(y).cpu().numpy().squeeze()
    print("preds, labels", predictions_tensor.shape, labels_tensor.shape)
    cm = confusion_matrix(labels_tensor, predictions_tensor)
    f1 = f1_score(labels_tensor, predictions_tensor, average='binary')
    accuracy = accuracy_score(labels_tensor, predictions_tensor)
    print("F1 score for positive class (1): {:.2f}".format(f1))
    print("Accuracy: {:.2f}".format(accuracy))
    data = {'y_Actual':  labels_tensor.copy(),
        'y_Predicted': predictions_tensor.copy()
        }
    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    
    confusion_matrix_pd = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    plt.figure()
    sn.heatmap(confusion_matrix_pd, annot=True)
    plt.title("Confusion Matrix")
    plt.savefig(f"confusion_matrix_{count_epochs}.jpg")
    plt.show()
    
    return cm, f1, accuracy


# In[58]:

if __name__ == '__main__':
    run_id = 3
    try:
        os.makedirs(f'/home/sneha/airway_exploration/run_{run_id}/')
    except:
        pass

    verbose= True
    device = set_device_and_seed()
    print("Device", device)

    batch_size = 256
    num_epochs = 200
    count_epochs = 0
    test_every = 2


    class_prop = my_data_train.class_proportions
    class_weight_tensor = torch.tensor([1/class_prop[0], 1/class_prop[1]], dtype=float)
    mag = class_weight_tensor.sum()
    class_weight_tensor = class_weight_tensor/mag
    print("class weights for weighted loss", class_weight_tensor)
    train_loader = DataLoader(my_data_train, batch_size=batch_size, shuffle=True)
    test_loader =  DataLoader(my_data_test, batch_size=batch_size, shuffle=False)



    model = GAT(agg='attn',dropout=0.2, hidden_dim1=64, hidden_dim2=32).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    model.train()
    test_accuracy = []
    test_f1 = []
    test_cm = []
    test_epoch = []
    train_accuracy = []
    train_loss = []
    while count_epochs <= (num_epochs-test_every):
        print(f"Training epochs: {count_epochs} to {count_epochs+test_every} ")
        plotting_dict_train = train_model(model, optimizer ,train_loader, device,num_epochs=test_every, count_epochs=count_epochs, verbose=verbose, class_weight =class_weight_tensor)
        count_epochs += test_every
        train_accuracy.extend(plotting_dict_train['accuracy'])
        train_loss.extend(plotting_dict_train['loss'])



        # test model
        cm, f1, accuracy =  test_model(model, test_loader, device, count_epochs, threshold=0.5)
        test_epoch.append(count_epochs)
        test_cm.append(cm)
        test_f1.append(f1)
        test_accuracy.append(accuracy)

        print("Visualise training progress")
        f, (ax1, ax2) = plt.subplots(1,2)
        ax1.plot(np.arange(0,len(train_accuracy)), train_accuracy, label='train_accuracy')
        ax2.plot(np.arange(0,len(train_loss)), train_loss, label='train_loss')
        ax1.set_xlabel('epoch')
        ax2.set_xlabel('epoch')
        ax1.set_ylabel('accuracy')
        ax2.set_ylabel('loss')
        plt.savefig(f'/home/sneha/airway_exploration/run_{run_id}/training_epoch_{count_epochs}')
        plt.show()

        f, (ax1, ax2) = plt.subplots(1,2)
        ax1.plot(test_epoch, test_accuracy, label='test_accuracy')
        ax2.plot(test_epoch, test_f1, label='test_f1')
        ax1.set_xlabel('epoch')
        ax2.set_xlabel('epoch')
        ax1.set_ylabel('accuracy')
        ax2.set_ylabel('f1')
        plt.savefig(f'/home/sneha/airway_exploration/run_{run_id}/test_epoch_{count_epochs}')
        plt.show()

    #   vis_ids are one anomalous and one normal all in train data not test
        for i in range(0,2):
            id_test = binary_label_df_train.iloc[np.argmax((binary_label_df_train.binaryLL_1==i))].idno
            index_test = [index for index, (idno, relabel) in train_loader.dataset.node_map.items() if idno==id_test][0]
            _vis_graph_example(train_loader,model, index_test, pilot_df, visualise_g = False, save_path = f'/home/sneha/airway_exploration/run_{run_id}/training_vis_epoch_{count_epochs}_{i}')


# In[ ]:





# In[ ]:




