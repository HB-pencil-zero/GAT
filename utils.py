import dgl.data
import torch 
import torch.nn as nn
import numpy as np
from torch import Tensor
import scipy.sparse as sp

def ecode_onehot(labels:Tensor):
    class_set=set(labels)
    class_dict={j: np.identity(len(class_set))[i,:] for i,j in enumerate(labels) }
    one_hot_code= np.array(list(map(class_dict.get,labels)),dtype=np.int32)
    return torch.from_numpy(one_hot_code)


def load_data():
    graph=dgl.data.CoraGraphDataset()[0]
    feature=graph.ndata['feat']
    labels=graph.ndata['label']
    test_mask=graph.ndata['test_mask']
    train_mask=graph.ndata['train_mask']
    val_mask=graph.ndata["val_mask"]
    #graph_size=np.size(len(graph.nodes().tolist()),len(graph.nodes().tolist()))
    adj=torch.sparse_coo_tensor(np.vstack([graph.edges()[0].tolist(),graph.edges()[1].tolist()]),
                                np.ones(len(graph.edges()[0].tolist())),
                        size= (len(graph.nodes().tolist()),len(graph.nodes().tolist()))).to_dense()
    adj=adj+adj.T.multiply(adj.T>adj)-adj.multiply(adj.T>adj)

    adj=normlise(adj+np.eye(adj.size(0)))

    return adj,feature,labels,train_mask,val_mask,test_mask


def normlise(mx:Tensor):
    row_sum=mx.sum(1)
    row_inv=torch.pow(row_sum,-1)
    row_inv[torch.where(row_sum==0)]=0
    return torch.diag(row_inv)@mx


def accuracy(outputs:torch.Tensor,labels):
    pred=outputs.max(1)[1].type_as(labels)
    correct=pred.eq(labels).double()
    correct=correct.sum()
    return correct/len(labels)

def  sparse_mx_to_torch_sparse_tensor(mx:sp.coo_matrix):
    '''把sparse的matrix转换为sparse tensor
    '''
    mx=mx.tocoo()#.astype(np.float32)
    indice=torch.from_numpy(
        np.vstack(mx.row,mx.col).astype(np.int32)
    )
    values=torch.from_numpy(mx.data)
    shape=torch.Size(mx.shape)
    return torch.sparse_coo_tensor(indice, values, shape).to_dense()