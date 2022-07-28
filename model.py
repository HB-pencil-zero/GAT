from layers import *
import torch 
import torch.nn as nn 
import dgl.data 
import time 
from torch import Tensor

class GAT(nn.Module):
    def __init__(self,n_features:int,n_hidden:int,n_class:int,headerNums:int=3,layerNums:int=3):
        super(GAT,self).__init__()
        self.layerlist=nn.ModuleList()
        self.layerlist.append(layers(n_features,n_hidden,headerNums))
        for _ in range(layerNums-2):
            self.layerlist.append(layers(n_hidden,n_hidden,headerNums))
        self.layerlist.append(layers(n_hidden,n_class,headerNums))

    def forward(self,x:Tensor,adj:Tensor):
        for i in range(len(self.layerlist)):
            x=self.layerlist[i](x,adj)
            x=torch.relu(x)
        return x
    
    def __repr__(self):
        return self.__class__.__name__ + ' ('+ \
              str(self.n_feature)+','+str(self.n_hidden)+','+str(self.n_class)+') ' +" GAT network"


        