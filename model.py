from layers import *
import torch 
import torch.nn as nn 
import dgl.data 
import time 

class GAT(nn.Module):
    def __init__(self):
        super(GAT,self).__init__()
        