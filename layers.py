import numbers
import torch.nn as nn
import torch 
from torch import Tensor
from torch.nn.parameter import Parameter


class layers(nn.Module):
    def __init__(self,in_features:int,out_features:int,headersNum:int,bias:bool=True)->Tensor:
        super(layers,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.headersNum=headersNum
        self.originList=[]
        self.alphaList=[]
        self.weightlist=[]
        self.atom_features=out_features // headersNum
        self.last_features=self.atom_features+ out_features % headersNum 
        for _ in range(headersNum-1):
            self.originList.append(Parameter(torch.FloatTensor(self.atom_features)))
            self.alphaList.append(Parameter(torch.FloatTensor(self.atom_features)))
            self.weightlist.append(Parameter(torch.FloatTensor(self.in_features,self.atom_features)))
        self.originList.append(Parameter(torch.FloatTensor(self.last_features)))
        self.alphaList.append(Parameter(torch.FloatTensor(self.last_features)))
        self.weightlist.append(Parameter(torch.FloatTensor(self.in_features,self.last_features)))

        self.bias=Parameter(data=torch.FloatTensor(out_features))
        
        if(not bias):
            self.register_parameter("bias",None)
        self.initialise()
    
    def initialise(self):
        for weight in self.weightlist:
            stdv=weight.size(1)
            weight.data.uniform_(-stdv,stdv)
        for origin_alpha,alpha in zip(self.originList,self.alphaList):
            stdv=alpha.size(0)
            origin_alpha.data.uniform_(-stdv,stdv)
            alpha.data.uniform_(-stdv,stdv)
        if(self.bias is not None):
            self.bias.data.uniform_(-stdv,stdv)
    
    
    def forward(self,x:Tensor,adj:Tensor):
        result_list=[]
        for weight,alpha_origin,alpha in zip (self.weightlist,self.originList,self.alphaList):
            #先做映射
            mx=x@weight

            tensorList=[]
            for i in range(x.size(0)):
                tensor=self.clacAttn(mx,i,adj,weight,alpha_origin,alpha)
                tensorList.append(tensor)

            result_list.append(torch.vstack(tensorList))

        return torch.hstack(result_list)

    #multihead 计算 注意力机制
    def clacAttn(self,mx:Tensor,loc:int,adj:Tensor,weight:Tensor,alpha_origin:Tensor,alpha:Tensor):
        """计算 aplha(x_i,x_j)"""

        x=mx
        vector=x[loc,:]
        
        
        coef1=vector@alpha_origin
        x=x@alpha+coef1
        adj=adj[loc,:]
        
        x=torch.exp(x)
        x[torch.where(adj==0)]=0
        sum=x.sum(0)
        x/=sum

        return x@mx


a=torch.Tensor([1,1,0])
b=torch.Tensor([1,1,0])
c=torch.Tensor([0,0,1])
mx=torch.vstack([a,b,c])
adj=mx


