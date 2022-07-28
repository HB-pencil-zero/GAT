import numbers
import torch.nn as nn
import torch 
from torch import ParameterDict, Tensor
from torch.nn.parameter import Parameter


class layers(nn.Module):
    def __init__(self,in_features:int,out_features:int,headersNum:int,bias:bool=True)->Tensor:
        super(layers,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.headersNum=headersNum
        self.originList=nn.ParameterList()
        self.alphaList=nn.ParameterList()
        self.weightlist=nn.ParameterList()
        self.atom_features=out_features // headersNum
        self.last_features=self.atom_features+ out_features % headersNum 
        for _ in range(headersNum-1):
            self.originList.append(nn.Parameter(torch.DoubleTensor(self.atom_features).cuda().requires_grad_()))
            self.alphaList.append(nn.Parameter(torch.DoubleTensor(self.atom_features).cuda().requires_grad_()))
            self.weightlist.append(nn.Parameter(torch.DoubleTensor(self.in_features,self.atom_features).cuda().requires_grad_()))
        self.originList.append(nn.Parameter(torch.DoubleTensor(self.last_features).cuda().requires_grad_()))
        self.alphaList.append(nn.Parameter(torch.DoubleTensor(self.last_features).cuda().requires_grad_()))
        self.weightlist.append(nn.Parameter(torch.DoubleTensor(self.in_features,self.last_features).cuda().requires_grad_()))

        self.bias=Parameter(data=torch.DoubleTensor(out_features))
        
        if(not bias):
            self.register_parameter("bias",None)
        self.initialise()
    
    def initialise(self):
        for weight in self.weightlist:
            weight.retain_grad()
            stdv=weight.size(1)
            weight.data.uniform_(-stdv,stdv)
        for origin_alpha,alpha in zip(self.originList,self.alphaList):
            alpha.retain_grad()
            origin_alpha.retain_grad()
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
            coef1=mx@alpha_origin
            coef2=mx@alpha
            tensorList=[]
            for i in range(x.size(0)):
                tensor=self.clacAttn(mx,i,adj,weight,coef1,coef2)
                tensorList.append(tensor)

            result_list.append(torch.vstack(tensorList))

        return torch.hstack(result_list).cuda()

    #multihead 计算 注意力机制
    def clacAttn(self,mx:Tensor,loc:int,adj:Tensor,weight:Tensor,coef1:Tensor,coef2:Tensor):
        """计算 aplha(x_i,x_j)"""
        x=coef1[loc]+coef2
        adj=adj[loc,:]
        
        x=torch.exp(x)
        x=x*adj
        sum=x.sum(0)
        x/=sum

        return x@mx

