# coding=utf-8
import torch
import torch.nn as nn
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD 
from torch_sparse import coalesce 

_norm_layer_factory = {
    'batchnorm': nn.BatchNorm1d,
    'layernorm': nn.LayerNorm,
}

_act_layer_factory = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'sigmoid': nn.Sigmoid,
}


def create_spectral_features(
        pos_edge_index: torch.LongTensor,  
        neg_edge_index: torch.LongTensor, 
        node_num: int,                     
        dim: int                           
) -> torch.FloatTensor:
    edge_index = torch.cat(
        [pos_edge_index, neg_edge_index], dim=1)
    N = node_num
    edge_index = edge_index.to(torch.device('cpu'))  

    pos_val = torch.full(
        (pos_edge_index.size(1),), 2, dtype=torch.float)
    neg_val = torch.full(
        (neg_edge_index.size(1),), 0, dtype=torch.float)
    val = torch.cat([pos_val, neg_val], dim=0)

    row, col = edge_index
    edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
    val = torch.cat([val, val], dim=0)

    edge_index, val = coalesce(edge_index, val, N, N)
    val = val - 1 

    edge_index = edge_index.detach().numpy()
    val = val.detach().numpy()
    A = sp.coo_matrix((val, edge_index), shape=(N, N))

    svd = TruncatedSVD(n_components=dim, n_iter=128)
    svd.fit(A)
    x = svd.components_.T  
    return torch.from_numpy(x).to(torch.float)


class MLP(nn.Module): 
    def __init__(self, dim_in=256, dim_hidden=32, dim_pred=1, num_layer=3, norm_layer=None, act_layer=None, p_drop=0.5,
                 sigmoid=False, tanh=False):
        super(MLP, self).__init__()
        '''
        The basic structure is refered from 
        '''
        assert num_layer >= 2, 'The number of layers shoud be larger or equal to 2.'
        # 初始化层组件
        if norm_layer in _norm_layer_factory.keys():
            self.norm_layer = _norm_layer_factory[norm_layer]
        if act_layer in _act_layer_factory.keys():
            self.act_layer = _act_layer_factory[act_layer]
        if p_drop > 0:
            self.dropout = nn.Dropout

        fc = []

        fc.append(nn.Linear(dim_in, dim_hidden))
        if norm_layer:
            fc.append(self.norm_layer(dim_hidden))
        if act_layer:
            fc.append(self.act_layer(inplace=True))
        if p_drop > 0:
            fc.append(self.dropout(p_drop))
            
        for _ in range(num_layer - 2):
            fc.append(nn.Linear(dim_hidden, dim_hidden))
            if norm_layer:
                fc.append(self.norm_layer(dim_hidden))
            if act_layer:
                fc.append(self.act_layer(inplace=True))
            if p_drop > 0:
                fc.append(self.dropout(p_drop))
 
        fc.append(nn.Linear(dim_hidden, dim_pred))
      
        if sigmoid:
            fc.append(nn.Sigmoid())  
        if tanh:
            fc.append(nn.Tanh())    
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        out = self.fc(x)
        return out                   
