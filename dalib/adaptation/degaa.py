from typing import Optional
from copy import deepcopy
from pathlib import Path
import torch 
from torch import nn

from common.modules.classifier import Classifier as ClassifierBase

def MLP(channels: list, do_bn = True):
  n = len(channels)
  layers = []
  for i in range(1, n):
    #layers.append(nn.Conv1d(channels[i - 1],
    #channels[i], kernel_size = 1, bias = True))
    layers.append(nn.Linear(channels[i - 1], channels[i]))
    do_bn = False
    if i < n - 1:
      if do_bn:
        layers.append(nn.BatchNorm1d(channels[i]))
      layers.append(nn.ReLU())
  return nn.Sequential(*layers)

def attention(query, key, value):
  dim = query.shape[1]
  scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
  prob = torch.nn.functional.softmax(scores, dim = -1)
  return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class ImageClassifier(ClassifierBase):
  def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
    bottleneck = nn.Sequential(
      nn.AdaptiveAvgPool2d(output_size=(1,1)),
      nn.Flatten(),
      nn.Linear(backbone.out_features, bottleneck_dim),
      nn.BatchNorm1d(bottleneck_dim),
      nn.ReLU()
    )
    super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

class MultiHeadedAttention(nn.Module):
  def __init__(self, num_heads: int, d_model: int):
    super().__init__()
    assert d_model % num_heads == 0
    self.dim = d_model // num_heads 
    self.num_heads = num_heads
    #self.merge = nn.Conv1d(d_model, d_model, kernel_size = 1)
    self.merge = nn.Linear(d_model, d_model)
    self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
  
  def forward(self, query, key, value):
    batch_dim = query.size(0)
    query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1) 
    for l, x in zip(self.proj, (query, key, value))]
    x, _ = attention(query, key, value)
    return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads))


class AttentionalPropagation(nn.Module):
  def __init__(self, feature_dim: int, num_heads: int):
    super().__init__()
    self.attn = MultiHeadedAttention(num_heads, feature_dim)
    self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
    nn.init.constant_(self.mlp[-1].bias, 0.0)

  def forward(self, x, source):
    message = self.attn(x, source, source)
    return self.mlp(torch.cat([x, message], dim = 1))

class AttentionalGNN(nn.Module):
  def __init__(self, feature_dim: int, layer_names: list, num_heads: int):
    super().__init__()
    self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, num_heads) 
    for _ in range(len(layer_names))])
    self.names = layer_names

  def forward(self, S, T):
    for layer, name in zip(self.layers, self.names):
      if name == 'cross':
        I0, I1 = T, S
      else:
        I0, I1 = S, T
      
      delta0, delta1 = layer(S, I0), layer(T, I1)
      S, T = (S + delta0), (T + delta1)
      return S, T

class GAA(nn.Module):
  def __init__(self, input_dim: int, num_classes: int, gnn_layers: int, num_heads: int, **kwargs):
    # self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs
    super().__init__()
    self.input_dim = input_dim
    self.gnn_layers = ['self', 'cross'] * gnn_layers
    self.num_heads = num_heads
    self.num_classes = num_classes
    #self.config = {**self.default_config, **config}
    #default_config = {'dim': 1024, 'GNN_layers': ['self', 'cross'] * 6}
    self.gnn = AttentionalGNN(self.input_dim, self.gnn_layers, self.num_heads)

    #self.final_proj = nn.Conv1d(self.input_dim, self.input_dim, kernel_size = 1, bias = True)
    self.final_proj = nn.Linear(self.input_dim, self.input_dim)
    self.head = nn.Linear(self.input_dim, self.num_classes)

  def forward(self, src, tgt):
    #src = data['source'].double()
    #tgt = data['target'].double()

    src, tgt = self.gnn(src, tgt)
    
    m_src, m_tgt = self.final_proj(src), self.final_proj(tgt)
    y_src, y_tgt = self.head(m_src), self.head(m_tgt)

    return m_src, m_tgt, y_src, y_tgt # To classification head

