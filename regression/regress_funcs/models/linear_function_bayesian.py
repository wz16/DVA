import torch
import torch.nn as nn
import importlib
from torch.nn.parameter import Parameter
from typing import Optional
from  .variational import GaussianVariational, ScaleMixture, BayesianModule, base_bayesian_model
import torch.nn.functional as F
from .linear_layer_bayesian import linear_layer
class Net(base_bayesian_model):

  def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
    super(Net, self).__init__()

    self.linear_layer = linear_layer(input_dim=input_dim,output_dim=output_dim)
    self.model_mode = "bayesian"

  def forward(self, x):
    x = self.linear_layer(x)

    return x

  def sample(self):
    self.linear_layer.sample(self.model_mode)
    self.kl_divergence = self.linear_layer.kl_divergence

    return 
  def set_model_mode(self, model_mode):
    self.model_mode = model_mode

  

