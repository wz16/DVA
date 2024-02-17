import torch
import torch.nn as nn
import importlib
from torch.nn.parameter import Parameter
from typing import Optional
from  .variational import GaussianVariational, ScaleMixture, BayesianModule, base_bayesian_model
import torch.nn.functional as F
from .linear_layer_bayesian import linear_layer
from .variational_approximator import variational_approximator

from torchdiffeq import odeint
@variational_approximator
class Net(base_bayesian_model):

  def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
    super(Net, self).__init__()

    self.bl1 = linear_layer(input_dim=input_dim,output_dim=hidden_dim)
    self.bl2 = linear_layer(input_dim=hidden_dim,output_dim=hidden_dim)
    self.bl3 = linear_layer(input_dim=hidden_dim,output_dim=output_dim)
  
  # def forward(self, x, t_span):
  #   return odeint(self.ode_function, x, t_span)
    

  def forward(self, x):
    x = F.tanh(self.bl1(x))
    x = F.tanh(self.bl2(x))
    x = self.bl3(x)
    return x


  

