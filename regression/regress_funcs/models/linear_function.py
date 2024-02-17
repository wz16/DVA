import torch
import torch.nn as nn
import importlib
from torch.nn.parameter import Parameter


class Net(torch.nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
    super(Net, self).__init__()

    self.weight = Parameter(torch.rand(output_dim))
    self.bias = Parameter(torch.rand(output_dim))


  def forward(self, x):
    return self.weight*x+self.bias

