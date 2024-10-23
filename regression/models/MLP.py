import torch
import torch.nn as nn
import importlib
import torch.nn.functional as F


class Net(torch.nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
    super(Net, self).__init__()

    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    h = torch.nn.Tanh()(self.linear1(x))
    h = torch.nn.Tanh()(self.linear2(h))
    return self.linear3(h)
  def sample(self):
    pass
  def kl_divergence(self):
    return 0

