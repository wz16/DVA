import torch
import random

import importlib
def get_model(args):

    input_dim = args.static_model_input_dim
    output_dim = args.static_model_output_dim
    hidden_dim = args.static_model_hidden_dim
    model_params = {'model_def':args.static_model_def, 'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': hidden_dim}
    model_module = importlib.import_module('models.'+args.static_model_def)
    model = model_module.Net(**model_params)
    return model
class stochastic_wrapper(torch.nn.Module):

  def __init__(self, wrapper_mode, args, **kwargs):
    super(stochastic_wrapper, self).__init__()
    self.wrapper_mode = wrapper_mode

    if self.wrapper_mode == "bayesian" and "bayesian" in args.static_model_def:
      self.static_model = get_model(args)
    elif self.wrapper_mode == "ensemble":
      num_ensemble = 5
      self.num_ensemble = num_ensemble
      self.static_models = torch.nn.ModuleList()
      for i in range(num_ensemble):
        self.static_models.append(get_model(args))
      self.sample_id = 0
    else:
      raise Exception("wrong stochastic wrapper_mode or imcompatible ")
    

  def sample(self):
    if self.wrapper_mode == "bayesian":
      self.static_model.sample()
    elif self.wrapper_mode == "ensemble":
      self.sample_id += 1
      self.sample_id = self.sample_id%self.num_ensemble
      # self.sample_id = random.randint(0,self.num_ensemble-1)

  def forward(self, x):

    if self.wrapper_mode == 'bayesian':
      fx_hat = self.static_model(x)
    elif self.wrapper_mode == "ensemble":
      fx_hat = self.static_models[self.sample_id](x)
    return fx_hat
  def kl_divergence(self):
    if self.wrapper_mode == "bayesian":
        return self.static_model.kl_divergence()
    elif self.wrapper_mode == "ensemble":
      return 0
    raise Exception("no KL")
  def forward_multiple(self, x, n_prediction=5):
    results = []
    if self.wrapper_mode == 'bayesian' or self.wrapper_mode == 'ensemble':
      for i in range(n_prediction):
        self.sample()
        fx_hat = self.forward(x)
        results.append(fx_hat)

    return torch.stack(results)
