import torch
import torch.nn as nn
import importlib
from torch.nn.parameter import Parameter
from typing import Optional
from  .variational import GaussianVariational, ScaleMixture, BayesianModule, base_bayesian_model
import torch.nn.functional as F

class linear_layer(BayesianModule):

  def __init__(self, input_dim, output_dim, 
                 prior_pi = 0.5,
                 prior_sigma1 = 1.0,
                 prior_sigma2 = 0.0025, **kwargs):
    super(linear_layer, self).__init__()

    w_mu = torch.empty(output_dim, input_dim).uniform_(-0.2, 0.2)
    w_rho = torch.empty(output_dim, input_dim).uniform_(-5.0, -4.0)

    bias_mu = torch.empty(output_dim).uniform_(-0.2, 0.2)
    bias_rho = torch.empty(output_dim).uniform_(-5.0, -4.0)

    self.w_posterior = GaussianVariational(w_mu, w_rho)
    self.bias_posterior = GaussianVariational(bias_mu, bias_rho)

    self.w_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)
    self.bias_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)

    self.kl_divergence = 1

    

  def sample(self, model_mode="bayesian"):
    
    if model_mode == "deterministic":

      w = self.w_posterior.sample_deterministic()
      b = self.bias_posterior.sample_deterministic()
      
    elif model_mode == "bayesian":
      w = self.w_posterior.sample()
      b = self.bias_posterior.sample()

   

    w_log_prior = self.w_prior.log_prior(w)
    b_log_prior = self.bias_prior.log_prior(b)
    w_log_posterior = self.w_posterior.log_posterior()
    b_log_posterior = self.bias_posterior.log_posterior()

    total_log_prior = w_log_prior + b_log_prior
    total_log_posterior = w_log_posterior + b_log_posterior
    self.kl_divergence = self.kld(total_log_prior, total_log_posterior)

    self.w = w
    self.b = b
    return 

  def forward(self, x):

    return F.linear(x, self.w, self.b)

  def kld(self, log_prior, log_posterior):

      """Calculates the KL Divergence.

      Uses the weight sampled from the posterior distribution to
      calculate the KL Divergence between the prior and posterior.

      Parameters
      ----------
      log_prior : Tensor
          Log likelihood drawn from the prior.
      log_posterior : Tensor
          Log likelihood drawn from the approximate posterior.

      Returns
      -------
      Tensor
          Calculated KL Divergence.
      """

      return log_posterior - log_prior