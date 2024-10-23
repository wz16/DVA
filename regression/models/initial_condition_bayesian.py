import torch
import torch.nn as nn
import importlib
from torch.nn.parameter import Parameter
from typing import Optional
from  .variational import GaussianVariational, ScaleMixture, BayesianModule, base_bayesian_model
import torch.nn.functional as F
class initial_condition_bayesian(BayesianModule):

  def __init__(self, n_data, state_dim, 
                 prior_pi = 1.0,
                 prior_sigma1 = 3.0,
                 prior_sigma2 = 0.0025, **kwargs):
    super(initial_condition_bayesian, self).__init__()

    w_mu = torch.empty(n_data, state_dim).uniform_(-0.2, 0.2)
    w_rho = torch.empty(n_data, state_dim).uniform_(-5.0, -4.0)

    self.w_posterior = GaussianVariational(w_mu, w_rho)

    self.w_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)

    # self.w = nn.Parameter(w_mu)


  def sample(self):
    if self.model_mode == "deterministic":
      self.w = self.w_posterior.mu
      return

    w = self.w_posterior.sample()

    w_log_prior = self.w_prior.log_prior(w)
    w_log_posterior = self.w_posterior.log_posterior()


    total_log_prior = w_log_prior
    total_log_posterior = w_log_posterior 
    self.kl_divergence = self.kld(total_log_prior, total_log_posterior)

    self.w = w
    return 

  def set_model_mode(self, model_mode):
    self.model_mode = model_mode

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