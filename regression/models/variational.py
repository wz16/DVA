import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# codes from (https://travis-ci.org/danielkelshaw/WeightUncertainty)

class GaussianVariational(nn.Module):

    """Gaussian Variational Weight Sampler.

    Section 3.2 of the 'Weight Uncertainty in Neural Networks' paper
    proposes the use of a Gaussian posterior in order to sample weights
    from the network for use in variational inference.
    """

    def __init__(self, mu: Tensor, rho: Tensor) -> None:

        """Gaussian Variational Weight Sampler.

        Parameters
        ----------
        mu : Tensor
            Mu used to shift the samples drawn from a unit Gaussian.
        rho : Tensor
            Rho used to generate the pointwise parameterisation of the
            standard deviation - used to scale the samples drawn a unit
            Gaussian.
        """

        super().__init__()

        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)

        self.w = None
        self.sigma = None

        self.normal = torch.distributions.Normal(0, 1)

    def sample_deterministic(self) -> Tensor:

        """Draws a sample from the posterior distribution.

        Samples a weight using:
            w = mu + log(1 + exp(rho)) * epsilon
                where epsilon ~ N(0, 1)

        Returns
        -------
        Tensor
            Sampled weight from the posterior distribution.
        """

        device = self.mu.device
        epsilon = 0
        self.sigma = torch.log(1 + torch.exp(self.rho)).to(device)
        self.w = self.mu + self.sigma * epsilon

        return self.w
    def sample(self) -> Tensor:

        """Draws a sample from the posterior distribution.

        Samples a weight using:
            w = mu + log(1 + exp(rho)) * epsilon
                where epsilon ~ N(0, 1)

        Returns
        -------
        Tensor
            Sampled weight from the posterior distribution.
        """

        device = self.mu.device
        epsilon = self.normal.sample(self.mu.size()).to(device)
        self.sigma = torch.log(1 + torch.exp(self.rho)).to(device)
        self.w = self.mu + self.sigma * epsilon

        return self.w
        # return self.mu

    def log_posterior(self) -> Tensor:

        """Log Likelihood for each weight sampled from the distribution.

        Calculates the Gaussian log likelihood of the sampled weight
        given the the current mean, mu, and standard deviation, sigma:

            LL = -log((2pi * sigma^2)^0.5) - 0.5(w - mu)^2 / sigma^2

        Returns
        -------
        Tensor
            Gaussian log likelihood for the weights sampled.
        """

        if self.w is None:
            raise ValueError('self.w must have a value.')

        log_const = np.log(np.sqrt(2 * np.pi))
        log_exp = ((self.w - self.mu) ** 2) / (2 * self.sigma ** 2)
        log_posterior = -log_const - torch.log(self.sigma) - log_exp

        return log_posterior.sum()


class ScaleMixture(nn.Module):

    """Scale Mixture Prior.

    Section 3.3 of the 'Weight Uncertainty in Neural Networks' paper
    proposes the use of a Scale Mixture prior for use in variational
    inference - this being a fixed-form prior.

    The authors note that, should the parameters be allowed to adjust
    during training, the prior changes rapidly and attempts to capture
    the empirical distribution of the weights. As a result the prior
    learns to fit poor initial parameters and struggles to improve.
    """

    def __init__(self, pi: float, sigma1: float, sigma2: float) -> None:

        """Scale Mixture Prior.

        The authors of 'Weight Uncertainty in Neural Networks' note:

            sigma1 > sigma2:
                provides a heavier tail in the prior density than is
                seen in a plain Gaussian prior.
            sigma2 << 1.0:
                causes many of the weights to a priori tightly
                concentrate around zero.

        Parameters
        ----------
        pi : float
            Parameter used to scale the two Gaussian distributions.
        sigma1 : float
            Standard deviation of the first normal distribution.
        sigma2 : float
            Standard deviation of the second normal distribution.
        """

        super().__init__()

        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        self.normal1 = torch.distributions.Normal(0, sigma1)
        self.normal2 = torch.distributions.Normal(0, sigma2)

    def log_prior(self, w: Tensor) -> Tensor:

        """Log Likelihood of the weight according to the prior.

        Calculates the log likelihood of the supplied weight given the
        prior distribution - the scale mixture of two Gaussians.

        Parameters
        ----------
        w : Tensor
            Weight to be used to calculate the log likelihood.

        Returns
        -------
        Tensor
            Log likelihood of the weights from the prior distribution.
        """

        likelihood_n1 = torch.exp(self.normal1.log_prob(w))
        likelihood_n2 = torch.exp(self.normal2.log_prob(w))

        p_scalemixture = self.pi * likelihood_n1 + (1 - self.pi) * likelihood_n2
        log_prob = torch.log(p_scalemixture).sum()

        return log_prob


class BayesianModule(nn.Module):

    """Base class for BNN to enable certain behaviour."""

    def __init__(self):
        super().__init__()

    def kld(self, *args):
        raise NotImplementedError('BayesianModule::kld()')
    


class base_bayesian_model(nn.Module):
    def __init__(self):
        super().__init__()
    def kl_divergence(self) -> Tensor:

        """Calculates the KL Divergence for each BayesianModule.

        The total KL Divergence is calculated by iterating through the
        BayesianModules in the model. KL Divergence for each module is
        calculated as the difference between the log_posterior and the
        log_prior.

        Returns
        -------
        kl : Tensor
            Total KL Divergence.
        """

        kl = 0
        for module in self.modules():
            if isinstance(module, BayesianModule):
                kl += module.kl_divergence

        return kl

    def elbo(self,
                inputs,
                targets,
                criterion,
                n_samples,
                w_complexity = 1):

        """Samples the ELBO loss for a given batch of data.

        The ELBO loss for a given batch of data is the sum of the
        complexity cost and a data-driven cost. Monte Carlo sampling is
        used in order to calculate a representative loss.

        Parameters
        ----------
        inputs : Tensor
            Inputs to the model.
        targets : Tensor
            Target outputs of the model.
        criterion : Any
            Loss function used to calculate data-dependant loss.
        n_samples : int
            Number of samples to use
        w_complexity : float
            Complexity weight multiplier.

        Returns
        -------
        Tensor
            Value of the ELBO loss for the given data.
        """

        loss = 0
        for sample in range(n_samples):
            outputs = self(inputs)
            loss += criterion(outputs, targets)
            loss += self.kl_divergence() * w_complexity

        return loss / n_samples