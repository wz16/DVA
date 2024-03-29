# codes from (https://travis-ci.org/danielkelshaw/WeightUncertainty)

from typing import Any, Optional

import torch.nn as nn
from torch import Tensor

from  .variational import BayesianModule




def variational_approximator(model: nn.Module) -> nn.Module:

    """Adds Variational Inference functionality to a nn.Module.

    Parameters
    ----------
    model : nn.Module
        Model to use variational approximation with.

    Returns
    -------
    model : nn.Module
        Model with additional variational approximation functionality.
    """

    def sample(self, model_mode="bayesian"):
        for module in self.modules():
            if isinstance(module, BayesianModule):
                module.sample(model_mode=model_mode)

    setattr(model, 'sample', sample)

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

    # add `kl_divergence` to the model
    setattr(model, 'kl_divergence', kl_divergence)


    def elbo(self,
             inputs: Tensor,
             targets: Tensor,
             criterion: Any,
             n_samples: int,
             w_complexity: Optional[float] = 1.0) -> Tensor:

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

    # add `elbo` to the model
    setattr(model, 'elbo', elbo)

    return model
