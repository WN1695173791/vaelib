
"""Base class for VAE models."""

from typing import Dict, Iterator, Optional, Tuple

import itertools

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class BaseVAE(nn.Module):
    """Base class for VAE models."""

    def __init__(self):
        super().__init__()

        self.encoder: nn.Module
        self.decoder: nn.Module

    def forward(self, x: Tensor, beta: float = 1.0) -> Dict[str, Tensor]:
        """Loss function for ELBO loss.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            beta (float, optional): Beta coefficient for KL loss.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Dict of lossses.
        """

        _, loss_dict = self.inference(x, beta)

        return loss_dict

    def reconstruct(self, x: Tensor) -> Tensor:
        """Reconstructs given observations.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            recon (torch.Tensor): Reconstructed observations, size
                `(b, c, h, w)`.
        """

        (recon, *_), _ = self.inference(x)

        return recon

    def inference(self, x: Tensor, beta: float = 1.0
                  ) -> Tuple[Tuple[Tensor, ...], Dict[str, Tensor]]:
        """Inferences reconstruction with ELBO loss calculation.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            beta (float, optional): Beta coefficient for KL loss.

        Returns:
            samples (tuple of torch.Tensor): Tuple of reconstructed or encoded
                data. The first element should be reconstructed observations.
            loss_dict (dict of [str, torch.Tensor]): Dict of lossses.
        """

        raise NotImplementedError

    def sample(self, batch_size: int) -> Tensor:
        """Samples data from model.

        Args:
            batch_size (int): Batch size of sampled data.

        Returns:
            x (torch.Tensor): Sampled observations, size `(b, c, h, w)`.
        """

        raise NotImplementedError

    def model_parameters(self) -> Iterator:
        """Iterator of parameters in encoder and decoder.

        Returns:
            params (iterator): Parameters of encoder and decoder.
        """

        return itertools.chain(
                   self.encoder.parameters(), self.decoder.parameters())

    def adversarial_parameters(self) -> Optional[Iterator]:
        """Model parameters for adversarial training.

        Returns:
            params (iterator or None): Parameters of discriminator.
        """

        return None


def kl_divergence_normal(mu0: Tensor, var0: Tensor, mu1: Tensor, var1: Tensor,
                         reduce: bool = True) -> Tensor:
    """Kullback Leibler divergence for 1-D Normal distributions.

    p = N(mu0, var0)
    q = N(mu1, var1)
    KL(p||q) = 1/2 * (var0/var1 + (mu1-mu0)^2/var1 - 1 + log(var1/var0))

    Args:
        mu0 (torch.Tensor): Mean vector of p, size.
        var0 (torch.Tensor): Diagonal variance of p.
        mu1 (torch.Tensor): Mean vector of q.
        var1 (torch.Tensor): Diagonal variance of q.
        reduce (bool, optional): If `True`, sum calculated loss for each
            data point.

    Returns:
        kl (torch.Tensor): Calculated kl divergence for each data.
    """

    diff = mu1 - mu0
    kl = (var0 / var1 + diff ** 2 / var1 - 1 + (var1 / var0).log()) * 0.5

    if reduce:
        return kl.sum(-1)
    return kl


def nll_bernoulli(x: Tensor, probs: Tensor, reduce: bool = True) -> Tensor:
    """Negative log likelihood for Bernoulli distribution.

    Ref)
    https://pytorch.org/docs/stable/_modules/torch/distributions/bernoulli.html#Bernoulli
    https://github.com/pytorch/pytorch/blob/master/torch/distributions/utils.py#L75

    Args:
        x (torch.Tensor): Inputs tensor, size `(*, dim)`.
        probs (torch.Tensor): Probability parameter, size `(*, dim)`.
        reduce (bool, optional): If `True`, sum calculated loss for each
            data point.

    Returns:
        nll (torch.Tensor): Calculated nll for each data, size `(*,)` if
            `reduce` is `True`, `(*, dim)` otherwise.
    """

    probs = probs.clamp(min=1e-6, max=1 - 1e-6)
    logits = torch.log(probs) - torch.log1p(-probs)
    nll = F.binary_cross_entropy_with_logits(logits, x, reduction="none")

    if reduce:
        return nll.sum(-1)
    return nll
