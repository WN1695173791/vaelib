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

    def forward(
        self, x: Tensor, y: Optional[Tensor] = None, beta: float = 1.0
    ) -> Dict[str, Tensor]:
        """Loss function for ELBO loss.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            y (torch.Tensor, optional): Labels, size `(b,)`.
            beta (float, optional): Beta coefficient for KL loss.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Dict of lossses.
        """

        _, loss_dict = self.inference(x, y, beta)

        return loss_dict

    def reconstruct(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """Reconstructs given observations.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            y (torch.Tensor, optional): Labels, size `(b,)`.

        Returns:
            recon (torch.Tensor): Reconstructed observations, size
                `(b, c, h, w)`.
        """

        (recon, *_), _ = self.inference(x, y)

        return recon

    def inference(
        self, x: Tensor, y: Optional[Tensor] = None, beta: float = 1.0
    ) -> Tuple[Tuple[Tensor, ...], Dict[str, Tensor]]:
        """Inferences reconstruction with ELBO loss calculation.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            y (torch.Tensor, optional): Labels, size `(b,)`.
            beta (float, optional): Beta coefficient for KL loss.

        Returns:
            samples (tuple of torch.Tensor): Tuple of reconstructed or encoded data. The
                first element should be reconstructed observations.
            loss_dict (dict of [str, torch.Tensor]): Dict of lossses.
        """

        raise NotImplementedError

    def sample(self, batch_size: int = 1, y: Optional[Tensor] = None) -> Tensor:
        """Samples data from model.

        Args:
            batch_size (int, optional): Batch size of sampled data.
            y (torch.Tensor, optional): Labels, size `(b,)`.

        Returns:
            x (torch.Tensor): Sampled observations, size `(b, c, h, w)`.
        """

        raise NotImplementedError

    def model_parameters(self) -> Iterator:
        """Iterator of parameters in encoder and decoder.

        Returns:
            params (iterator): Parameters of encoder and decoder.
        """

        return itertools.chain(self.encoder.parameters(), self.decoder.parameters())

    def adversarial_parameters(self) -> Optional[Iterator]:
        """Model parameters for adversarial training.

        Returns:
            params (iterator or None): Parameters of discriminator.
        """

        return None


def kl_divergence_normal(
    mu0: Tensor, var0: Tensor, mu1: Tensor, var1: Tensor, reduce: bool = True
) -> Tensor:
    """Kullback Leibler divergence for 1-D Normal distributions.

    p = N(mu0, var0)
    q = N(mu1, var1)
    KL(p||q) = 1/2 * (var0/var1 + (mu1-mu0)^2/var1 - 1 + log(var1/var0))

    Args:
        mu0 (torch.Tensor): Mean vector of p, size.
        var0 (torch.Tensor): Diagonal variance of p.
        mu1 (torch.Tensor): Mean vector of q.
        var1 (torch.Tensor): Diagonal variance of q.
        reduce (bool, optional): If `True`, sum calculated loss for each data point.

    Returns:
        kl (torch.Tensor): Calculated kl divergence for each data.
    """

    diff = mu1 - mu0
    kl = (var0 / var1 + diff ** 2 / var1 - 1 + (var1 / var0).log()) * 0.5

    if reduce:
        return kl.sum(-1)
    return kl


def kl_divergence_normal_diff(
    delta_mu: Tensor, delta_var: Tensor, var: Tensor, reduce: bool = True
) -> Tensor:
    """Kullback Leibler divergence for 1-D normal with parameter differences.

    p = N(mu, var)
    q = N(mu + delta_mu, var * delta_var)

    Args:
        delta_mu (torch.Tensor): Difference of mu.
        delta_var(torch.Tensor): Difference of variance.
        var (torch.Tensor): Variance of original normnal.
        reduce (bool, optional): If `True`, sum calculated loss for each data point.

    Returns:
        kl (torch.Tensor): Calculated kl divergence for each data.
    """

    kl = (delta_mu ** 2 / var + delta_var - delta_var.log() - 1) * 0.5

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
        reduce (bool, optional): If `True`, sum calculated loss for each data point.

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


def nll_logistic(
    x: Tensor,
    mu: Tensor,
    scale: Tensor,
    reduce: bool = True,
    binsize: float = 1 / 256,
) -> Tensor:
    """Negative log likelihood for discretized logsitc.


    ref) https://github.com/pclucas14/iaf-vae/blob/master/utils.py#L74

    Args:
        x (torch.Tensor): Inputs tensor, size `(*, dim)`.
        mu (torch.Tensor): Mean of logsitc function, size `(*, dim)`.
        scale (torch.Tensor): Scale of logsitc function, size `(*, dim)`.
        reduce (bool, optional): If `True`, sum calculated loss for each data point.

    Returns:
        nll (torch.Tensor): Calculated nll for each data, size `(*,)` if
            `reduce` is `True`, `(*, dim)` otherwise.
    """

    x = (torch.floor(x / binsize) * binsize - mu) / scale
    nll = -torch.log(torch.sigmoid(x + binsize / scale) - torch.sigmoid(x) + 1e-7)

    if reduce:
        return nll.sum(-1)
    return nll
