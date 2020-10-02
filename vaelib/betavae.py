"""Beta VAE class.

β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK
https://openreview.net/forum?id=Sy2fzU9gl

Understanding disentangling in β-VAE
https://arxiv.org/abs/1804.03599
"""

from typing import Tuple, Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import BaseVAE, kl_divergence_normal, nll_bernoulli


class Encoder(nn.Module):
    """Encoder q(z|x).

    Args:
        in_channels (int): Channel size of inputs.
        z_dim (int): Dimension size of latents.
    """

    def __init__(self, in_channels: int, z_dim: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(1024, 256)
        self.fc21 = nn.Linear(256, z_dim)
        self.fc22 = nn.Linear(256, z_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes z given x.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            mu (torch.Tensor): Mean of Gausssian, size `(b, z)`.
            var (torch.Tensor): Variance of Gausssian, size `(b, z)`.
        """

        h = self.conv(x)
        h = h.view(-1, 1024)
        h = F.relu(self.fc1(h))

        mu = self.fc21(h)
        var = F.softplus(self.fc22(h))

        return mu, var


class Decoder(nn.Module):
    """Decoder p(x|z).

    Args:
        in_channels (int): Channel size of inputs.
        z_dim (int): Dimension size of latents.
    """

    def __init__(self, in_channels: int, z_dim: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Encodes z given x.

        Args:
            z (torch.Tensor): Latents, size `(b, z)`.

        Returns:
            probs (torch.Tensor): Decoded observations, size `(b, c, h, w)`.
        """

        h = self.fc(z)
        h = h.view(-1, 64, 4, 4)
        probs = self.deconv(h)

        return probs


class BetaVAE(BaseVAE):
    """beta-VAE class.

    Args:
        in_channels (int, optional): Channel size of inputs.
        z_dim (int, optional): Dimension size of latents.
        beta (float, optional): Beta coefficient of KL term.
        capacity (float, optional): Capacity regularization term.
        do_anneal (bool, optional): If `True`, beta is given from kwargs.
    """

    def __init__(
        self,
        in_channels: int = 3,
        z_dim: int = 10,
        beta: float = 10.0,
        capacity: float = 0.0,
        do_anneal: bool = False,
    ):
        super().__init__()

        self.beta = beta
        self.capacity = capacity
        self.do_anneal = do_anneal

        # Modules
        self.encoder = Encoder(in_channels, z_dim)
        self.decoder = Decoder(in_channels, z_dim)

        # Prior
        self.register_buffer("p_mu", torch.zeros(1, z_dim))
        self.register_buffer("p_var", torch.ones(1, z_dim))

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

        # Encode and sample latents
        z_mu, z_var = self.encoder(x)
        z = z_mu + z_var ** 2 * torch.randn_like(z_var)

        # Decode reconstruction
        recon = self.decoder(z)

        # Loss calculation
        ce_loss = nll_bernoulli(x, recon, reduce=False)
        ce_loss = ce_loss.sum(dim=[1, 2, 3])

        beta = beta if self.do_anneal else self.beta
        kl_loss = kl_divergence_normal(z_mu, z_var, self.p_mu, self.p_var)
        kl_loss = beta * (kl_loss - self.capacity).abs()

        loss_dict = {
            "loss": ce_loss + kl_loss,
            "kl_loss": kl_loss,
            "ce_loss": ce_loss,
        }

        return (recon, z), loss_dict

    def sample(self, batch_size: int = 1, y: Optional[Tensor] = None) -> Tensor:
        """Samples data from model.

        Args:
            batch_size (int, optional): Batch size of sampled data.
            y (torch.Tensor, optional): Labels, size `(b,)`.

        Returns:
            x (torch.Tensor): Sampled observations, size `(b, c, h, w)`.
        """

        mu = self.p_mu.repeat(batch_size, 1)
        var = self.p_var.repeat(batch_size, 1)

        z = mu + var ** 0.5 * torch.randn_like(var)
        x = self.decoder(z)

        return x
