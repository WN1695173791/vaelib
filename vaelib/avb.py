
"""Adversarial Variational Bayes (AVB).

Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative
Adversarial Networks
http://arxiv.org/abs/1701.04722

Ref)
https://github.com/gdikov/adversarial-variational-bayes
http://seiya-kumada.blogspot.com/2018/07/adversarial-variational-bayes.html
https://github.com/LMescheder/AdversarialVariationalBayes
https://nbviewer.jupyter.org/github/hayashiyus/Thermal-VAE/blob/master/adversarial%20variational%20bayes%20toy%20example-cyclical-annealing-MNIST-898-4000.ipynb
"""

from typing import Dict, Iterator, Optional, Tuple

import torch
from torch import Tensor, nn

from .base import BaseVAE, nll_bernoulli


class Encoder(nn.Module):
    """Encoder q(z|x, e).

    Args:
        in_channels (int): Channel size of inputs.
        z_dim (int): Dimension size of latents.
        e_dim (int): Dimension size of noises.
    """

    def __init__(self, in_channels: int, z_dim: int, e_dim: int):
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

        self.fc_x = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim),
            nn.ReLU(),
        )

        self.fc_e = nn.Sequential(
            nn.Linear(e_dim, z_dim),
            nn.ReLU(),
        )

        self.fc = nn.Linear(z_dim * 2, z_dim)

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        """Encodes z given x, e.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            e (torch.Tensor): Noises, size `(b, e)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(b, z)`.
        """

        h_x = self.conv(x)
        h_x = h_x.view(-1, 1024)
        h_x = self.fc_x(h_x)

        h_e = self.fc_e(e)

        z = self.fc(torch.cat([h_x, h_e], dim=1))

        return z


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


class Discriminator(nn.Module):
    """Discriminator T(x, z).

    Args:
        in_channels (int): Channel size of inputs.
        z_dim (int): Dimension size of latents.
    """

    def __init__(self, in_channels: int, z_dim: int):
        super().__init__()

        self.disc_x = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.fc_x = nn.Linear(1024, 256)

        self.disc_z = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
        )

        self.fc = nn.Linear(512, 1)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        """Discriminate p(x)p(z) from p(x)q(z|x).

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            z (torch.Tensor): Latents, size `(b, z)`.

        Returns:
            logits (torch.Tensor): Logits, size `(b, 1)`.
        """

        # Encode
        h_x = self.disc_x(x)
        h_x = self.fc_x(h_x.view(-1, 1024))
        h_z = self.disc_z(z)

        # Calculate logits
        logits = self.fc(torch.cat([h_x, h_z], dim=1))

        return logits


class AVB(BaseVAE):
    """Adversarial Variational Bayes.

    Args:
        in_channels (int, optional): Channel size of inputs.
        z_dim (int, optional): Dimension size of latents.
        e_dim (int, optional): Dimension size of noises.
    """

    def __init__(self, in_channels: int = 3, z_dim: int = 10, e_dim: int = 10):
        super().__init__()

        self.z_dim = z_dim
        self.e_dim = e_dim

        self.encoder = Encoder(in_channels, z_dim, e_dim)
        self.decoder = Decoder(in_channels, z_dim)

        # Discriminator for estimating density ratio
        self.discriminator = Discriminator(in_channels, z_dim)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        # Prior
        self.register_buffer("p_mu", torch.zeros(1, z_dim))
        self.register_buffer("p_var", torch.ones(1, z_dim))

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

        # Data size
        batch = x.size(0)

        # Sample e
        e_mu = x.new_zeros((batch, self.e_dim))
        e_var = x.new_ones((batch, self.e_dim))
        e = e_mu + e_var ** 0.5 * torch.randn_like(e_var)

        # Sample z_p from prior
        z_mu = x.new_zeros((batch, self.z_dim))
        z_var = x.new_ones((batch, self.z_dim))
        z_p = z_mu + z_var ** 0.5 * torch.randn_like(z_var)

        # Encode latents
        z_q = self.encoder(x, e)

        # Decode reconstruction
        recon = self.decoder(z_q)

        # Discriminator
        logits = self.discriminator(x, z_q)
        logits = beta * logits.sum(dim=1)

        # Reconstruction loss
        ce_loss = nll_bernoulli(x, recon, reduce=False)
        ce_loss = ce_loss.sum(dim=[1, 2, 3])

        # Discriminator loss
        log_d_q = self.bce_loss(
            self.discriminator(x, z_q.detach()), z_q.new_ones((batch, 1)))
        log_d_p = self.bce_loss(
            self.discriminator(x, z_p), z_p.new_zeros((batch, 1)))
        loss_d = (log_d_q + log_d_p).sum(dim=1)

        # Returned loss
        loss_dict = {
            "loss": logits + ce_loss, "ce_loss": ce_loss, "logits": logits,
            "loss_d": loss_d,
        }

        return (recon, z_q), loss_dict

    def sample(self, batch_size: int) -> Tensor:
        """Samples data from model.

        Args:
            batch_size (int): Batch size of sampled data.

        Returns:
            x (torch.Tensor): Sampled observations, size `(b, c, h, w)`.
        """

        mu = self.p_mu.repeat(batch_size, 1)
        var = self.p_var.repeat(batch_size, 1)

        z = mu + var ** 0.5 * torch.randn_like(var)
        x = self.decoder(z)

        return x

    def adversarial_parameters(self) -> Optional[Iterator]:
        """Model parameters for adversarial training.

        Returns:
            params (iterator or None): Parameters of discriminator.
        """

        return self.discriminator.parameters()
