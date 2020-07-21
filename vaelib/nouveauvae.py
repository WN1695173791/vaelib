
"""Nouveau VAE (NVAE).

ref)
Arash Vahdat, Jan Kautz. "NVAE: A Deep Hierarchical Variational Autoencoder"
https://arxiv.org/abs/2007.03898

Tim Salimans, Andrej Karpathy, Xi Chen, Diederik P. Kingma. "PixelCNN++:
Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other
Modifications"
https://arxiv.org/abs/1701.05517, https://github.com/openai/pixel-cnn

Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen.
"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
https://arxiv.org/abs/1801.04381

Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu.
"Squeeze-and-Excitation Networks"
https://arxiv.org/abs/1709.01507

Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever,
Max Welling.
"Improving Variational Inference with Inverse Autoregressive Flow"
https://arxiv.org/abs/1606.04934, https://github.com/openai/iaf
Other implementation: https://github.com/pclucas14/iaf-vae
"""

from typing import Dict, Optional, Tuple, Union, List

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import BaseVAE, kl_divergence_normal, nll_logistic

Iterable = Union[List[int], Tuple[int]]


class SwishActivation(nn.Module):
    """Swish activation: f(u) = u / (1 + exp(-u))."""

    def forward(self, x: Tensor) -> Tensor:
        """Forwards.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            y (torch.Tensor): Activated tensor.
        """

        return x / (1 + (-x).exp())


class SELayer(nn.Module):
    """Squeeze-and Excitation layer.

    Hu+ 2017, "Squeeze-and-Excitation Networks."

    ref)
    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py

    Args:
        in_channels (int): Channel size of inputs.
        reduction (int, optional): Reduction scale.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward computation.

        Args:
            x (torch.Tensor): Input tensor, size `(b, c, h, w)`.

        Returns:
            y (torch.Tensor): Computed tensor, size `(b, c, h, w)`.
        """

        b, c, *_ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GenerativeResidualCell(nn.Module):
    """Residual cell for generative model.

    Args:
        in_channels (int): Channel size of inputs.
        expansion_dim (int): Dimension size for expansion.
    """

    def __init__(self, in_channels: int, expansion_dim: int):
        super().__init__()

        self.resblock = nn.Sequential(
            # BN
            nn.BatchNorm2d(in_channels),

            # Conv. 1x1
            nn.Conv2d(in_channels, expansion_dim * in_channels, 1),

            # BN - Swish
            nn.BatchNorm2d(expansion_dim * in_channels),
            SwishActivation(),

            # dep. sep. conv. 5x5
            nn.Conv2d(expansion_dim * in_channels, expansion_dim * in_channels,
                      kernel_size=5, stride=1, padding=2,
                      groups=expansion_dim * in_channels),

            # BN - Swish
            nn.BatchNorm2d(expansion_dim * in_channels),
            SwishActivation(),

            # Conv. 1x1
            nn.Conv2d(expansion_dim * in_channels, in_channels, 1),

            # BN
            nn.BatchNorm2d(in_channels),

            # SE
            SELayer(in_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward computation.

        Args:
            x (torch.Tensor): Input tensor, size `(b, c, h, w)`.

        Returns:
            y (torch.Tensor): Computed tensor, size `(b, c, h, w)`.
        """

        return x + self.resblock(x)


class EncodingResidualCell(nn.Module):
    """Residual cell for encoder.

    Args:
        in_channels (int): Channel size of inputs.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self.resblock = nn.Sequential(
            # BN - Swish
            nn.BatchNorm2d(in_channels),
            SwishActivation(),

            # Conv. 3x3
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),

            # BN - Swish
            nn.BatchNorm2d(in_channels),
            SwishActivation(),

            # Conv. 3x3
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),

            # SE
            SELayer(in_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward computation.

        Args:
            x (torch.Tensor): Input tensor, size `(b, c, h, w)`.

        Returns:
            y (torch.Tensor): Computed tensor, size `(b, c, h, w)`.
        """

        return x + self.resblock(x)


class HierarchicalLayer(nn.Module):
    """Hierarchical layer.

    Args:
        in_channels (int): Number of channels in inputs.
        z_channels (int): Number of channels in z.
        expansion_dim (int): Dimension size for expansion in residual cell.
        num_cells (int): Number of residual cells in block.
        temperature (float, optional): Temperature of the prior for sampling.
        do_downsample (bool, optional): If `True`, down & up sample inputs.
        up_channels (int, optional): Number of channels in upsampled inputs.
    """

    def __init__(self, in_channels: int, z_channels: int, expansion_dim: int,
                 num_cells: int, temperature: float = 1.0,
                 do_downsample: bool = False, up_channels: int = 3):
        super().__init__()

        # Residual blocks
        self.inference_block = nn.ModuleList(
            [EncodingResidualCell(in_channels) for _ in range(num_cells)])
        self.generative_block = nn.ModuleList(
            [GenerativeResidualCell(in_channels, expansion_dim)
             for _ in range(num_cells)])

        # Conv for z params
        self.conv_inf = nn.Conv2d(in_channels, z_channels * 2, 1)
        self.conv_gen = nn.Conv2d(in_channels, z_channels * 2, 1)
        self.conv_cat = nn.Conv2d(z_channels, in_channels, 1)

        # Down and up sample function, used only if do_downsample is True
        self.do_downsample = do_downsample
        self.down_sample = nn.Conv2d(
            up_channels, in_channels, kernel_size=1, stride=2)
        self.up_sample = nn.ConvTranspose2d(
            in_channels, up_channels, kernel_size=4, stride=2, padding=1)

        # Temperature for prior
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward computation for inference.

        Args:
            x (torch.Tensor): Input tensor, size `(b, c, h, w)`.

        Returns:
            x (torch.Tensor): Inferred states.
            delta_mu (torch.Tensor, optional): Encoded delta mu of q(z|x).
            delta_var (torch.Tensor, optional): Encoded delta log variance of
                q(z|x).
        """

        if self.do_downsample:
            x = self.down_sample(x)

        for layer in self.inference_block:
            x = layer(x)

        # Encode variational parameters
        delta_mu, delta_logvar = torch.chunk(self.conv_inf(x), 2, dim=1)

        return x, delta_mu, delta_logvar

    def inverse(self, x: Tensor, delta_mu: Optional[Tensor] = None,
                delta_logvar: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        """Inverse computation for generation.

        Args:
            x (torch.Tensor): Input tensor, size `(b, c, h, w)`.
            delta_mu (torch.Tensor, optional): Encoded delta mu of q(z|x).
            delta_var (torch.Tensor, optional): Encoded delta log variance of
                q(z|x).

        Returns:
            x (torch.Tensor): Generated samples.
            kl_loss (torch.Tensor): Calculated KL loss for latents.
        """

        # Prior params
        p_mu, p_logvar = torch.chunk(self.conv_gen(x), 2, dim=1)

        # Concat
        if delta_mu is not None and delta_logvar is not None:
            mu = p_mu + delta_mu
            var = F.softplus(p_logvar + delta_logvar)
        else:
            mu = p_mu
            var = F.softplus(p_logvar) * self.temperature ** 2

        # Sample latents
        z = mu + var ** 0.5 + torch.randn_like(var)

        # Concat input and sampled latents
        x = x + self.conv_cat(z)

        # Generate
        for layer in self.generative_block:
            x = layer(x)

        # Upsample
        if self.do_downsample:
            x = self.up_sample(x)

        # Calculate kl
        if delta_mu is not None and delta_logvar is not None:
            kl_loss = kl_divergence_normal(
                mu, var, p_mu, F.softplus(p_logvar), reduce=False)
        else:
            kl_loss = torch.zeros_like(p_mu)
        kl_loss = kl_loss.sum(dim=[1, 2, 3])

        return x, kl_loss


class NouveauVAE(BaseVAE):
    """Nouveau VAE class.

    Args:
        in_channels (int, optional): Channel size of inputs.
        num_nflows (int, optional): Number of normalizing flows.
        num_groups (iterable, optional): Iterable of number of groups in each
            scale.
        z_channels (int, optional): Number of channels in z.
        enc_channels (int, optional): Number of initial channels in encoder.
        num_rescells (int, optional): Number of residual cells per group.
        annealing_lmd (float, optional): Coefficient of the smoothness loss.
        temperature (float, optional): Temperature of the prior for sampling.
        expansion_dim (int, optional): Dimension size for expansion in residual
            cell.
    """

    def __init__(self, in_channels: int = 3, in_dims: int = 32,
                 num_nflows: int = 0, num_groups: Iterable = (30,),
                 z_channels: int = 20, enc_channels: int = 128,
                 num_rescells: int = 2, annealing_lmd: float = 0.1,
                 temperature: float = 0.7, expansion_dim: int = 3):
        super().__init__()

        # Hierarchical blocks
        layers = []
        for i, groups in enumerate(num_groups[::-1]):
            for j in range(groups):
                if j == 0:
                    # Convert channel size
                    if i == 0:
                        layers.append(HierarchicalLayer(
                            enc_channels, z_channels, expansion_dim,
                            num_rescells, temperature, True, in_channels))
                    else:
                        layers.append(HierarchicalLayer(
                            enc_channels * 2, z_channels, expansion_dim,
                            num_rescells, temperature, True, enc_channels))
                        enc_channels *= 2
                else:
                    layers.append(HierarchicalLayer(
                        enc_channels, z_channels, expansion_dim, num_rescells,
                        temperature, do_downsample=False))

        self.layers = nn.ModuleList(layers)

        # Parameter
        self.annealing_lmd = annealing_lmd

        # Initial states of latents
        self.h_dims = in_dims // (len(num_groups) + 1)
        self.h_init = nn.Parameter(torch.zeros(1, enc_channels, 1, 1))

        # Log scale for outputs
        self.log_scale = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def inference(self, x: Tensor, y: Optional[Tensor] = None,
                  beta: float = 1.0
                  ) -> Tuple[Tuple[Tensor, ...], Dict[str, Tensor]]:
        """Inferences reconstruction with ELBO loss calculation.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            y (torch.Tensor, optional): Labels, size `(b,)`.
            beta (float, optional): Beta coefficient for KL loss.

        Returns:
            samples (tuple of torch.Tensor): Tuple of reconstructed or encoded
                data. The first element should be reconstructed observations.
            loss_dict (dict of [str, torch.Tensor]): Dict of lossses.
        """

        # Save original inputs
        inputs = x

        # Bottom-up inference
        q_params_list = []
        for layer in self.layers:
            x, *params = layer(x)
            q_params_list.append(params)

        # Top-down generation
        kl_loss = x.new_zeros((x.size(0),))
        h = self.h_init.expand_as(x)
        for layer, q_params in zip(self.layers[::-1], q_params_list[::-1]):
            h, _kl_tmp = layer.inverse(h, *q_params)
            kl_loss += _kl_tmp

        recon = h.clamp(-0.5 + 1 / 512, 0.5 - 1 / 512)

        # NLL loss
        nll_loss = nll_logistic(
            recon, inputs, self.log_scale.exp(), reduce=False)
        nll_loss = nll_loss.sum(dim=[1, 2, 3])

        # KL annealing
        kl_loss *= beta

        # Spectral regularization
        sr_loss = x.new_zeros((x.size(0),))
        for layer in self.layers:
            for name, value in layer.named_parameters():
                if name == "weight":
                    weight = value.view(value.size(0), -1)
                    v = torch.randn(weight.size(1))
                    u = weight @ v
                    v = weight.t() @ u
                    sr_loss += u.norm() / v.norm()
        sr_loss *= self.annealing_lmd

        # ELBO loss
        loss = nll_loss + kl_loss + sr_loss

        # Bit loss per pixel
        _, *x_dims = x.size()
        pixel_num = torch.tensor(x_dims).prod()
        bit_loss = (loss / pixel_num + math.log(256)) / math.log(2)

        # Loss dict
        loss_dict = {"loss": loss, "bit_loss": bit_loss, "nll_loss": nll_loss,
                     "kl_loss": kl_loss, "sr_loss": sr_loss}

        return (recon,), loss_dict

    def sample(self, batch_size: int = 1, y: Optional[Tensor] = None
               ) -> Tensor:
        """Samples data from model.

        Args:
            batch_size (int, optional): Batch size of sampled data.
            y (torch.Tensor, optional): Labels, size `(b,)`.

        Returns:
            x (torch.Tensor): Sampled observations, size `(b, c, h, w)`.
        """

        # Top-down generation
        h = self.h_init.repeat(batch_size, 1, self.h_dims, self.h_dims)
        for layer in self.layers[::-1]:
            h, _ = layer.inverse(h)

        sample = h.clamp(-0.5 + 1 / 512, 0.5 - 1 / 512)

        return sample
