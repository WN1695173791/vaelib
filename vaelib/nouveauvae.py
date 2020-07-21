
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
https://arxiv.org/abs/1606.04934
Example code: https://github.com/pclucas14/iaf-vae
"""

from typing import Dict, Optional, Tuple, Union, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import BaseVAE, kl_divergence_normal, nll_bernoulli

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
                      kernel_size=5, groups=expansion_dim * in_channels),

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
            nn.Conv2d(in_channels, in_channels, 3),

            # BN - Swish
            nn.BatchNorm2d(in_channels),
            SwishActivation(),

            # Conv. 3x3
            nn.Conv2d(in_channels, in_channels, 3),

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
    """Inverse Autoregressive Flow layer.

    Args:
        in_channels (int): Channel size of inputs.
        z_channels (int): Number of channels in z.
        expansion_dim (int): Dimension size for expansion in residual cell.
        num_cells (int): Number of residual cells in block.
        do_downsample (bool): If `True`, down & up sample inputs.
    """

    def __init__(self, in_channels: int, z_channels: int, expansion_dim: int,
                 num_cells: int, do_downsample: bool = False):
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
            in_channels * 2, in_channels, kernel_size=1, stride=2)
        self.up_sample = nn.ConvTranspose2d(
            in_channels, in_channels * 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward computation for inference.

        Args:
            x (torch.Tensor): Input tensor, size `(b, c, h, w)`.

        Returns:
            x (torch.Tensor): Inferred states.
            q_mu (torch.Tensor): Encoded mu of q(z|x).
            q_var (torch.Tensor): Encoded log variance of q(z|x).
        """

        if self.do_downsample:
            x = self.down_sample(x)

        for layer in self.inference_block:
            x = layer(x)

        # Encode variational parameters
        q_mu, q_logvar = torch.chunk(self.conv_inf(x), 2, dim=1)

        return x, q_mu, q_logvar

    def inverse(self, x: Tensor, q_mu: Optional[Tensor] = None,
                q_logvar: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Inverse computation for generation.

        Args:
            x (torch.Tensor): Input tensor, size `(b, c, h, w)`.
            q_mu (torch.Tensor, optional): Encoded mu of q(z|x).
            q_var (torch.Tensor, optional): Encoded log variance of q(z|x).

        Returns:
            x (torch.Tensor): Generated samples.
            kl_loss (torch.Tensor): Calculated KL loss for latents.
        """

        # Prior params
        p_mu, p_logvar = torch.chunk(self.conv_gen(x), 2, dim=1)

        # Concat
        if q_mu is not None and q_logvar is not None:
            mu = p_mu + q_mu
            var = F.softplus(p_logvar + q_logvar)
        else:
            mu = p_mu * 2
            var = F.softplus(p_logvar * 2)

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
        if q_mu is not None and q_logvar is not None:
            kl_loss = kl_divergence_normal(
                q_mu, F.softplus(q_logvar), p_mu, F.softplus(p_logvar), False)
        else:
            kl_loss = torch.zeros_like(p_mu)
        kl_loss = kl_loss.sum(dim=[1, 2, 3])

        return x, kl_loss
