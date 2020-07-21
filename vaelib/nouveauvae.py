
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
