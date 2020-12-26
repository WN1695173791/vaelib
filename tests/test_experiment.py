import tempfile
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import vaelib
from vaelib.base import BaseVAE


def test_trainer_run() -> None:

    model = TempModel()
    train_data = TempDataset()
    test_data = TempDataset()

    org_params = deepcopy(model.state_dict())

    with tempfile.TemporaryDirectory() as logdir:
        trainer = vaelib.Trainer(logdir=logdir)
        trainer.run(model, train_data, test_data)

        root = trainer._logdir
        assert (root / "training.log").exists()
        assert (root / "config.json").exists()

    updated_params = model.state_dict()
    for key in updated_params:
        assert not (updated_params[key] == org_params[key]).all()


class TempModel(BaseVAE):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Linear(64 * 64 * 3, 10)
        self.decoder = nn.Linear(10, 64 * 64 * 3)

    def inference(
        self, x: Tensor, y: Optional[Tensor] = None, beta: float = 1.0
    ) -> Tuple[Tuple[Tensor, ...], Dict[str, Tensor]]:

        z = self.encoder(x.view(-1, 64 * 64 * 3))
        recon = self.decoder(z)
        recon = recon.view(-1, 3, 64, 64)

        loss_dict = {"loss": F.mse_loss(recon, x)}

        return (recon,), loss_dict

    def sample(self, batch_size: int = 1, y: Optional[Tensor] = None) -> Tensor:

        x = torch.rand(batch_size, 10)
        sample = self.decoder(x)
        sample = sample.view(-1, 3, 64, 64)

        return sample


class TempDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        self._data = torch.rand(10, 3, 64, 64)
        self._label = torch.randint(0, 100, (10,))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self._data[index], self._label[index]

    def __len__(self) -> int:
        return self._data.size(0)
