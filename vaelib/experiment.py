from typing import Dict, DefaultDict, Union, Optional, Any

import collections
import dataclasses
import json
import logging
import os
import pathlib
import time

import torch
from torch import Tensor, optim
from torch.optim import optimizer
from torch.utils.data import dataloader
from torch.utils.data.dataset import Dataset
from torchvision.utils import make_grid

import vaelib

try:
    import matplotlib.pyplot as plt
    import tensorboardX as tb
    import tqdm
    IS_SUCCESSFUL = True
except ImportError:
    IS_SUCCESSFUL = False


@dataclasses.dataclass
class Config:
    seed: int = 0
    batch_size: int = 64
    max_steps: int = 2
    test_interval: int = 2
    save_interval: int = 2
    beta_annealer_params: dict = {}
    max_grad_value: float = 5.0
    max_grad_norm: float = 100.0
    logdir: Union[str, os.PathLike] = "./logs/"
    gpus: Optional[str] = None


class Trainer:
    """Trainer class for ML models.

    Args:
        seed: Random seed.
        batch_size: Batch size for training and testing.
        max_steps: Max number of training steps.
        test_interval: Interval steps for testing.
        save_interval: Interval steps for saving checkpoints.
        beta_annealer_params: Parameter dict for beta annealing.
        max_grad_value: Max gradient value.
        max_grad_norm: Max gradient norm.
        logdir: Path to log directory.
        gpus: GPU options.
    """

    def __init__(self, **kwargs: Any) -> None:

        self._config = Config(**kwargs)
        self._global_steps = 0
        self._postfix: Dict[str, float] = {}
        self._beta = 1.0

        self._logdir: pathlib.Path
        self._logger: logging.Logger
        self._writer: tb.SummaryWriter
        self._train_loader: dataloader.DataLoader
        self._test_loader: dataloader.DataLoader
        self._optimizer: optimizer.Optimizer
        self._adv_optimizer: Optional[optimizer.Optimizer]
        self._beta_anneler: vaelib.LinearAnnealer
        self._device: torch.device
        self._pbar: tqdm.tqdm

    def run(self, model: vaelib.BaseVAE, train_data: Dataset, test_data: Dataset) -> None:
        """Main run method."""

        self._make_logdir()
        self._init_logger()
        self._init_writer()

        try:
            self._load_dataloader(train_data, test_data)
            self._run_body(model)
        except Exception as e:
            self._logger.exception(f"Run function error: {e}")
        finally:
            self._quit()

    def _make_logdir(self) -> None:

        self._logdir = pathlib.Path(self._config.logdir, time.strftime("%Y%m%d%H%M"))
        self._logdir.mkdir(parents=True, exist_ok=True)

    def _init_logger(self) -> None:

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh_fmt = logging.Formatter(
            "%(asctime)s - %(module)s.%(funcName)s " "- %(levelname)s : %(message)s"
        )
        sh.setFormatter(sh_fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(filename=self._logdir / "training.log")
        fh.setLevel(logging.DEBUG)
        fh_fmt = logging.Formatter(
            "%(asctime)s - %(module)s.%(funcName)s " "- %(levelname)s : %(message)s"
        )
        fh.setFormatter(fh_fmt)
        logger.addHandler(fh)

        self._logger = logger

    def _init_writer(self) -> None:

        self._writer = tb.SummaryWriter(str(self._logdir))

    def _load_dataloader(self, train_data: Dataset, test_data: Dataset) -> None:

        self._logger.info("Load dataset")

        if torch.cuda.is_available():
            kwargs = {"num_workers": 0, "pin_memory": True}
        else:
            kwargs = {}

        self._train_loader = dataloader.DataLoader(
            train_data,
            shuffle=True,
            batch_size=self._config.batch_size,
            **kwargs,
        )

        self._test_loader = dataloader.DataLoader(
            test_data,
            shuffle=False,
            batch_size=self._config.batch_size,
            **kwargs,
        )

        self._logger.info(f"Train dataset size: {len(self._train_loader)}")
        self._logger.info(f"Test dataset size: {len(self._test_loader)}")

    def _run_body(self, model: vaelib.BaseVAE) -> None:

        self._logger.info("Start experiment")
        self._logger.info(f"Logdir: {self._logdir}")
        self._logger.info(f"Params: {self._config}")

        if self._config.gpus:
            self._device = torch.device(f"cuda:{self._config.gpus}")
        else:
            self._device = torch.device("cpu")

        self._model = model.to(self._device)
        adv_params = self._model.adversarial_parameters()
        if adv_params is not None:
            self._optimizer = optim.Adam(self._model.model_parameters())
            self._adv_optimizer = optim.Adam(adv_params)
        else:
            self._optimizer = optim.Adam(self._model.parameters())
            self._adv_optimizer = None

        self._beta_anneler = vaelib.LinearAnnealer(**self._config.beta_annealer_params)

        self._pbar = tqdm.tqdm(total=self._config.max_steps)
        self._global_steps = 0
        self._postfix = {"train/loss": 0.0, "test/loss": 0.0}

        while self._global_steps < self._config.max_steps:
            self._train()

        self._pbar.close()
        self._logger.info("Finish experiment")

    def _train(self) -> None:

        for data, _ in self._train_loader:
            self._model.train()
            data = data.to(self._device)
            self._beta = next(self._beta_anneler)

            self._optimizer.zero_grad()
            loss_dict = self._model(data, beta=self._beta)
            loss = loss_dict["loss"].mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.max_grad_norm)
            torch.nn.utils.clip_grad_value_(self._model.parameters(), self._config.max_grad_value)
            self._optimizer.step()

            if self._adv_optimizer is not None:
                self._adv_optimizer.zero_grad()
                loss_dict = self._model(data)
                loss_d = loss_dict["loss_d"].mean()

                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), self._config.max_grad_norm
                )
                torch.nn.utils.clip_grad_value_(
                    self._model.parameters(), self._config.max_grad_value
                )
                self._adv_optimizer.step()

            self._global_steps += 1
            self._pbar.update(1)

            self._postfix["train/loss"] = loss.item()
            self._pbar.set_postfix(self._postfix)

            for key, value in loss_dict.items():
                self._writer.add_scalar(f"train/{key}", value.mean(), self._global_steps)

            if self._global_steps % self._config.test_interval == 0:
                self._test()

            if self._global_steps % self._config.save_interval == 0:
                self._save_checkpoint()

                loss_logger = {k: v.mean() for k, v in loss_dict.items()}
                self._logger.debug(f"Train loss (steps={self._global_steps}): " f"{loss_logger}")

                self._save_plots()

            if self._global_steps >= self._config.max_steps:
                break

    def _test(self) -> None:

        loss_logger: DefaultDict[str, float] = collections.defaultdict(float)
        self._model.eval()
        for data, _ in self._test_loader:
            with torch.no_grad():
                data = data.to(self._device)
                loss_dict = self._model(data, beta=self._beta)
                loss = loss_dict["loss"]

            self._postfix["test/loss"] = loss.mean().item()
            self._pbar.set_postfix(self._postfix)

            for key, value in loss_dict.items():
                loss_logger[key] += value.sum().item()

        for key, value in loss_logger.items():
            self._writer.add_scalar(
                f"test/{key}",
                value / (len(self._test_loader)),
                self._global_steps,
            )

        self._logger.debug(f"Test loss (steps={self._global_steps}): {loss_logger}")

    def _save_checkpoint(self) -> None:

        self._logger.debug("Save trained model")

        # Remove unused prefix
        model_state_dict = {}
        for k, v in self._model.state_dict().items():
            model_state_dict[k.replace("module.", "")] = v

        optimizer_state_dict = {}
        for k, v in self._optimizer.state_dict().items():
            optimizer_state_dict[k.replace("module.", "")] = v

        state_dict = {
            "steps": self._global_steps,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }
        path = self._logdir / f"checkpoint_{self._global_steps}.pt"
        torch.save(state_dict, path)

    def _save_configs(self) -> None:

        self._logger.debug("Save configs")
        config = dataclasses.asdict(self._config)
        config["logdir"] = str(self._logdir)

        with (self._logdir / "config.json").open("w") as f:
            json.dump(config, f)

    def _save_plots(self) -> None:
        def gridshow(img: Tensor) -> None:
            if img.dim() == 5 and img.size(1) == 1:
                img = img.squeeze(1)
            elif img.dim() != 4:
                raise ValueError(f"Wrong image size: {img.size()}")

            grid = make_grid(img)
            npgrid = grid.permute(1, 2, 0).numpy()
            plt.imshow(npgrid, interpolation="nearest")

        with torch.no_grad():
            x, _ = next(iter(self._test_loader))
            x = x[:16].to(self._device)
            recon = self._model.reconstruct(x)
            sample = self._model.sample(16)

            x = x.cpu()
            recon = recon.cpu()
            sample = sample.cpu()

        plt.figure(figsize=(20, 12))

        plt.subplot(311)
        gridshow(x)
        plt.title("Original")

        plt.subplot(312)
        gridshow(recon)
        plt.title("Reconstructed")

        plt.subplot(313)
        gridshow(sample)
        plt.title("Sampled")

        plt.tight_layout()
        plt.savefig(self._logdir / f"fig_{self._global_steps}.png")
        plt.close()

    def _quit(self) -> None:

        self._save_configs()
        self._writer.close()
