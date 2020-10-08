"""Trainer class."""

from typing import Dict, DefaultDict, Union, Optional

import collections
import dataclasses
import json
import logging
import pathlib
import time

import matplotlib.pyplot as plt
import tqdm

import torch
from torch import Tensor, optim
from torch.optim import optimizer
from torch.utils.data import dataloader

from torchvision import datasets, transforms
from torchvision.utils import make_grid

import tensorboardX as tb

import vaelib


@dataclasses.dataclass
class Config:
    # From kwargs
    cuda: str
    model: str
    seed: int
    batch_size: int
    max_steps: int
    test_interval: int
    save_interval: int

    # From config
    betavae_params: dict
    avb_params: dict
    nvae_params: dict
    optimizer_params: dict
    adv_optimizer_params: dict
    beta_annealer_params: dict
    max_grad_value: float
    max_grad_norm: float

    # From params
    logdir: Union[str, pathlib.Path]
    gpus: Optional[str]
    data_dir: Union[str, pathlib.Path]


class Trainer:
    """Trainer class for ML models.

    Args:
        model (vaelib.BaseVAE): ML model.
        config (dict): Dictionary of hyper-parameters.
    """

    def __init__(self, model: vaelib.BaseVAE, config: dict):

        self._model = model
        self._config = Config(**config)
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

    def run(self) -> None:
        """Main run method."""

        self._check_logdir()
        self._init_logger()
        self._init_writer()

        try:
            self._base_run()
        except Exception as e:
            self._logger.exception(f"Run function error: {e}")
        finally:
            self._quit()

    def _base_run(self) -> None:
        """Base running method."""

        self._logger.info("Start experiment")
        self._logger.info(f"Logdir: {self._logdir}")
        self._logger.info(f"Params: {self._config}")

        if self._config.gpus:
            self._device = torch.device(f"cuda:{self._config.gpus}")
        else:
            self._device = torch.device("cpu")

        self._load_dataloader()
        self._model = self._model.to(self._device)
        adv_params = self._model.adversarial_parameters()
        if adv_params is not None:
            self._optimizer = optim.Adam(
                self._model.model_parameters(), **self._config.optimizer_params
            )
            self._adv_optimizer = optim.Adam(adv_params, **self._config.adv_optimizer_params)
        else:
            self._optimizer = optim.Adam(self._model.parameters(), **self._config.optimizer_params)
            self._adv_optimizer = None

        self._beta_anneler = vaelib.LinearAnnealer(**self._config.beta_annealer_params)

        self._pbar = tqdm.tqdm(total=self._config.max_steps)
        self._global_steps = 0
        self._postfix = {"train/loss": 0.0, "test/loss": 0.0}

        while self._global_steps < self._config.max_steps:
            self._train()

        self._pbar.close()
        self._logger.info("Finish experiment")

    def _check_logdir(self) -> None:
        """Checks log directory.

        This method specifies logdir and make the directory if it does not
        exist.
        """

        self._logdir = pathlib.Path(self._config.logdir, time.strftime("%Y%m%d%H%M"))
        self._logdir.mkdir(parents=True, exist_ok=True)

    def _init_logger(self) -> None:
        """Initalizes logger."""

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
        """Initializes tensorboard writer."""

        self._writer = tb.SummaryWriter(str(self._logdir))

    def _load_dataloader(self) -> None:
        """Loads data loader for training and test."""

        self._logger.info("Load dataset")

        if self._config.model == "nvae":
            _transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        else:
            _transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])

        train_data = datasets.MNIST(
            root=self._config.data_dir,
            train=True,
            download=True,
            transform=_transform,
        )
        test_data = datasets.MNIST(
            root=self._config.data_dir,
            train=False,
            download=True,
            transform=_transform,
        )

        if torch.cuda.is_available():
            kwargs = {"num_workers": 0, "pin_memory": True}
        else:
            kwargs = {}

        self._train_loader = torch.utils.data.DataLoader(
            train_data,
            shuffle=True,
            batch_size=self._config.batch_size,
            **kwargs,
        )

        self._test_loader = torch.utils.data.DataLoader(
            test_data,
            shuffle=False,
            batch_size=self._config.batch_size,
            **kwargs,
        )

        self._logger.info(f"Train dataset size: {len(self._train_loader)}")
        self._logger.info(f"Test dataset size: {len(self._test_loader)}")

    def _train(self) -> None:
        """Trains model."""

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
        """Tests model."""

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
        """Saves trained model and optimizer to checkpoint file.

        Args:
            loss (float): Saved loss value.
        """

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
        """Saves setting including config and args in json format."""

        self._logger.debug("Save configs")

        config = dataclasses.asdict(self._config)
        config["logdir"] = str(self._logdir)

        with (self._logdir / "config.json").open("w") as f:
            json.dump(config, f)

    def _save_plots(self) -> None:
        """Save reconstructed and sampled plots."""

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
        """Post process."""

        self._save_configs()
        self._writer.close()
