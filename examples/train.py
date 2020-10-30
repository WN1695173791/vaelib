import argparse
import json
import os
import pathlib
import random

import torch
from torchvision import transforms, datasets

import vaelib


def main() -> None:

    args = init_args()

    config_path = pathlib.Path(os.getenv("CONFIG_PATH", "./examples/config.json"))
    with config_path.open() as f:
        config = json.load(f)

    logdir = str(pathlib.Path(os.getenv("LOGDIR", "./logs/"), os.getenv("EXPERIMENT_NAME", "tmp")))
    dataset_name = os.getenv("DATASET_NAME", "mnist")
    data_dir = pathlib.Path(os.getenv("DATASET_DIR", "./data/"), dataset_name)

    params = vars(args)
    args_cuda = params.pop("cuda")
    args_seed = params.pop("seed")
    args_model = params.pop("model")

    use_cuda = torch.cuda.is_available() and args_cuda != "null"
    gpus = args_cuda if use_cuda else ""

    params.update(
        {
            "logdir": str(logdir),
            "gpus": gpus,
            "beta_annealer_params": config["beta_annealer_params"],
        }
    )

    torch.manual_seed(args_seed)
    random.seed(args_seed)

    model_dict = {
        "betavae": vaelib.BetaVAE,
        "avb": vaelib.AVB,
        "nvae": vaelib.NouveauVAE,
    }
    model = model_dict[args_model](**config[f"{args_model}_params"])

    if args_model == "nvae":
        _transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    else:
        _transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=_transform)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=_transform)

    trainer = vaelib.Trainer(**params)
    trainer.run(model, train_data, test_data)


def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML training")
    parser.add_argument(
        "--cuda",
        type=str,
        default="0",
        help="Index of CUDA device with comma separation: ex. '0,1'. 'null' means cpu device.",
    )
    parser.add_argument("--model", type=str, default="betavae", help="Model name.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--max-steps", type=int, default=2, help="Number of gradient steps.")
    parser.add_argument("--max-grad-value", type=float, default=5.0, help="Clipping value.")
    parser.add_argument("--max-grad-norm", type=float, default=100.0, help="Clipping norm.")
    parser.add_argument("--test-interval", type=int, default=2, help="Interval steps for testing.")
    parser.add_argument("--save-interval", type=int, default=2, help="Interval steps for saving.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
