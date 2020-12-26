from setuptools import find_packages, setup

install_requires = [
    "torch>=1.7",
    "torchvision>=0.8",
]


extras_require = {
    "training": [
        "matplotlib>=3.2",
        "tqdm>=4.47",
        "tensorboardX>=2.1",
    ],
    "dev": [
        "pytest",
        "black",
        "flake8",
        "mypy==790",
        "isort",
    ],
}


setup(
    name="vaelib",
    version="0.2",
    description="VAE models in PyTorch",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
