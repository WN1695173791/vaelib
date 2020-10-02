from setuptools import setup, find_packages


install_requires = [
    "torch>=1.6",
]


extras_require = {
    "testing": [
        "pytest",
        "black",
        "flake8",
        "mypy",
    ],
    "example": [
        "torchvision>=0.7",
        "matplotlib>=3.2",
        "tqdm>=4.47",
        "tensorboardX>=2.1",
    ],
}


setup(
    name="vaelib",
    version="0.1",
    description="VAE models in PyTorch",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
