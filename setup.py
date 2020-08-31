
from setuptools import setup, find_packages


install_requires = [
    "torch>=1.6.0",
]


extras_require = {
    "dev": ["pytest"]
}


setup(
    name="vaelib",
    version="0.1",
    description="VAE models by PyTorch",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
