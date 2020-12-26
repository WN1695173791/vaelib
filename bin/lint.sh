set -x

isort .
mypy .
flake8 .
black .
