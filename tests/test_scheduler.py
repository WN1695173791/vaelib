import vaelib


def test_scheduling() -> None:
    annealer = vaelib.LinearAnnealer(0, 1, 10)

    for i in range(20):
        value = next(annealer)
        assert value == min((i + 1) / 10, 1)
