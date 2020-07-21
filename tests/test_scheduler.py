
import unittest

import vaelib


class TestAnnealer(unittest.TestCase):

    def test_scheduling(self):
        annealer = vaelib.LinearAnnealer(0, 1, 10)

        for i in range(20):
            value = next(annealer)
            self.assertEqual(value, min((i + 1) / 10, 1))


if __name__ == "__main__":
    unittest.main()
