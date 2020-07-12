
import unittest

import torch

import vaelib


class TestBetaVAE(unittest.TestCase):

    def setUp(self):
        self.model = vaelib.BetaVAE()

    def test_inference(self):
        x = torch.rand(10, 3, 64, 64)
        (recon, z), loss_dict = self.model.inference(x)

        self.assertTupleEqual(recon.size(), x.size())
        self.assertTupleEqual(z.size(), (10, 10))

        self.assertSetEqual(
            set(loss_dict.keys()), set(["loss", "kl_loss", "ce_loss"]))
        self.assertTupleEqual(loss_dict["loss"].size(), (10,))
        self.assertTupleEqual(loss_dict["kl_loss"].size(), (10,))
        self.assertTupleEqual(loss_dict["ce_loss"].size(), (10,))

    def test_sample(self):
        x = self.model.sample(4)

        self.assertTupleEqual(x.size(), (4, 3, 64, 64))
        self.assertTrue((x >= 0).all() and (x <= 1).all())


if __name__ == "__main__":
    unittest.main()
