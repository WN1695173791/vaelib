
import unittest

import torch

import vaelib


class TestNouveauVAE(unittest.TestCase):

    def setUp(self):
        self.model = vaelib.NouveauVAE()

    def test_inference(self):
        x = torch.rand(10, 3, 32, 32) - 0.5
        (recon,), loss_dict = self.model.inference(x)

        self.assertTupleEqual(recon.size(), x.size())

        self.assertSetEqual(
            set(loss_dict.keys()),
            set(["loss", "bit_loss", "kl_loss", "nll_loss", "sr_loss"]))
        self.assertTupleEqual(loss_dict["loss"].size(), (10,))
        self.assertTupleEqual(loss_dict["bit_loss"].size(), (10,))
        self.assertTupleEqual(loss_dict["kl_loss"].size(), (10,))
        self.assertTupleEqual(loss_dict["nll_loss"].size(), (10,))
        self.assertTupleEqual(loss_dict["sr_loss"].size(), (10,))

    def test_sample(self):
        x = self.model.sample(4)

        self.assertTupleEqual(x.size(), (4, 3, 32, 32))
        self.assertTrue((x >= -0.5).all() and (x <= 0.5).all())

    def test_other_setting(self):
        x = torch.rand(10, 3, 64, 64) - 0.5
        model = vaelib.NouveauVAE(num_groups=[5, 10, 20], enc_channels=64)
        (recon,), loss_dict = model.inference(x)

        self.assertTupleEqual(recon.size(), x.size())


if __name__ == "__main__":
    unittest.main()
