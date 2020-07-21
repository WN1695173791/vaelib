
import unittest

import torch

import vaelib


class TestNouveauVAE(unittest.TestCase):

    def setUp(self):
        self.model = vaelib.NouveauVAE()

    def test_inference(self):
        x = torch.rand(10, 3, 32, 32)
        (recon,), loss_dict = self.model.inference(x)

        self.assertTupleEqual(recon.size(), x.size())
        self.assertFalse(torch.isnan(recon).any())

        self.assertSetEqual(
            set(loss_dict.keys()),
            set(["loss", "bit_loss", "kl_loss", "nll_loss", "sr_loss"]))

        keys = ["loss", "bit_loss", "kl_loss", "nll_loss", "sr_loss"]
        for k in keys:
            self.assertTupleEqual(loss_dict[k].size(), (10,))

        for k in keys:
            self.assertFalse(torch.isnan(loss_dict[k]).any())

    def test_sample(self):
        x = self.model.sample(4)

        self.assertTupleEqual(x.size(), (4, 3, 32, 32))
        self.assertTrue((x >= 0).all() and (x <= 1).all())

    def test_other_setting(self):
        x = torch.rand(10, 3, 64, 64)
        model = vaelib.NouveauVAE(num_groups=[5, 10, 20], enc_channels=64)
        (recon,), _ = model.inference(x)

        self.assertTupleEqual(recon.size(), x.size())


if __name__ == "__main__":
    unittest.main()
