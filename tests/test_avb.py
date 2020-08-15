
import unittest

import torch

import vaelib


class TestAVB(unittest.TestCase):

    def setUp(self):
        self.model = vaelib.AVB()

    def test_inference(self):
        x = torch.rand(10, 3, 64, 64)
        (recon, z), loss_dict = self.model.inference(x)

        self.assertTupleEqual(recon.size(), x.size())
        self.assertTupleEqual(z.size(), (10, 10))

        self.assertSetEqual(
            set(loss_dict.keys()),
            set(["loss", "ce_loss", "logits", "loss_d"]))
        self.assertTupleEqual(loss_dict["loss"].size(), (10,))
        self.assertTupleEqual(loss_dict["ce_loss"].size(), (10,))
        self.assertTupleEqual(loss_dict["logits"].size(), (10,))
        self.assertTupleEqual(loss_dict["loss_d"].size(), (10,))

    def test_sample(self):
        x = self.model.sample(4)

        self.assertTupleEqual(x.size(), (4, 3, 64, 64))
        self.assertTrue((x >= 0).all() and (x <= 1).all())


if __name__ == "__main__":
    unittest.main()
