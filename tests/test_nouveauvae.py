import torch
import vaelib


def test_inference() -> None:
    model = vaelib.NouveauVAE()
    x = torch.rand(10, 3, 32, 32)
    (recon,), loss_dict = model.inference(x)

    assert recon.size() == x.size()
    assert not torch.isnan(recon).any()
    assert (recon >= 0).all() and (recon <= 1).all()

    assert set(loss_dict.keys()) == set(["loss", "bit_loss", "kl_loss", "nll_loss", "sr_loss"])

    keys = ["loss", "bit_loss", "kl_loss", "nll_loss", "sr_loss"]
    for k in keys:
        assert loss_dict[k].size() == (10,)

    for k in keys:
        assert not torch.isnan(loss_dict[k]).any()


def test_sample() -> None:
    model = vaelib.NouveauVAE()
    x = model.sample(4)

    assert x.size() == (4, 3, 32, 32)
    assert (x >= 0).all() and (x <= 1).all()


def test_other_setting() -> None:
    model = vaelib.NouveauVAE()
    x = torch.rand(10, 3, 64, 64)
    model = vaelib.NouveauVAE(num_groups=[5, 10, 20], enc_channels=64)
    (recon,), _ = model.inference(x)

    assert recon.size() == x.size()
