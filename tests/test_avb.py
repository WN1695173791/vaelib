import torch
import vaelib


def test_inference() -> None:
    model = vaelib.AVB()
    x = torch.rand(10, 3, 64, 64)
    (recon, z), loss_dict = model.inference(x)

    assert recon.size() == x.size()
    assert z.size() == (10, 10)

    assert set(loss_dict.keys()) == set(
        ["loss", "ce_loss", "logits", "loss_d"]
    )
    assert loss_dict["loss"].size() == (10,)
    assert loss_dict["ce_loss"].size() == (10,)
    assert loss_dict["logits"].size() == (10,)
    assert loss_dict["loss_d"].size() == (10,)


def test_sample() -> None:
    model = vaelib.AVB()
    x = model.sample(4)

    assert x.size() == (4, 3, 64, 64)
    assert (x >= 0).all() and (x <= 1).all()
