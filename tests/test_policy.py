"""Architecture and temporal contracts for the paper-aligned policy interface."""

import pytest
import torch
from torch import nn

from g1_supermimic.policy import Actor, PolicySpec, reset_history, push_history
from g1_supermimic.learning import Replay
from g1_supermimic.task import observation


def small_spec(**kwargs):
    return PolicySpec(
        encoder_hidden=(32, 32),
        decoder_hidden=(32, 32),
        velocity_hidden=(32, 32),
        **kwargs,
    )


def test_paper_network_widths_and_latent():
    actor = Actor()

    def widths(net):
        return [m.out_features for m in net if isinstance(m, nn.Linear)]

    assert widths(actor.motion_encoder) == [2048, 1024, 512, 256, 32]
    assert widths(actor.decoder) == [2048, 1024, 512, 256, 29]
    assert widths(actor.velocity_encoder) == [256, 256, 32]
    assert actor.motion_encoder[0].in_features == 10 * 68
    assert actor.decoder[0].in_features == 32 + 10 * 93
    assert actor.velocity_encoder[0].in_features == 3 + 10 * 93


def test_history_is_chronological_and_resets_per_env():
    hist = torch.arange(10.0).reshape(1, 10, 1).repeat(2, 1, 1)
    out = push_history(
        hist, torch.tensor([[10.0], [50.0]]), torch.tensor([False, True])
    )
    torch.testing.assert_close(out[0, :, 0], torch.arange(1.0, 11.0))
    torch.testing.assert_close(out[1, :, 0], torch.full((10,), 50.0))
    assert hist[0, 0, 0] == 0  # No in-place mutation of replay seeds.


def test_proprioception_order_and_noise_bounds():
    s = torch.arange(68.0).expand(64, -1)
    previous = torch.arange(29.0).expand(64, -1)
    clean = observation(s, previous)
    assert clean.shape == (64, 93)
    torch.testing.assert_close(clean[:, :58], s[:, 10:])
    torch.testing.assert_close(clean[:, 58:61], s[:, 7:10])
    torch.testing.assert_close(clean[:, 61:64], s[:, 1:4])
    diff = (observation(s, previous, noise=True) - clean).abs()
    assert diff[:, :29].max() <= 0.01001
    assert diff[:, 29:58].max() <= 0.50001
    assert diff[:, 58:61].max() <= 0.20001
    assert diff[:, 61:64].max() <= 0.05001
    assert diff[:, 64:].max() == 0


def test_both_goals_drive_shared_decoder_and_bound_latent():
    actor = Actor(small_spec())
    hist = torch.randn(3, 10, 93, requires_grad=True)
    future = torch.randn(3, 10, 68, requires_grad=True)
    z = actor.encode(hist, future, "motion")
    assert z.shape == (3, 32) and z.abs().max() <= 1
    motion = actor(hist, future, "motion")
    motion.square().mean().backward()
    assert future.grad.abs().sum() > 0 and hist.grad.abs().sum() > 0
    assert actor.motion_encoder[0].weight.grad.abs().sum() > 0
    assert actor.velocity_encoder[0].weight.grad is None
    assert motion.shape == actor(hist.detach(), torch.randn(3, 3)).shape == (3, 29)
    with pytest.raises(ValueError, match="future reference"):
        actor(hist, future[:, :1], "motion")
    with pytest.raises(ValueError, match="history"):
        actor(hist[:, :1], torch.randn(3, 3))


def test_frozen_decoder_transmits_gradient_only_to_selected_encoder():
    actor = Actor(small_spec())
    actor.train_interface("velocity", train_decoder=False)
    hist = torch.randn(3, 10, 93)
    actor(hist, torch.randn(3, 3)).square().mean().backward()
    assert all(p.grad is None for p in actor.decoder.parameters())
    assert all(p.grad is None for p in actor.motion_encoder.parameters())
    assert sum(p.grad.abs().sum() for p in actor.velocity_encoder.parameters()) > 0


def test_replay_preserves_full_seed_history():
    b = Replay(1, 12, 68, 29, "cpu", history_shape=(10, 93))
    for i in range(12):
        frame = torch.full((1, 93), float(i))
        b.add(
            torch.full((1, 68), float(i)),
            torch.zeros(1, 68),
            torch.zeros(1, 29),
            torch.zeros(1, 29),
            torch.zeros(1, 3),
            torch.zeros(1, dtype=torch.bool),
            history=reset_history(frame),
        )
    seeds = b.seeds(50)
    torch.testing.assert_close(seeds["history"][:, -1, 0].float(), seeds["s"][:, 0])
    assert "history" not in b.windows(2, 8)


def test_checkpoint_spec_roundtrip_and_old_interface_rejected():
    actor = Actor(small_spec())
    restored = Actor(PolicySpec.from_dict(actor.spec.to_dict()))
    restored.load_state_dict(actor.state_dict())
    h, g = torch.randn(2, 10, 93), torch.randn(2, 3)
    torch.testing.assert_close(actor(h, g), restored(h, g))
    with pytest.raises(ValueError, match="history"):
        restored(torch.randn(2, 96), g)


def test_old_checkpoint_is_rejected_before_environment_creation(tmp_path):
    from g1_supermimic.train import TrainCfg, Trainer

    path = tmp_path / "old.pt"
    torch.save({"format": "g1-supermimic-v1"}, path)
    with pytest.raises(ValueError, match="earlier policy interface"):
        Trainer(TrainCfg(resume=str(path)))
