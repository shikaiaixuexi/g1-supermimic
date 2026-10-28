"""Learning contracts independent of the simulator; integration checks use CUDA."""

import torch

from g1_supermimic.learning import Replay, WorldModel, td_lambda
from g1_supermimic.task import observation, reward
from g1_supermimic.policy import Actor, PolicySpec, reset_history


def test_replay_masks_reset_transitions_and_wraps():
    b = Replay(1, 8, 2, 1, "cpu")
    for i in range(12):
        b.add(
            torch.full((1, 2), float(i)),
            torch.full((1, 2), float(i + 1)),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 3),
            torch.tensor([i == 7]),
        )
    w = b.windows(200, 4)
    assert torch.all(w["s"][:, 1:, 0] - w["s"][:, :-1, 0] == 1)
    expected = (w["s"][..., 0] != 7).float().cumprod(1)
    torch.testing.assert_close(w["valid"], expected)


def test_td_lambda_termination_has_no_bootstrap():
    class FixedCritic:
        def __call__(self, x):
            return torch.full(x.shape[:2], 100.0)

    r = torch.tensor([[1.0, 2.0, 3.0]])
    y = td_lambda(
        FixedCritic(),
        torch.zeros(1, 4, 2),
        r,
        torch.tensor([[1.0, 1.0, 0.0]]),
        gamma=1,
        lam=1,
    )
    torch.testing.assert_close(y, torch.tensor([[6.0, 5.0, 3.0]]))


def test_frozen_dynamics_still_transmits_actor_gradient():
    n, nj, sd = 8, 29, 68
    s = torch.randn(n, sd) * 0.01
    s[:, 0], s[:, 3] = 0.76, -1
    prev, cmd = torch.zeros(n, nj), torch.zeros(n, 3)
    actor = Actor(
        PolicySpec(
            encoder_hidden=(32, 32), decoder_hidden=(32, 32), velocity_hidden=(32, 32)
        )
    )
    wm = WorldModel(sd, nj, width=32).requires_grad_(False)
    actor.update_stats(observation(s, prev), cmd)
    limits = torch.stack((-torch.ones(1, nj), torch.ones(1, nj)), -1)
    a = actor(reset_history(observation(s, prev)), cmd)
    nxt = wm(s, a, prev)
    loss = -reward(nxt, a, prev, cmd, torch.zeros(1, nj), limits).mean()
    loss.backward()
    assert all(p.grad is None for p in wm.parameters())
    assert (
        sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None)
        > 0
    )


def test_multistep_model_fit_reduces_heldout_error():
    torch.manual_seed(4)
    model = WorldModel(2, 1, width=32)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)

    def windows(batch):
        a = torch.randn(batch, 4, 1)
        s = torch.randn(batch, 2)
        states, targets = [], []
        for j in range(4):
            states.append(s)
            s = s + torch.cat((0.1 * a[:, j], -0.05 * a[:, j]), -1)
            targets.append(s)
        return {
            "s": torch.stack(states, 1),
            "next": torch.stack(targets, 1),
            "a": a,
            "previous": torch.zeros_like(a),
            "valid": torch.ones(batch, 4),
        }

    train, test = windows(64), windows(32)
    model.update_stats(
        train["s"].reshape(-1, 2),
        train["a"].reshape(-1, 1),
        train["previous"].reshape(-1, 1),
        train["next"].reshape(-1, 2),
    )
    before = float(model.loss(test).detach())
    for _ in range(80):
        opt.zero_grad()
        model.loss(train).backward()
        opt.step()
    assert float(model.loss(test).detach()) < before * 0.2
