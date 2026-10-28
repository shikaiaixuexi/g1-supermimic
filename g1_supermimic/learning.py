"""Dimension-independent replay and normalized residual dynamics for G1."""

from __future__ import annotations

import torch
from torch import nn

from .networks import RunningNorm, mlp


class Critic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RunningNorm(dim)
        self.net = mlp(dim, (256, 128), 1, act=nn.ELU)

    def forward(self, x):
        return self.net(self.norm(x)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, state_dim, act_dim, width=2048, project_rotation=False):
        super().__init__()
        self.project_rotation = project_rotation
        self.in_norm = RunningNorm(state_dim + 2 * act_dim)
        self.out_norm = RunningNorm(state_dim)
        self.net = mlp(state_dim + 2 * act_dim, (width,) * 3, state_dim)

    def forward(self, s, a, previous):
        nxt = s + self.out_norm.unnormalize(
            self.net(self.in_norm(torch.cat((s, a, previous), -1)))
        )
        if self.project_rotation:
            from .kinematics import matrix6d, rotation6d

            nxt = torch.cat(
                (nxt[:, :3], rotation6d(matrix6d(nxt[:, 3:9])), nxt[:, 9:]), -1
            )
        return nxt

    @torch.no_grad()
    def update_stats(self, s, a, previous, nxt):
        self.in_norm.update(torch.cat((s, a, previous), -1))
        self.out_norm.update(nxt - s)

    def loss(self, w):
        s, total = w["s"][:, 0], 0.0
        scale = self.out_norm.var.sqrt() + 1e-6
        for j in range(w["a"].shape[1]):
            s = self(s, w["a"][:, j], w["previous"][:, j])
            err = ((s - w["next"][:, j]) / scale).square().mean(-1)
            total = total + (err * w["valid"][:, j]).sum()
        return total / w["valid"].sum().clamp(min=1)


class Replay:
    def __init__(
        self,
        n,
        length,
        state_dim,
        act_dim,
        device,
        history_shape=None,
        command_dim=3,
        reference_indices=False,
    ):
        self.n, self.length, self.device = n, length, device
        self.data = {
            k: torch.zeros(n, length, d, device=device)
            for k, d in (
                ("s", state_dim),
                ("next", state_dim),
                ("a", act_dim),
                ("previous", act_dim),
                ("command", command_dim),
            )
        }
        if history_shape is not None:
            self.data["history"] = torch.zeros(
                n, length, *history_shape, device=device, dtype=torch.float16
            )
        if reference_indices:
            self.data["reference_index"] = torch.zeros(
                n, length, device=device, dtype=torch.long
            )
        self.done = torch.ones(n, length, device=device, dtype=torch.bool)
        self.ptr = self.size = 0

    def add(
        self, s, nxt, a, previous, command, done, history=None, reference_index=None
    ):
        for k, v in (
            ("s", s),
            ("next", nxt),
            ("a", a),
            ("previous", previous),
            ("command", command),
        ):
            self.data[k][:, self.ptr] = v
        if "history" in self.data:
            if (
                history is None
                or history.shape != self.data["history"][:, self.ptr].shape
            ):
                raise ValueError("Replay history shape mismatch")
            self.data["history"][:, self.ptr] = history.detach().half()
        if "reference_index" in self.data:
            if reference_index is None or reference_index.shape != (self.n,):
                raise ValueError("Motion replay requires reference frame indices")
            self.data["reference_index"][:, self.ptr] = reference_index
        self.done[:, self.ptr] = done
        self.ptr = (self.ptr + 1) % self.length
        self.size = min(self.size + 1, self.length)

    def windows(self, batch, k):
        e = torch.randint(self.n, (batch,), device=self.device)
        t = torch.randint(self.size - k + 1, (batch,), device=self.device)
        idx = (
            self.ptr - self.size + t[:, None] + torch.arange(k, device=self.device)
        ) % self.length
        w = {
            name: x[e[:, None], idx]
            for name, x in self.data.items()
            if name != "history"
        }
        # Auto-reset returns a reset state; mask that transition and the following window.
        w["valid"] = (~self.done[e[:, None], idx]).float().cumprod(1)
        return w

    def seeds(self, batch):
        e = torch.randint(self.n, (batch,), device=self.device)
        t = torch.randint(self.size, (batch,), device=self.device)
        idx = (self.ptr - self.size + t) % self.length
        return {name: x[e, idx].clone() for name, x in self.data.items()}


@torch.no_grad()
def td_lambda(critic, inputs, rewards, continuation, gamma=0.99, lam=0.95):
    values = critic(inputs)
    nxt = values[:, -1]
    targets = torch.empty_like(rewards)
    for t in reversed(range(rewards.shape[1])):
        nxt = rewards[:, t] + gamma * continuation[:, t] * (
            (1 - lam) * values[:, t + 1] + lam * nxt
        )
        targets[:, t] = nxt
    return targets
