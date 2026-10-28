"""Running normalization and MLP building blocks for G1 networks."""

from __future__ import annotations

import torch
from torch import nn


class RunningNorm(nn.Module):
    """Running mean/std over batches (Welford-style merge), frozen when not updated."""

    def __init__(self, dim: int, eps: float = 1e-4, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (
            self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot
        ) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-self.clip, self.clip)

    def unnormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * (self.var.sqrt() + 1e-6) + self.mean


def mlp(
    in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.SiLU
) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)
