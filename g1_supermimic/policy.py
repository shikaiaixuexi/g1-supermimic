"""Goal-conditioned actor with motion/velocity encoders and a shared decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .networks import RunningNorm, mlp


@dataclass(frozen=True)
class PolicySpec:
    joints: int = 29
    history: int = 10
    future_frames: int = 10
    reference_frame_dim: int = 68
    latent_dim: int = 32
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    velocity_hidden: tuple[int, ...] = (256, 256)

    @property
    def frame_dim(self):
        return 3 * self.joints + 6

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ("encoder_hidden", "decoder_hidden", "velocity_hidden"):
            data[key] = tuple(data[key])
        return cls(**data)


def reset_history(frame, length=10):
    """Repeat the first available observation, oldest to newest; no old episode leak."""
    return frame[:, None].expand(-1, length, -1).clone()


def push_history(history, frame, done=None):
    if history.ndim != 3 or frame.shape != (history.shape[0], history.shape[2]):
        raise ValueError("Expected history [B,H,F] and frame [B,F]")
    out = torch.cat((history[:, 1:], frame[:, None]), 1)
    if done is not None:
        out = torch.where(
            done[:, None, None], reset_history(frame, history.shape[1]), out
        )
    return out


class Actor(nn.Module):
    def __init__(self, spec: PolicySpec | None = None):
        super().__init__()
        self.spec = spec or PolicySpec()
        s = self.spec
        if s.history != 10 or s.future_frames != 10 or s.latent_dim != 32:
            raise ValueError(
                "Paper interface requires 10 history frames, 10 future frames and latent 32"
            )
        if s.joints < 1 or s.reference_frame_dim < 1:
            raise ValueError("Joint and reference dimensions must be positive")
        self.norm = RunningNorm(s.frame_dim)
        self.motion_norm = RunningNorm(s.reference_frame_dim)
        self.command_norm = RunningNorm(3)
        self.motion_encoder = mlp(
            s.future_frames * s.reference_frame_dim,
            s.encoder_hidden,
            s.latent_dim,
            act=nn.ELU,
        )
        self.velocity_encoder = mlp(
            3 + s.history * s.frame_dim, s.velocity_hidden, s.latent_dim, act=nn.ELU
        )
        self.decoder = mlp(
            s.latent_dim + s.history * s.frame_dim,
            s.decoder_hidden,
            s.joints,
            act=nn.ELU,
        )

    def _history(self, history):
        expected = (self.spec.history, self.spec.frame_dim)
        if history.ndim != 3 or history.shape[1:] != expected:
            raise ValueError(
                f"Expected history [B,{expected[0]},{expected[1]}], got {tuple(history.shape)}"
            )
        return self.norm(history).flatten(1)

    def encode(self, history, goal, interface="velocity"):
        hist = self._history(history)
        if interface == "motion":
            expected = (
                history.shape[0],
                self.spec.future_frames,
                self.spec.reference_frame_dim,
            )
            if goal.shape != expected:
                raise ValueError(
                    f"Expected future reference {expected}, got {tuple(goal.shape)}"
                )
            z = self.motion_encoder(self.motion_norm(goal).flatten(1))
        elif interface == "velocity":
            if goal.shape != (history.shape[0], 3):
                raise ValueError("Velocity goal must be [B,3]")
            z = self.velocity_encoder(torch.cat((self.command_norm(goal), hist), -1))
        else:
            raise ValueError(f"Unknown policy interface: {interface}")
        return torch.tanh(z)

    def forward(self, history, goal, interface="velocity"):
        z = self.encode(history, goal, interface)
        return self.decoder(torch.cat((z, self._history(history)), -1))

    @torch.no_grad()
    def update_stats(self, frame, goal, interface="velocity"):
        self.norm.update(frame)
        if interface == "velocity":
            self.command_norm.update(goal)
        elif interface == "motion":
            self.motion_norm.update(goal)
        else:
            raise ValueError(f"Unknown policy interface: {interface}")

    def train_interface(self, interface, train_decoder=True):
        """Select an encoder; optional decoder freeze supports downstream fine-tuning.

        Freezing parameters still propagates gradients through decoder inputs.
        Joint motion/velocity training and latent alignment require reference data.
        """
        self.motion_encoder.requires_grad_(interface == "motion")
        self.velocity_encoder.requires_grad_(interface == "velocity")
        self.decoder.requires_grad_(train_decoder)
        if interface not in ("motion", "velocity"):
            raise ValueError(f"Unknown policy interface: {interface}")
