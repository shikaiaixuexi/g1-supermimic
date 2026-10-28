"""Differentiable G1 link kinematics and motion-tracking state.

State: root position (3), first two rotation columns (6), world linear/angular
velocity (3+3), absolute joint positions and velocities (29+29): 73 values.
Rigid transforms and link velocities use the actual MuJoCo model constants.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


def quat_matrix(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def rotation6d(r):
    return torch.cat((r[..., :, 0], r[..., :, 1]), -1)


def matrix6d(x):
    a, b = x[..., :3], x[..., 3:6]
    e1 = torch.zeros_like(a)
    e1[..., 0] = 1
    a = torch.where(a.norm(dim=-1, keepdim=True) < 1e-6, e1, a)
    c1 = F.normalize(a, dim=-1, eps=1e-6)
    perpendicular = b - (c1 * b).sum(-1, keepdim=True) * c1
    basis = torch.zeros_like(a)
    basis[..., 1] = 1
    other = torch.zeros_like(a)
    other[..., 2] = 1
    basis = torch.where(c1[..., 1:2].abs() > 0.9, other, basis)
    fallback = basis - (basis * c1).sum(-1, keepdim=True) * c1
    perpendicular = torch.where(
        perpendicular.norm(dim=-1, keepdim=True) < 1e-6, fallback, perpendicular
    )
    c2 = F.normalize(perpendicular, dim=-1, eps=1e-6)
    return torch.stack((c1, c2, torch.cross(c1, c2, dim=-1)), -1)


def apply(r, x):
    return (r @ x.unsqueeze(-1)).squeeze(-1)


def yaw_matrix(r):
    x, y = r[..., 0, 0], r[..., 1, 0]
    # At a vertical forward axis, use the horizontal right axis. Select the
    # inputs before atan2 so even unselected gradients never evaluate atan2(0,0).
    vertical = x.square() + y.square() < 1e-8
    x = torch.where(vertical, r[..., 1, 1], x)
    y = torch.where(vertical, -r[..., 0, 1], y)
    yaw = torch.atan2(y, x)
    c, s = yaw.cos(), yaw.sin()
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(
        *c.shape, 3, 3
    )


def rotation_error_squared(a, b):
    """Squared SO(3) geodesic angle, finite zero-error gradients (unlike acos)."""
    r = a @ b.transpose(-1, -2)
    skew = (
        torch.stack(
            (
                r[..., 2, 1] - r[..., 1, 2],
                r[..., 0, 2] - r[..., 2, 0],
                r[..., 1, 0] - r[..., 0, 1],
            ),
            -1,
        )
        * 0.5
    )
    sine = torch.linalg.vector_norm(skew, dim=-1)
    cosine = ((r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    return torch.atan2(sine, cosine).square()


def read_state(env):
    d = env.scene["robot"].data
    return torch.cat(
        (
            d.root_link_pos_w - env.scene.env_origins,
            rotation6d(quat_matrix(d.root_link_quat_w)),
            d.root_link_lin_vel_w,
            d.root_link_ang_vel_w,
            d.joint_pos,
            d.joint_vel,
        ),
        -1,
    )


def proprioception(s, previous, default_q, noise=False):
    from .task import observation

    n = previous.shape[-1]
    r = matrix6d(s[:, 3:9])
    gravity = -r[:, 2, :]
    angular_b = apply(r.transpose(-1, -2), s[:, 12:15])
    # Reuse Table A.1 ordering and noise from the velocity task; its extra
    # root-height/linear-velocity slots are not observed by the policy.
    low_state = torch.cat(
        (
            s[:, 2:3],
            gravity,
            s[:, 9:12],
            angular_b,
            s[:, 15 : 15 + n] - default_q,
            s[:, 15 + n :],
        ),
        -1,
    )
    return observation(low_state, previous, noise=noise)


class G1Kinematics(nn.Module):
    def __init__(
        self, model, body_ids, joint_ids, body_names, tracked_names=TRACKED_BODIES
    ):
        super().__init__()
        body_ids, joint_ids = list(map(int, body_ids)), list(map(int, joint_ids))
        self.body_names = tuple(body_names)
        self.tracked_names = tuple(tracked_names)
        if self.body_names[0] != "pelvis" or len(joint_ids) != 29:
            raise ValueError("Expected a floating-base, 29-joint G1 rooted at pelvis")
        lookup = {b: i for i, b in enumerate(body_ids)}
        self.parents = [-1] + [
            lookup[int(model.body_parentid[b])] for b in body_ids[1:]
        ]
        jlookup = {j: i for i, j in enumerate(joint_ids)}
        self.joints = []
        axes, pivots = [], []
        for i, bid in enumerate(body_ids):
            adr, num = int(model.body_jntadr[bid]), int(model.body_jntnum[bid])
            if i == 0:
                if num != 1 or int(model.jnt_type[adr]) != 0:
                    raise ValueError("Root must have exactly one free joint")
                jid = -1
            elif num == 0:
                jid = -1
            else:
                if num != 1 or int(model.jnt_type[adr]) != 3 or adr not in jlookup:
                    raise ValueError(
                        "Only fixed links and one hinge per child link are supported"
                    )
                jid = jlookup[adr]
            self.joints.append(jid)
            axes.append(
                model.jnt_axis[adr].copy() if jid >= 0 else np.array([1.0, 0.0, 0.0])
            )
            pivots.append(model.jnt_pos[adr].copy() if jid >= 0 else np.zeros(3))
        self.register_buffer(
            "rest_pos", torch.tensor(model.body_pos[body_ids], dtype=torch.float32)
        )
        self.register_buffer(
            "rest_rot",
            quat_matrix(torch.tensor(model.body_quat[body_ids], dtype=torch.float32)),
        )
        self.register_buffer(
            "joint_ref",
            torch.tensor(
                model.qpos0[model.jnt_qposadr[joint_ids]], dtype=torch.float32
            ),
        )
        self.register_buffer(
            "axes", torch.tensor(np.asarray(axes), dtype=torch.float32)
        )
        self.register_buffer(
            "pivots", torch.tensor(np.asarray(pivots), dtype=torch.float32)
        )
        self.register_buffer(
            "tracked", torch.tensor([self.body_names.index(x) for x in tracked_names])
        )

    @classmethod
    def from_env(cls, env):
        robot = env.scene["robot"]
        return cls(
            env.sim.mj_model,
            robot.indexing.body_ids.cpu().tolist(),
            robot.indexing.joint_ids.cpu().tolist(),
            robot.body_names,
        ).to(env.device)

    def forward(self, s, all_bodies=False):
        if s.shape[-1] != 73:
            raise ValueError(
                "Tracking kinematics requires 73-dimensional root/joint state"
            )
        p, r, v, w = [s[:, :3]], [matrix6d(s[:, 3:9])], [s[:, 9:12]], [s[:, 12:15]]
        q, qd = s[:, 15:44], s[:, 44:73]
        eye = torch.eye(3, device=s.device, dtype=s.dtype)
        for i in range(1, len(self.parents)):
            parent, joint = self.parents[i], self.joints[i]
            rp = r[parent]
            rest = rp @ self.rest_rot[i]
            if joint >= 0:
                axis = self.axes[i]
                x, y, z = axis.unbind()
                zero = torch.zeros_like(x)
                skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(
                    3, 3
                )
                angle = (q[:, joint] - self.joint_ref[joint])[:, None, None]
                turn = eye + angle.sin() * skew + (1 - angle.cos()) * (skew @ skew)
                ri = rest @ turn
                shift = self.rest_pos[i] + apply(
                    self.rest_rot[i], self.pivots[i] - apply(turn, self.pivots[i])
                )
                pi = p[parent] + apply(rp, shift)
                axis_w = apply(rest, axis)
                pivot_w = p[parent] + apply(
                    rp, self.rest_pos[i] + apply(self.rest_rot[i], self.pivots[i])
                )
                delta_w = axis_w * qd[:, joint, None]
                wi = w[parent] + delta_w
                vi = (
                    v[parent]
                    + torch.cross(w[parent], pi - p[parent], dim=-1)
                    + torch.cross(delta_w, pi - pivot_w, dim=-1)
                )
            else:
                ri = rest.expand(s.shape[0], -1, -1)
                pi = p[parent] + apply(rp, self.rest_pos[i])
                wi = w[parent]
                vi = v[parent] + torch.cross(w[parent], pi - p[parent], dim=-1)
            p.append(pi)
            r.append(ri)
            v.append(vi)
            w.append(wi)
        out = {
            "pos": torch.stack(p, 1),
            "rot": torch.stack(r, 1),
            "lin_vel": torch.stack(v, 1),
            "ang_vel": torch.stack(w, 1),
        }
        return out if all_bodies else {k: x[:, self.tracked] for k, x in out.items()}
