"""Named, frame-validated motion clips for full G1 reference tracking.

Files use world-frame link positions/quaternions/velocities, wxyz quaternions,
metres/radians/seconds and exactly the control sample period. No action labels
are inferred from poses. Full datasets can be listed in a JSON clip manifest.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from .kinematics import (
    TRACKED_BODIES,
    apply,
    matrix6d,
    quat_matrix,
    rotation6d,
    yaw_matrix,
)
from .config import resolve_path


class MotionLibrary:
    feature_dim = 73
    feature_schema = "g1-root-joint-yaw-v1"

    def __init__(
        self,
        source,
        joint_names,
        tracked_names=TRACKED_BODIES,
        dt=0.02,
        device="cpu",
        allow_test_motion=False,
    ):
        source = resolve_path(source)
        if source.is_dir():
            paths = sorted(source.glob("*.npz"))
        elif source.suffix == ".json":
            obj = json.loads(source.read_text())
            paths = [resolve_path(x, source.parent) for x in obj["clips"]]
        else:
            paths = [source]
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("Motion source must contain at least one unique clip")
        self.joint_names, self.tracked_names = tuple(joint_names), tuple(tracked_names)
        self.device, self.dt = device, dt
        data, ends, starts, labels = {}, [], [], []
        digest = hashlib.sha256()
        count = 0
        self.synthetic = False
        required = {
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "joint_names",
            "body_names",
        }
        for path in paths:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
            with np.load(path, allow_pickle=False) as f:
                if not required.issubset(f.files):
                    raise ValueError(
                        f"{path}: missing keys {sorted(required - set(f.files))}"
                    )
                fps = float(f["fps"].item()) if "fps" in f.files else float("nan")
                clip_dt = (
                    float(f["dt"].item())
                    if "dt" in f.files
                    else 1 / fps
                    if np.isfinite(fps) and fps > 0
                    else float("nan")
                )
                if not np.isfinite(clip_dt) or abs(clip_dt - dt) > 1e-7:
                    raise ValueError(
                        "Motion clips must be explicitly resampled to the control dt"
                    )
                if (
                    "dt" in f.files
                    and "fps" in f.files
                    and (
                        not np.isfinite(fps)
                        or fps <= 0
                        or abs(clip_dt - 1 / fps) > 1e-7
                    )
                ):
                    raise ValueError("Motion dt and fps disagree")
                names = tuple(str(x) for x in f["joint_names"].tolist())
                bodies = tuple(str(x) for x in f["body_names"].tolist())
                if names != self.joint_names or len(set(names)) != 29:
                    raise ValueError(
                        "Motion joint names/order must exactly match the 29-joint G1"
                    )
                if (
                    len(set(bodies)) != len(bodies)
                    or "pelvis" not in bodies
                    or not set(tracked_names).issubset(bodies)
                ):
                    raise ValueError(
                        "Motion body_names must identify pelvis and every tracked body uniquely"
                    )
                synthetic = (
                    bool(f["synthetic_test_fixture"].item())
                    if "synthetic_test_fixture" in f.files
                    else False
                )
                if synthetic and not allow_test_motion:
                    raise ValueError(
                        "Synthetic fixture requires explicit --allow-test-motion 1; it is not a dataset"
                    )
                self.synthetic |= synthetic
                arr = {
                    k: np.asarray(f[k], dtype=np.float32)
                    for k in required
                    if k not in ("joint_names", "body_names")
                }
                if arr["joint_pos"].ndim != 2:
                    raise ValueError("joint_pos must be [T,29]")
                t = arr["joint_pos"].shape[0]
                if (
                    t < 11
                    or arr["joint_pos"].shape != (t, 29)
                    or arr["joint_vel"].shape != (t, 29)
                ):
                    raise ValueError(
                        "Motion clip needs at least 11 frames with joint_pos/joint_vel [T,29]"
                    )
                for key, dim in (
                    ("body_pos_w", 3),
                    ("body_quat_w", 4),
                    ("body_lin_vel_w", 3),
                    ("body_ang_vel_w", 3),
                ):
                    if arr[key].shape != (t, len(bodies), dim):
                        raise ValueError(f"Bad shape for {key}")
                if not all(np.isfinite(x).all() for x in arr.values()):
                    raise ValueError("Non-finite motion data")
                norms = np.linalg.norm(arr["body_quat_w"], axis=-1)
                if np.max(np.abs(norms - 1)) > 1e-3:
                    raise ValueError("Motion quaternions must be unit wxyz quaternions")
                arr["body_quat_w"] = arr["body_quat_w"] / norms[..., None]
                root = bodies.index("pelvis")
                root_rot = quat_matrix(torch.from_numpy(arr["body_quat_w"][:, root]))
                state = torch.cat(
                    (
                        torch.from_numpy(arr["body_pos_w"][:, root]),
                        rotation6d(root_rot),
                        torch.from_numpy(arr["body_lin_vel_w"][:, root]),
                        torch.from_numpy(arr["body_ang_vel_w"][:, root]),
                        torch.from_numpy(arr["joint_pos"]),
                        torch.from_numpy(arr["joint_vel"]),
                    ),
                    -1,
                )
                ids = [bodies.index(x) for x in tracked_names]
                part = {
                    "state": state,
                    "pos": torch.from_numpy(arr["body_pos_w"][:, ids]),
                    "rot": quat_matrix(torch.from_numpy(arr["body_quat_w"][:, ids])),
                    "lin_vel": torch.from_numpy(arr["body_lin_vel_w"][:, ids]),
                    "ang_vel": torch.from_numpy(arr["body_ang_vel_w"][:, ids]),
                    "root_quat": torch.from_numpy(arr["body_quat_w"][:, root]),
                }
                for key, x in part.items():
                    data.setdefault(key, []).append(x)
                starts.append(count)
                ends.append(count + t - 1)
                labels.append(path.name)
                count += t
        self.data = {k: torch.cat(v).to(device) for k, v in data.items()}
        self.starts = torch.tensor(starts, device=device, dtype=torch.long)
        self.ends = torch.tensor(ends, device=device, dtype=torch.long)
        self.end_at = torch.cat(
            [
                torch.full((e - s + 1,), e, dtype=torch.long)
                for s, e in zip(starts, ends)
            ]
        ).to(device)
        self.valid_starts = torch.cat(
            [torch.arange(s, e, dtype=torch.long) for s, e in zip(starts, ends)]
        ).to(device)
        self.signature = digest.hexdigest()
        self.labels = labels
        self.source = str(source)

    def advance(self, index, offset=1):
        return torch.minimum(index + offset, self.end_at[index])

    def ending(self, index):
        return index + 1 >= self.end_at[index]

    def sample(self, n, mode="uniform"):
        if mode == "start":
            return self.starts[
                torch.randint(len(self.starts), (n,), device=self.device)
            ]
        if mode != "uniform":
            raise ValueError("Reference reset mode must be uniform or start")
        return self.valid_starts[
            torch.randint(len(self.valid_starts), (n,), device=self.device)
        ]

    def frame(self, index):
        return {k: x[index] for k, x in self.data.items()}

    def goal(self, index, state):
        offsets = torch.arange(1, 11, device=state.device)
        ids = torch.minimum(index[:, None] + offsets, self.end_at[index, None])
        ref = self.data["state"][ids]
        yaw = yaw_matrix(matrix6d(state[:, 3:9])).transpose(-1, -2)[:, None]
        pos = apply(yaw, ref[..., :3] - state[:, None, :3])
        rot = yaw @ matrix6d(ref[..., 3:9])
        vel = apply(yaw, ref[..., 9:12])
        angular = apply(yaw, ref[..., 12:15])
        return torch.cat((pos, rotation6d(rot), vel, angular, ref[..., 15:]), -1)

    def metadata(self):
        return {
            "signature": self.signature,
            "clips": self.labels,
            "frames": len(self.end_at),
            "dt": self.dt,
            "feature_schema": self.feature_schema,
            "feature_dim": self.feature_dim,
            "tracked_bodies": list(self.tracked_names),
            "synthetic_test_fixture": self.synthetic,
        }
