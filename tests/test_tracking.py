"""Tracking reward mathematics, reference timelines and differentiable kinematics."""

import json
import math

import numpy as np
import pytest
import torch

from g1_supermimic.kinematics import (
    matrix6d,
    rotation6d,
    rotation_error_squared,
    quat_matrix,
    proprioception,
)
from g1_supermimic.motion import MotionLibrary
from g1_supermimic.tracking import (
    WEIGHTS,
    tracking_terms,
    total_reward,
    hard_failure,
    soft_failure,
)
from g1_supermimic.learning import Replay, WorldModel


def state():
    s = torch.zeros(2, 73)
    s[:, 2] = 0.8
    s[:, 3:9] = rotation6d(torch.eye(3)).expand(2, -1)
    return s


def reward_case():
    s = state()
    p = torch.tensor([[[0.0, 0.0, 0.8], [0.0, 0.0, 1.1]]]).expand(2, -1, -1).clone()
    r = torch.eye(3).expand(2, 2, 3, 3).clone()
    b = {
        "pos": p,
        "rot": r,
        "lin_vel": torch.zeros(2, 2, 3),
        "ang_vel": torch.zeros(2, 2, 3),
    }
    ref = {k: v.clone() for k, v in b.items()}
    ref["state"] = s.clone()
    a = torch.zeros(2, 29)
    limits = torch.stack((-torch.ones(1, 29), torch.ones(1, 29)), -1)
    return s, b, ref, a, limits


def test_perfect_tracking_has_five_reward_rate_and_no_self_collision_term():
    s, b, ref, a, limits = reward_case()
    terms = tracking_terms(s, b, ref, a, a, limits)
    assert set(terms) == set(WEIGHTS)
    for k, v in terms.items():
        torch.testing.assert_close(
            v,
            torch.full(
                (2,), 0.0 if k in ("action_smoothness", "joint_limits") else 1.0
            ),
        )
    torch.testing.assert_close(total_reward(terms), torch.full((2,), 5.0))
    assert WEIGHTS["joint_limits"] == -2 and WEIGHTS["action_smoothness"] == -0.1


def test_body_alignment_does_not_remove_global_root_position_reward():
    s, b, ref, a, limits = reward_case()
    s[:, 0] += 0.3
    b["pos"][:, :, 0] += 0.3
    terms = tracking_terms(s, b, ref, a, a, limits)
    torch.testing.assert_close(terms["body_position"], torch.ones(2))
    torch.testing.assert_close(terms["root_position"], torch.full((2,), math.exp(-1)))


def test_body_alignment_does_not_remove_global_root_heading_reward():
    s, b, ref, a, limits = reward_case()
    q = torch.tensor([math.cos(0.2), 0.0, 0.0, math.sin(0.2)])
    r = quat_matrix(q)
    s[:, 3:9] = rotation6d(r)
    b["rot"] = r.expand(2, 2, 3, 3)
    terms = tracking_terms(s, b, ref, a, a, limits)
    torch.testing.assert_close(terms["body_orientation"], torch.ones(2))
    torch.testing.assert_close(
        terms["root_orientation"], torch.full((2,), math.exp(-1))
    )


def test_global_body_velocity_scales_and_regularizers():
    s, b, ref, a, limits = reward_case()
    b["lin_vel"][:, :, 0] = 1.0
    b["ang_vel"][:, :, 2] = 3.14
    s[:, 15:44] = 1.2
    terms = tracking_terms(s, b, ref, torch.ones_like(a), a, limits)
    torch.testing.assert_close(
        terms["body_linear_velocity"], torch.full((2,), math.exp(-1))
    )
    torch.testing.assert_close(
        terms["body_angular_velocity"], torch.full((2,), math.exp(-1))
    )
    torch.testing.assert_close(terms["action_smoothness"], torch.full((2,), 29.0))
    torch.testing.assert_close(terms["joint_limits"], torch.full((2,), 29 * 0.2))


def test_geodesic_angle_and_zero_error_gradients_are_finite():
    q = torch.tensor([math.cos(0.25), 0.0, 0.0, math.sin(0.25)])
    r = quat_matrix(q)
    torch.testing.assert_close(
        rotation_error_squared(r, torch.eye(3)), torch.tensor(0.25)
    )
    x = rotation6d(torch.eye(3)).clone().requires_grad_()
    rotation_error_squared(matrix6d(x), torch.eye(3)).backward()
    assert torch.isfinite(x.grad).all()


def test_failure_uses_reference_height_and_tilt_not_absolute_uprightness():
    ref = state()
    s = ref.clone()
    s[0, 2] += 0.249
    s[1, 2] += 0.251
    assert hard_failure(s, ref).tolist() == [False, True]
    r = quat_matrix(torch.tensor([0.0, 1.0, 0.0, 0.0]))
    ref[:, 3:9] = rotation6d(r)
    s = ref.clone()
    assert not hard_failure(s, ref).any()  # An inverted reference can be tracked.
    s[0, 3:9] = rotation6d(torch.eye(3))
    assert hard_failure(s, ref).tolist() == [True, False]
    s = s.detach().requires_grad_()
    soft_failure(s, ref).sum().backward()
    assert torch.isfinite(s.grad).all()


def write_clip(path, marker=0.0, synthetic=True):
    t = 12
    joint_names = np.array([f"joint_{i}" for i in range(29)])
    pos = np.zeros((t, 2, 3), np.float32)
    pos[:, 0, 2] = 0.8
    pos[:, 1, 2] = 1.1
    quat = np.zeros((t, 2, 4), np.float32)
    quat[..., 0] = 1
    np.savez(
        path,
        joint_pos=np.full((t, 29), marker, np.float32),
        joint_vel=np.zeros((t, 29), np.float32),
        body_pos_w=pos,
        body_quat_w=quat,
        body_lin_vel_w=np.zeros_like(pos),
        body_ang_vel_w=np.zeros_like(pos),
        joint_names=joint_names,
        body_names=np.array(["pelvis", "torso_link"]),
        dt=0.02,
        synthetic_test_fixture=synthetic,
    )
    return joint_names.tolist()


def test_manifest_timeline_never_crosses_clips_and_goal_has_real_future(tmp_path):
    names = write_clip(tmp_path / "a.npz", 0.1)
    write_clip(tmp_path / "b.npz", 0.7)
    manifest = tmp_path / "clips.json"
    manifest.write_text(json.dumps({"clips": ["a.npz", "b.npz"]}))
    lib = MotionLibrary(
        manifest, names, ("pelvis", "torso_link"), allow_test_motion=True
    )
    idx = torch.tensor([10, 11, 22, 23])
    assert lib.advance(idx).tolist() == [11, 11, 23, 23]
    assert lib.ending(idx).all()
    goals = lib.goal(torch.tensor([10, 22]), state())
    assert goals.shape == (2, 10, 73)
    torch.testing.assert_close(goals[0, :, 15:44], torch.full((10, 29), 0.1))
    torch.testing.assert_close(goals[1, :, 15:44], torch.full((10, 29), 0.7))
    assert not torch.isin(lib.sample(200), lib.ends).any()
    assert lib.synthetic


def test_motion_loader_rejects_unlabeled_fixture_bad_order_and_dt(tmp_path):
    path = tmp_path / "clip.npz"
    names = write_clip(path)
    with pytest.raises(ValueError, match="Synthetic fixture"):
        MotionLibrary(path, names, ("pelvis", "torso_link"))
    with pytest.raises(ValueError, match="names/order"):
        MotionLibrary(
            path, names[::-1], ("pelvis", "torso_link"), allow_test_motion=True
        )
    with pytest.raises(ValueError, match="resampled"):
        MotionLibrary(
            path, names, ("pelvis", "torso_link"), dt=0.04, allow_test_motion=True
        )


def test_tracking_replay_keeps_reference_frame_identity():
    replay = Replay(1, 12, 73, 29, "cpu", command_dim=0, reference_indices=True)
    for i in range(12):
        replay.add(
            torch.full((1, 73), float(i)),
            torch.zeros(1, 73),
            torch.zeros(1, 29),
            torch.zeros(1, 29),
            torch.empty(1, 0),
            torch.tensor([i == 5]),
            reference_index=torch.tensor([i]),
        )
    sample = replay.seeds(50)
    torch.testing.assert_close(sample["s"][:, 0], sample["reference_index"].float())
    assert sample["reference_index"].dtype == torch.long
    w = replay.windows(50, 8)
    expected = (w["reference_index"] != 5).float().cumprod(1)
    torch.testing.assert_close(w["valid"], expected)


def test_tracking_observation_is_the_same_proprioceptive_schema():
    s = state()
    s[:, 15:44] = 0.3
    s[:, 44:73] = 0.4
    frame = proprioception(s, torch.full((2, 29), 0.5), torch.full((1, 29), 0.1))
    assert frame.shape == (2, 93)
    torch.testing.assert_close(frame[:, :29], torch.full((2, 29), 0.2))
    torch.testing.assert_close(frame[:, 29:58], torch.full((2, 29), 0.4))
    torch.testing.assert_close(
        frame[:, 61:64], torch.tensor([[0.0, 0.0, -1.0]]).expand(2, -1)
    )


def test_tracking_world_model_projects_rotation_and_remains_differentiable():
    wm = WorldModel(73, 29, width=32, project_rotation=True)
    s = state().requires_grad_()
    a = torch.randn(2, 29, requires_grad=True)
    nxt = wm(s, a, torch.zeros_like(a))
    r = matrix6d(nxt[:, 3:9])
    torch.testing.assert_close(
        r.transpose(-1, -2) @ r, torch.eye(3).expand(2, -1, -1), atol=1e-5, rtol=1e-5
    )
    (nxt[:, :3].square().sum() + r.square().sum()).backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0


def test_vertical_heading_and_degenerate_rotation_have_finite_gradients():
    from g1_supermimic.kinematics import yaw_matrix

    vertical = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], requires_grad=True
    )
    yaw_matrix(vertical).sum().backward()
    assert torch.isfinite(vertical.grad).all()
    zero = torch.zeros(2, 6, requires_grad=True)
    r = matrix6d(zero)
    torch.testing.assert_close(r.transpose(-1, -2) @ r, torch.eye(3).expand(2, -1, -1))
    r.sum().backward()
    assert torch.isfinite(zero.grad).all()


def test_critic_distinguishes_identical_stationary_goals_at_different_remaining_times():
    from types import SimpleNamespace
    from g1_supermimic.train import Trainer

    runtime = Trainer.__new__(Trainer)
    runtime.motion = SimpleNamespace(
        library=SimpleNamespace(end_at=torch.full((12,), 11))
    )
    runtime.env = SimpleNamespace(step_dt=0.02)
    s = state()
    previous = torch.zeros(2, 29)
    goal = torch.zeros(2, 10, 73)
    x = runtime.critic_input(s, previous, goal, torch.tensor([0, 10]))
    assert x.shape == (2, 833)
    torch.testing.assert_close(x[:, -1], torch.tensor([0.22, 0.02]))
