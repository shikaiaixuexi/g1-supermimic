"""Numeric convention and velocity checks for real CSV conversion."""

import io
import re

import mujoco
import numpy as np
import pytest
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from scipy.spatial.transform import Rotation

from g1_supermimic.convert_motion import make_model, resample_csv, convert_clip
from g1_supermimic.motion import MotionLibrary


@pytest.fixture(scope="module")
def model():
    return make_model()


def csv_data(model):
    names = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    t = np.arange(121) / 120
    q = np.zeros((len(t), 29))
    for pattern, value in get_g1_robot_cfg().init_state.joint_pos.items():
        for j, name in enumerate(names):
            if re.fullmatch(pattern, name):
                q[:, j] = np.rad2deg(value)
    q[:, names.index("left_shoulder_pitch_joint")] += 5 * t
    # Reverse joint columns to ensure mapping uses names rather than CSV order.
    columns = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
    ] + [n + "_dof" for n in names[::-1]]
    values = np.column_stack(
        (
            np.arange(len(t)),
            100 + 10 * t,
            200 + 20 * t,
            np.full(len(t), 80),
            np.full(len(t), 15),
            np.full(len(t), 10),
            20 + 30 * t,
            q[:, ::-1],
        )
    )
    stream = io.StringIO()
    np.savetxt(stream, values, delimiter=",", header=",".join(columns), comments="")
    return stream.getvalue().encode(), names, values


def test_csv_units_rotation_named_order_and_rate(model):
    raw, names, values = csv_data(model)
    pos, quat, q = resample_csv(raw, names)
    assert len(pos) == 51
    np.testing.assert_allclose(pos[0], [0, 0, 0.8], atol=1e-10)
    np.testing.assert_allclose(pos[-1], [0.1, 0.2, 0.8], atol=1e-10)
    np.testing.assert_allclose(q[0], np.deg2rad(values[0, 7:][::-1]), atol=1e-10)
    actual = Rotation.from_quat(quat[0, [1, 2, 3, 0]]).as_matrix()
    # Independent composition verifies Euler ordering for nonzero roll/pitch/yaw.
    expected = (
        Rotation.from_euler("z", 20, degrees=True)
        * Rotation.from_euler("y", 10, degrees=True)
        * Rotation.from_euler("x", 15, degrees=True)
    ).as_matrix()
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_missing_and_nonconsecutive_frames_rejected(model):
    raw, names, _ = csv_data(model)
    with pytest.raises(ValueError, match="Missing CSV columns"):
        resample_csv(raw.replace(b"root_translateZ", b"wrong_column"), names)
    lines = raw.splitlines()
    with pytest.raises(ValueError, match="consecutive"):
        resample_csv(b"\n".join(lines[:30] + lines[31:]), names)


def test_converted_world_velocities_and_training_contract(model, tmp_path):
    raw, names, _ = csv_data(model)
    arrays, report = convert_clip(raw, model)
    assert report["frames"] == 51
    pos = arrays["body_pos_w"].astype(float)
    velocity_fd = (pos[2:] - pos[:-2]) / 0.04
    np.testing.assert_allclose(arrays["body_lin_vel_w"][1:-1], velocity_fd, atol=1e-4)
    root = list(arrays["body_names"]).index("pelvis")
    np.testing.assert_allclose(
        arrays["body_ang_vel_w"][:, root],
        np.tile([0, 0, np.deg2rad(30)], (51, 1)),
        atol=1e-5,
    )
    path = tmp_path / "converted.npz"
    np.savez_compressed(path, **arrays)
    library = MotionLibrary(path, names)
    assert not library.synthetic
    assert library.data["state"].shape == (51, 73)
