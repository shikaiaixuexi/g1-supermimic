"""BONES-SEED CSV -> named 50 Hz motion NPZ, using the training G1 model.

Reads CSV files/directories or streams a tar archive without extracting it.
CSV convention follows bones-studio/seed-viewer: cm, degrees, Z-up, Rz Ry Rx.
Source frame rate is explicit because the CSV has frame indices, not timestamps.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from .config import resolve_path


def make_model():
    spec = Entity(get_g1_robot_cfg()).spec
    spec.worldbody.add_geom(
        name="conversion_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1],
        pos=[0, 0, 0],
    )
    return spec.compile()


def resample_csv(raw, joint_names, source_fps=120.0, target_fps=50.0):
    if not np.isfinite(source_fps) or source_fps <= 0 or target_fps != 50:
        raise ValueError("Source FPS must be positive; current control rate is 50 Hz")
    a = np.genfromtxt(io.StringIO(raw.decode("utf-8-sig")), delimiter=",", names=True)
    a = np.atleast_1d(a)
    columns = a.dtype.names or ()
    expected = ["Frame"] + [
        f"root_{kind}{axis}" for kind in ("translate", "rotate") for axis in "XYZ"
    ]
    expected += [name + "_dof" for name in joint_names]
    if not set(expected).issubset(columns):
        raise ValueError(f"Missing CSV columns: {sorted(set(expected) - set(columns))}")
    if {x for x in columns if x.endswith("_dof")} != {n + "_dof" for n in joint_names}:
        raise ValueError("CSV joint names must match the G1 model")
    values = np.column_stack([a[x] for x in expected])
    if len(a) < 3 or not np.isfinite(values).all():
        raise ValueError("Need at least three finite CSV frames")
    frames = a["Frame"]
    if not np.allclose(frames, np.round(frames)) or not np.all(np.diff(frames) == 1):
        raise ValueError("CSV frame indices must be consecutive integers")
    source_time = (frames - frames[0]) / source_fps
    time = (
        np.arange(int(np.floor(source_time[-1] * target_fps + 1e-8)) + 1) / target_fps
    )
    if len(time) < 11:
        raise ValueError("Clip is shorter than 11 control frames")
    position = values[:, 1:4] * 0.01
    position[:, :2] -= position[0, :2].copy()
    # Intrinsic ZYX receives z,y,x; equivalent to Three.js Euler(x,y,z,'ZYX').
    rotation = Rotation.from_euler("ZYX", values[:, 4:7][:, ::-1], degrees=True)
    quat_xyzw = Slerp(source_time, rotation)(time).as_quat()
    q = np.deg2rad(values[:, 7:])

    def interp(x):
        return np.column_stack(
            [np.interp(time, source_time, x[:, j]) for j in range(x.shape[1])]
        )

    return interp(position), quat_xyzw[:, [3, 0, 1, 2]], interp(q)


def convert_clip(raw, model, source_fps=120.0):
    joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(joints) != 29 or len(free) != 1:
        raise ValueError("Expected a floating-base 29-joint G1")
    names = [model.joint(int(i)).name for i in joints]
    bodies = list(range(1, model.nbody))
    body_names = [model.body(i).name for i in bodies]
    if body_names[0] != "pelvis":
        raise ValueError("Expected pelvis root")
    pos, quat, q = resample_csv(raw, names, source_fps)
    dt = 0.02
    qpos = np.tile(model.qpos0, (len(q), 1))
    root_adr = int(model.jnt_qposadr[free[0]])
    qpos[:, root_adr : root_adr + 7] = np.column_stack((pos, quat))
    qpos[:, model.jnt_qposadr[joints]] = q
    qvel = np.empty((len(q), model.nv))
    for i in range(len(q)):
        lo, hi = max(0, i - 1), min(len(q) - 1, i + 1)
        # Handles quaternion derivatives and MuJoCo's local root angular velocity.
        mujoco.mj_differentiatePos(model, qvel[i], (hi - lo) * dt, qpos[lo], qpos[hi])
    output = {
        key: []
        for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    }
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "conversion_floor")
    feet = [
        i
        for i in range(model.ngeom)
        if re.fullmatch(r"(?:left|right)_foot\d+_collision", model.geom(i).name)
    ]
    if floor < 0 or not feet:
        raise ValueError("Missing validation floor/foot collision geometries")
    foot_distance = []
    data = mujoco.MjData(model)
    for p, v in zip(qpos, qvel):
        data.qpos[:] = p
        data.qvel[:] = v
        mujoco.mj_forward(model, data)
        bp = data.xpos[bodies].copy()
        bw = data.cvel[bodies, :3].copy()
        bv = data.cvel[bodies, 3:].copy() - np.cross(
            bw, data.subtree_com[bodies[0]] - bp
        )
        for key, value in zip(output, (bp, data.xquat[bodies].copy(), bv, bw)):
            output[key].append(value)
        foot_distance.append(
            min(mujoco.mj_geomDistance(model, data, floor, g, 10, None) for g in feet)
        )
    limits = model.jnt_range[joints]
    violation = np.maximum(limits[:, 0] - q, 0) + np.maximum(q - limits[:, 1], 0)
    foot_distance = np.asarray(foot_distance)
    report = {
        "frames": len(q),
        "duration_s": (len(q) - 1) * dt,
        "root_height_min_m": float(pos[:, 2].min()),
        "root_height_max_m": float(pos[:, 2].max()),
        "joint_limit_max_violation_rad": float(violation.max()),
        "foot_ground_min_m": float(foot_distance.min()),
        "foot_ground_max_m": float(foot_distance.max()),
    }
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in output.items()}
    arrays.update(
        joint_pos=q.astype(np.float32),
        joint_vel=qvel[:, model.jnt_dofadr[joints]].astype(np.float32),
        joint_names=np.asarray(names),
        body_names=np.asarray(body_names),
        dt=dt,
        source_fps=source_fps,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )
    if not all(
        np.isfinite(v).all() for v in arrays.values() if np.asarray(v).dtype.kind == "f"
    ):
        raise ValueError("Non-finite converted motion")
    return arrays, report


def csv_sources(source, pattern, exclude):
    """Yield matching files without unpacking archives or trusting archive paths."""

    def match(name):
        return pattern.search(name) and not (exclude and exclude.search(name))

    if source.is_file() and tarfile.is_tarfile(source):
        with tarfile.open(
            source, "r|gz" if source.name.endswith((".gz", ".tgz")) else "r|"
        ) as archive:
            for member in archive:
                if (
                    member.isfile()
                    and member.name.lower().endswith(".csv")
                    and match(member.name)
                ):
                    with archive.extractfile(member) as stream:
                        yield member.name, stream.read()
    else:
        paths = sorted(source.rglob("*.csv")) if source.is_dir() else [source]
        for path in paths:
            if match(str(path)):
                yield str(path), path.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=resolve_path)
    parser.add_argument("--output", required=True, type=resolve_path)
    parser.add_argument("--source-fps", type=float, default=120)
    parser.add_argument(
        "--match", default=r"(?i)(?:^|/)(?:walk(?:ing)?|idle|stand(?:ing)?)(?:_|\b)"
    )
    parser.add_argument(
        "--exclude",
        default=r"(?i)stair|ladder|box|chair|sit|climb|jump|run|backward|backwards|sideways|crouch|fence|slope|uneven",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--max-joint-violation", type=float, default=0.15)
    parser.add_argument("--min-root-height", type=float, default=0.45)
    parser.add_argument("--max-penetration", type=float, default=0.08)
    args = parser.parse_args()
    if args.limit < 1 or args.max_candidates < args.limit:
        parser.error("Require 1 <= limit <= max-candidates")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        parser.error(
            "Output manifest exists; choose a new directory to preserve dataset/checkpoint identity"
        )
    model = make_model()
    accepted, records = [], []
    for label, raw in csv_sources(
        args.source, re.compile(args.match), re.compile(args.exclude)
    ):
        record = {"source": label}
        try:
            arrays, report = convert_clip(raw, model, args.source_fps)
            record.update(report)
            reasons = []
            if report["joint_limit_max_violation_rad"] > args.max_joint_violation:
                reasons.append("joint_limit")
            if report["root_height_min_m"] < args.min_root_height:
                reasons.append("root_height")
            if report["foot_ground_min_m"] < -args.max_penetration:
                reasons.append("ground_penetration")
            if report["foot_ground_max_m"] > 0.20:
                reasons.append("elevated_support")
            if reasons:
                record.update(status="rejected", reasons=reasons)
            else:
                suffix = hashlib.sha256(label.encode()).hexdigest()[:8]
                filename = Path(label).stem + "-" + suffix + ".npz"
                if (args.output / filename).exists():
                    raise FileExistsError(filename)
                np.savez_compressed(
                    args.output / filename,
                    **arrays,
                    source=label,
                    source_convention="BONES-SEED cm/deg Z-up RzRyRx; XY recentered",
                )
                accepted.append(filename)
                record.update(status="accepted", output=filename)
        except (ValueError, KeyError) as error:
            record.update(status="rejected", reasons=[str(error)])
        records.append(record)
        print(json.dumps(record), flush=True)
        if len(accepted) >= args.limit or len(records) >= args.max_candidates:
            break
    report = {
        "source": str(args.source.resolve()),
        "source_fps": args.source_fps,
        "target_fps": 50,
        "selection_match": args.match,
        "selection_exclude": args.exclude,
        "max_joint_violation_rad": args.max_joint_violation,
        "min_root_height_m": args.min_root_height,
        "max_penetration_m": args.max_penetration,
        "records": records,
        "accepted": len(accepted),
        "note": "Small bringup subset; basic kinematic screening, not full paper feasibility filtering",
    }
    (args.output / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if not accepted:
        raise SystemExit("No clips passed screening; inspect conversion_report.json")
    (args.output / "manifest.json").write_text(
        json.dumps({"clips": accepted}, indent=2) + "\n"
    )
    from .motion import MotionLibrary

    MotionLibrary(
        args.output / "manifest.json",
        [
            model.joint(int(j)).name
            for j in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        ],
    )
    print(
        json.dumps(
            {"manifest": str(args.output / "manifest.json"), "clips": len(accepted)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
