"""Portable paths, JSON training profiles and checkpoint discovery."""

import json
import math
import os
from pathlib import Path
import re
import time


def resolve_path(value, base=None):
    expanded = os.path.expandvars(os.fspath(value))
    if re.search(r"\$(?:[A-Za-z_]\w*|\{[^}]+\})", expanded):
        raise ValueError(f"Unset environment variable in path: {value}")
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = (Path(base) if base is not None else Path.cwd()) / path
    return path.resolve()


def training_defaults(filename, defaults):
    path = resolve_path(filename)
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Training config must be a JSON object")
    unknown = values.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown training options: {', '.join(sorted(unknown))}")
    for key, value in values.items():
        expected = type(defaults[key])
        if not (type(value) is expected or (expected is float and type(value) is int)):
            raise ValueError(f"Config option {key!r} must be {expected.__name__}")
        if key in ("motion_file", "resume", "run_dir") and value:
            values[key] = str(resolve_path(value, path.parent))
    return {**defaults, **values}


def motion_path(checkpoint, checkpoint_path, override=None):
    source = (
        override
        or os.environ.get("G1_MOTION_FILE")
        or checkpoint["cfg"].get("motion_file")
    )
    if not source:
        raise ValueError("Provide --motion-file or set G1_MOTION_FILE")
    path = resolve_path(source)
    if not path.exists() and not override and not os.environ.get("G1_MOTION_FILE"):
        path = resolve_path(source, resolve_path(checkpoint_path).parent)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference data not found: {path}. Provide --motion-file pointing to your local NPZ or manifest."
        )
    return path


def latest_checkpoint(sources, minimum_iteration=0, settle_seconds=2):
    candidates = []
    now = time.time()
    for source in sources:
        path = resolve_path(source)
        paths = [path] if path.suffix == ".pt" else path.glob("model_*.pt")
        for candidate in paths:
            match = re.fullmatch(r"model_(\d+)\.pt", candidate.name)
            if (
                match
                and int(match[1]) > minimum_iteration
                and candidate.is_file()
                and now - candidate.stat().st_mtime >= settle_seconds
            ):
                candidates.append((int(match[1]), candidate))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def load_asset_bundle(filename):
    path = resolve_path(filename)
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Asset bundle must be a JSON object")
    names = values.get("joint_names", [])
    if (
        not isinstance(names, list)
        or len(names) != 29
        or not all(isinstance(n, str) and n for n in names)
        or len(set(names)) != 29
    ):
        raise ValueError("Asset bundle must contain 29 unique joint names")
    for key in (
        "stiffness",
        "damping",
        "effort",
        "armature",
        "default_q",
        "action_scale",
        "target_min",
        "target_max",
    ):
        table = values.get(key, {})
        if set(table) != set(names) or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in table.values()
        ):
            raise ValueError(f"Invalid asset control field: {key}")
    for name in names:
        if any(
            values[k][name] < 0 for k in ("stiffness", "damping", "armature")
        ) or any(values[k][name] <= 0 for k in ("effort", "action_scale")):
            raise ValueError(f"Invalid actuator parameters for {name}")
        if values["target_min"][name] >= values["target_max"][name]:
            raise ValueError(f"Invalid target limits for {name}")
    control_dt, physics_dt = values.get("control_dt", 0), values.get("physics_dt", 0)
    if not (
        math.isfinite(control_dt)
        and math.isfinite(physics_dt)
        and control_dt > 0
        and physics_dt > 0
    ):
        raise ValueError("Asset time steps must be finite and positive")
    ratio = control_dt / physics_dt
    if ratio < 1 or abs(ratio - round(ratio)) > 1e-6:
        raise ValueError("Control dt must be an integer multiple of physics dt")
    values["mjcf"] = str(resolve_path(values["mjcf"], path.parent))
    if not Path(values["mjcf"]).is_file():
        raise FileNotFoundError(f"Asset MJCF not found: {values['mjcf']}")
    values["usd_dir"] = str(resolve_path(values.get("usd_dir", "usd"), path.parent))
    return values
