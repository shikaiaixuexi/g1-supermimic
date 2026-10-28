import json
import os
from pathlib import Path
import shutil
import time

import pytest

from g1_supermimic.config import (
    resolve_path,
    training_defaults,
    motion_path,
    latest_checkpoint,
    load_asset_bundle,
)
from g1_supermimic.train import parse_cfg


def test_paths_expand_variables_and_reject_unset_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_G1_ROOT", str(tmp_path / "data with spaces"))
    assert (
        resolve_path("$TEST_G1_ROOT/motion.npz")
        == tmp_path / "data with spaces/motion.npz"
    )
    assert resolve_path("~") == Path.home()
    monkeypatch.delenv("TEST_G1_UNSET", raising=False)
    with pytest.raises(ValueError, match="Unset environment variable"):
        resolve_path("${TEST_G1_UNSET}/motion.npz")


def test_training_profile_paths_and_cli_precedence(tmp_path, monkeypatch):
    directory = tmp_path / "profiles"
    directory.mkdir()
    config = directory / "motion.json"
    config.write_text(
        json.dumps(
            {
                "task": "motion",
                "iterations": 10000,
                "motion_file": "../data/manifest.json",
                "run_dir": "../runs/motion",
            }
        )
    )
    work = tmp_path / "elsewhere"
    work.mkdir()
    monkeypatch.chdir(work)
    cfg = parse_cfg(
        ["--config", str(config), "--iterations", "12", "--run-dir", "my-run"]
    )
    assert cfg.task == "motion" and cfg.iterations == 12
    assert cfg.motion_file == str(tmp_path / "data/manifest.json")
    assert cfg.run_dir == str(work / "my-run")
    assert cfg.device == "cuda:0"


@pytest.mark.parametrize(
    "value, message",
    [
        ({"typo": 3}, "Unknown"),
        ({"iterations": "10"}, "must be int"),
        ([], "JSON object"),
    ],
)
def test_bad_training_profiles_fail_early(tmp_path, value, message):
    config = tmp_path / "bad.json"
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=message):
        training_defaults(config, {"iterations": 10})


def test_motion_override_after_checkpoint_relocation(tmp_path, monkeypatch):
    monkeypatch.delenv("G1_MOTION_FILE", raising=False)
    checkpoint = {"cfg": {"motion_file": str(tmp_path / "missing/manifest.json")}}
    source = tmp_path / "local/manifest.json"
    source.parent.mkdir()
    source.write_text('{"clips": []}')
    with pytest.raises(FileNotFoundError, match="--motion-file"):
        motion_path(checkpoint, tmp_path / "model.pt")
    assert motion_path(checkpoint, tmp_path / "model.pt", source) == source
    monkeypatch.setenv("G1_MOTION_FILE", str(source))
    assert motion_path(checkpoint, tmp_path / "model.pt") == source


def test_latest_checkpoint_ignores_fresh_files_and_selects_numeric_iteration(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    for directory, filename in [
        (first, "model_9000.pt"),
        (second, "model_10000.pt"),
        (second, "model_latest.pt"),
    ]:
        path = directory / filename
        path.touch()
        os.utime(path, (time.time() - 10, time.time() - 10))
    (second / "model_11000.pt").touch()
    assert latest_checkpoint([first, second], 8000) == second / "model_10000.pt"
    assert latest_checkpoint([first, second], 10000) is None


def bundle(directory):
    directory.mkdir()
    (directory / "g1.xml").write_text("<mujoco/>")
    names = [f"joint_{i}" for i in range(29)]
    values = {
        "joint_names": names,
        "mjcf": "g1.xml",
        "control_dt": 0.02,
        "physics_dt": 0.005,
    }
    for key, value in {
        "stiffness": 1.0,
        "damping": 0.1,
        "effort": 25.0,
        "armature": 0.01,
        "default_q": 0.0,
        "action_scale": 0.5,
        "target_min": -2.0,
        "target_max": 2.0,
    }.items():
        values[key] = dict.fromkeys(names, value)
    (directory / "control.json").write_text(json.dumps(values))
    return values


def test_exported_asset_bundle_survives_relocation(tmp_path):
    original, moved = tmp_path / "original", tmp_path / "moved"
    values = bundle(original)
    shutil.move(original, moved)
    loaded = load_asset_bundle(moved / "control.json")
    assert loaded["mjcf"] == str(moved / "g1.xml")
    assert loaded["usd_dir"] == str(moved / "usd")
    values["mjcf"] = str(moved / "g1.xml")
    (moved / "control.json").write_text(json.dumps(values))
    assert load_asset_bundle(moved / "control.json")["mjcf"] == values["mjcf"]


@pytest.mark.parametrize(
    "field, value", [("effort", 0), ("armature", -0.1), ("stiffness", float("nan"))]
)
def test_invalid_asset_parameters_are_rejected(tmp_path, field, value):
    values = bundle(tmp_path / "asset")
    values[field]["joint_0"] = value
    (tmp_path / "asset/control.json").write_text(json.dumps(values))
    with pytest.raises(ValueError):
        load_asset_bundle(tmp_path / "asset/control.json")
