"""Run a motion policy in Isaac Lab beside its reference trajectory."""

import argparse
from pathlib import Path
import sys
import traceback


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_supermimic.config import (
    resolve_path,
    motion_path,
    load_asset_bundle,
    latest_checkpoint,
)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, type=resolve_path)
parser.add_argument(
    "--checkpoint-dir",
    action="append",
    type=resolve_path,
    default=[],
    help="Follow newer checkpoints in these directories (repeatable)",
)
parser.add_argument(
    "--final-checkpoint",
    type=resolve_path,
    help="Switch to this checkpoint once training writes it",
)
parser.add_argument(
    "--bundle", default="assets/isaac_g1/control.json", type=resolve_path
)
parser.add_argument("--motion-file", type=resolve_path)
parser.add_argument(
    "--steps", type=int, default=0, help="Zero runs until the GUI is closed"
)
parser.add_argument("--output", default="logs/isaac/rollout.json", type=resolve_path)
parser.add_argument("--realtime", type=int, choices=[0, 1], default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 0:
    parser.error("--steps must be nonnegative")
launcher = AppLauncher(args)
app = launcher.app
print("ISAAC_APP_READY", flush=True)

import json
import logging
import os
import tempfile
import time
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.sim import SimulationContext
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.utils.logger import configure_logging

from g1_supermimic.policy import Actor, PolicySpec, reset_history, push_history
from g1_supermimic.motion import MotionLibrary
from g1_supermimic.kinematics import quat_matrix, matrix6d, rotation6d, proprioception


def hard_failure(state, reference):
    # Keep the Isaac environment independent of mjlab.
    height = (state[:, 2] - reference[:, 2]).abs()
    tilt = (
        matrix6d(state[:, 3:9])[:, 2, 2] - matrix6d(reference[:, 3:9])[:, 2, 2]
    ).abs()
    return (height > 0.25) | (tilt > 0.8)


def main():
    control = load_asset_bundle(args.bundle)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if ckpt.get("cfg", {}).get("task") != "motion":
        raise ValueError("Expected a motion-tracking checkpoint")
    names = list(ckpt["joint_names"])
    if names != control["joint_names"]:
        raise ValueError("Checkpoint and exported training robot joint order differ")
    library = MotionLibrary(
        motion_path(ckpt, args.checkpoint, args.motion_file), names, device=args.device
    )
    if (
        not (args.motion_file or os.environ.get("G1_MOTION_FILE"))
        and library.signature != ckpt["motion_metadata"]["signature"]
    ):
        raise ValueError("Reference dataset differs from the trained checkpoint")
    actor = Actor(PolicySpec.from_dict(ckpt["policy_spec"])).to(args.device).eval()
    actor.load_state_dict(ckpt["actor"])
    training_signature = ckpt["motion_metadata"]["signature"]
    enable_extension("isaacsim.asset.importer.mjcf")
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=control["physics_dt"], device=args.device)
    )
    temporary_logs = []
    for handler in logging.getLogger().handlers[:]:
        filename = getattr(handler, "baseFilename", "")
        if filename and Path(filename).is_relative_to(
            Path(tempfile.gettempdir()) / "isaaclab/logs"
        ):
            temporary_logs.append(Path(filename))
            handler.close()
    configure_logging(log_dir=str(Path(args.output).resolve().parent / "isaac_logs"))
    for path in temporary_logs:
        path.unlink(missing_ok=True)
    floor = sim_utils.CuboidCfg(
        size=(100.0, 100.0, 0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6, dynamic_friction=0.6
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.18, 0.22)),
    )
    floor.func("/World/Floor", floor, translation=(0, 0, -0.05))
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85))
    light.func("/World/Light", light)
    spawn = sim_utils.MjcfFileCfg(
        asset_path=control["mjcf"],
        usd_dir=control["usd_dir"],
        usd_file_name="g1.usd",
        fix_base=False,
        make_instanceable=False,
        import_sites=False,
        self_collision=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True
        ),
    )
    pd = IdealPDActuatorCfg(
        joint_names_expr=names,
        stiffness=control["stiffness"],
        damping=control["damping"],
        effort_limit=control["effort"],
        effort_limit_sim=control["effort"],
        armature=control["armature"],
    )
    cfg = ArticulationCfg(
        prim_path="/World/PolicyRobot",
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.8), joint_pos=control["default_q"]
        ),
        actuators={"motors": pd},
    )
    robot = Articulation(cfg)
    ghost_cfg = cfg.copy()
    ghost_cfg.prim_path = "/World/ReferenceRobot"
    ghost_cfg.spawn = sim_utils.UsdFileCfg(
        usd_path=str(Path(control["usd_dir"]) / "g1.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    )
    ghost = Articulation(ghost_cfg)
    from pxr import UsdPhysics, PhysxSchema, UsdShade
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    # Remove the importer's dummy articulation; retain the floating pelvis.
    for prim in list(stage.TraverseAll()):
        if prim.GetPath().pathString.startswith(
            ("/World/PolicyRobot", "/World/ReferenceRobot")
        ):
            if prim.IsInstance():
                prim.SetInstanceable(False)
            if prim.GetName() == "worldBody":
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    sim_utils.modify_collision_properties(
        "/World/ReferenceRobot",
        sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    )
    reference_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 0.95))
    reference_material.func("/World/ReferenceMaterial", reference_material)
    UsdShade.MaterialBindingAPI.Apply(
        stage.GetPrimAtPath("/World/ReferenceRobot")
    ).Bind(
        UsdShade.Material(stage.GetPrimAtPath("/World/ReferenceMaterial")),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
    sim.set_camera_view([3.5, -5.0, 2.5], [0.0, 1.0, 0.8])
    sim.reset()
    ids, found = robot.find_joints(names, preserve_order=True)
    gids, gfound = ghost.find_joints(names, preserve_order=True)
    if found != names or gfound != names or len(ids) != 29:
        raise ValueError("Imported robot joint names differ from the checkpoint")

    def tensor(key):
        return torch.tensor([[control[key][n] for n in names]], device=args.device)

    q0, scale = tensor("default_q"), tensor("action_scale")
    lower, upper = tensor("target_min"), tensor("target_max")
    offset = torch.tensor([[0.0, 2.0, 0.0]], device=args.device)
    index = library.starts[:1].clone()
    previous = torch.zeros(1, 29, device=args.device)
    failures = completed = clip = step = 0
    error_sum = 0.0
    history = None
    label = None
    if not args.headless:
        import omni.ui as ui

        window = ui.Window(
            "G1 SuperMimic: actual policy / blue reference", width=540, height=125
        )
        with window.frame:
            with ui.VStack():
                ui.Label("Gray: policy in PhysX | Blue: reference, offset 2 m")
                ui.Label(
                    "Training backend: MuJoCo; this window is sim-to-sim evaluation."
                )
                label = ui.Label("Loading...")

    def write_reference(art, joint_ids, ref, shift):
        state = art.data.default_root_state.clone()
        state[:, :3] = ref["state"][:, :3] + shift
        from isaaclab.utils.math import quat_from_matrix

        state[:, 3:7] = quat_from_matrix(library.data["rot"][index, 0])
        state[:, 7:10], state[:, 10:13] = ref["state"][:, 9:12], ref["state"][:, 12:15]
        art.write_root_link_state_to_sim(state)
        art.write_joint_state_to_sim(
            ref["state"][:, 15:44], ref["state"][:, 44:], joint_ids=joint_ids
        )
        art.reset()

    def current_state():
        d = robot.data
        return torch.cat(
            (
                d.root_link_pos_w,
                rotation6d(quat_matrix(d.root_link_quat_w)),
                d.root_link_lin_vel_w,
                d.root_link_ang_vel_w,
                d.joint_pos[:, ids],
                d.joint_vel[:, ids],
            ),
            -1,
        )

    def reset():
        nonlocal history
        previous.zero_()
        ref = library.frame(index)
        write_reference(robot, ids, ref, torch.zeros_like(offset))
        write_reference(ghost, gids, ref, offset)
        sim.forward()
        robot.update(0)
        ghost.update(0)
        history = reset_history(
            proprioception(ref["state"], previous, q0), actor.spec.history
        )

    reset()
    torch.testing.assert_close(
        current_state(), library.frame(index)["state"], atol=2e-4, rtol=2e-4
    )
    tracked, _ = robot.find_bodies(list(library.tracked_names), preserve_order=True)
    pose_error = float(
        (robot.data.body_link_pos_w[:, tracked] - library.frame(index)["pos"])
        .abs()
        .max()
    )
    rotation_error = float(
        (
            quat_matrix(robot.data.body_link_quat_w[:, tracked])
            - library.frame(index)["rot"]
        )
        .abs()
        .max()
    )
    if max(pose_error, rotation_error) > 2e-4:
        raise ValueError(
            f"Imported robot/reference FK mismatch: {pose_error}, {rotation_error}"
        )
    last_view = time.monotonic()
    last_checkpoint_check = 0.0
    print(
        json.dumps(
            {
                "isaac_rollout": "ready",
                "checkpoint": str(args.checkpoint),
                "iteration": ckpt["iteration"],
                "joint_mapping": found,
                "reset_fk_position_error": pose_error,
                "reset_fk_rotation_error": rotation_error,
            }
        ),
        flush=True,
    )
    with torch.inference_mode():
        while app.is_running() and (args.steps == 0 or step < args.steps):
            start = time.monotonic()
            if not sim.is_playing():
                sim.render()
                time.sleep(0.01)
                continue
            if (
                args.final_checkpoint or args.checkpoint_dir
            ) and start - last_checkpoint_check > 5:
                last_checkpoint_check = start
                sources = args.checkpoint_dir + (
                    [args.final_checkpoint] if args.final_checkpoint else []
                )
                final_path = latest_checkpoint(sources, ckpt["iteration"])
                if final_path:
                    updated = torch.load(
                        final_path, map_location=args.device, weights_only=False
                    )
                    if (
                        list(updated["joint_names"]) != names
                        or updated["motion_metadata"]["signature"] != training_signature
                        or updated["policy_spec"] != ckpt["policy_spec"]
                    ):
                        raise ValueError(
                            "New checkpoint policy/reference contract changed"
                        )
                    actor.load_state_dict(updated["actor"])
                    ckpt = {
                        "iteration": updated["iteration"],
                        "policy_spec": updated["policy_spec"],
                    }
                    del updated
                    args.checkpoint = str(final_path)
                    failures = completed = step = clip = 0
                    error_sum = 0.0
                    index = library.starts[:1].clone()
                    reset()
                    print(
                        json.dumps(
                            {
                                "isaac_rollout": "checkpoint_loaded",
                                "checkpoint": str(args.checkpoint),
                                "iteration": ckpt["iteration"],
                            }
                        ),
                        flush=True,
                    )
            s = current_state()
            goal = library.goal(index, s)
            action = actor(history, goal, "motion")
            target = torch.maximum(torch.minimum(q0 + scale * action, upper), lower)
            robot.set_joint_position_target(target, joint_ids=ids)
            next_index = library.advance(index)
            ref = library.frame(next_index)
            index = next_index
            substeps = round(control["control_dt"] / control["physics_dt"])
            for substep in range(substeps):
                write_reference(ghost, gids, ref, offset)
                robot.write_data_to_sim()
                sim.step(render=substep == substeps - 1)
                robot.update(control["physics_dt"])
                ghost.update(control["physics_dt"])
            nxt = current_state()
            error_sum += float(
                torch.linalg.vector_norm(nxt[:, :3] - ref["state"][:, :3])
            )
            failed = bool(hard_failure(nxt, ref["state"]).item())
            ended = bool((index >= library.end_at[index]).item())
            previous.copy_(action)
            history = push_history(history, proprioception(nxt, previous, q0))
            step += 1
            if step % 250 == 0:
                progress = {
                    "backend": "Isaac Lab / PhysX",
                    "stage": "running",
                    "checkpoint": str(args.checkpoint),
                    "iteration": ckpt["iteration"],
                    "steps": step,
                    "reference_failures": failures,
                    "reference_completions": completed,
                    "mean_root_position_error_m": error_sum / step,
                }
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(progress, indent=2) + "\n")
                print(json.dumps(progress), flush=True)
            if failed or ended:
                failures += int(failed)
                completed += int(ended and not failed)
                clip = (clip + 1) % len(library.starts)
                index = library.starts[clip : clip + 1].clone()
                reset()
            if label is not None:
                label.text = f"Iter {ckpt['iteration']} | step {step} | clip {clip + 1}/{len(library.starts)} | failures {failures} | completed {completed}"
            if time.monotonic() - last_view > 0.5:
                center = robot.data.root_link_pos_w[0].cpu().numpy().copy()
                center[2] = 0.8
                sim.set_camera_view(center + [3.5, -5.0, 2.0], center + [0.0, 0.8, 0.0])
                last_view = time.monotonic()
            if args.realtime:
                time.sleep(max(0, control["control_dt"] - (time.monotonic() - start)))
    result = {
        "backend": "Isaac Lab / PhysX",
        "checkpoint": str(args.checkpoint),
        "steps": step,
        "reference_failures": failures,
        "reference_completions": completed,
        "mean_root_position_error_m": error_sum / max(step, 1),
        "motion_signature": library.signature,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
