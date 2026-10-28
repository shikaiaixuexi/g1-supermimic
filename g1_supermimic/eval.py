"""Evaluate a G1 checkpoint in actual simulation, optionally render a video."""

import argparse
import json
import math
import os
from pathlib import Path

import torch

from .task import make_env_cfg, robot_state
from .policy import Actor, PolicySpec, reset_history, push_history
from .config import resolve_path, motion_path


def main():
    from mjlab.envs import ManagerBasedRlEnv

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=resolve_path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--video", type=resolve_path)
    p.add_argument("--motion-file", type=resolve_path)
    p.add_argument("--reference-reset", choices=["start", "uniform"], default="start")
    p.add_argument("--allow-test-motion", type=int, default=0)
    p.add_argument("--output", default="logs/g1/eval.json", type=resolve_path)
    args = p.parse_args()
    if args.num_envs < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        p.error("--num-envs and --seconds must be positive")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if ckpt.get("format") not in (
        "g1-supermimic-v2-paper-policy",
        "g1-supermimic-v3-motion-tracking",
    ):
        raise ValueError(
            "Expected a v2/v3 checkpoint; earlier single-frame checkpoints require retraining"
        )
    is_motion = ckpt["cfg"]["task"] == "motion"
    motion_file = (
        motion_path(ckpt, args.checkpoint, args.motion_file) if is_motion else ""
    )
    cfg = make_env_cfg(
        ckpt["cfg"]["task"],
        args.num_envs,
        args.seed,
        motion_file,
        args.reference_reset,
        bool(args.allow_test_motion),
    )
    cfg.viewer.width, cfg.viewer.height = 960, 720
    env = ManagerBasedRlEnv(
        cfg, device=args.device, render_mode="rgb_array" if args.video else None
    )
    writer = None
    try:
        if ckpt["joint_names"] != env.scene["robot"].joint_names:
            raise ValueError("Checkpoint joint order mismatch")
        motion = env.command_manager.get_term("motion") if is_motion else None
        if (
            motion
            and not (args.motion_file or os.environ.get("G1_MOTION_FILE"))
            and ckpt["motion_metadata"]["signature"] != motion.library.signature
        ):
            raise ValueError(
                "Evaluation motion signature differs from training checkpoint"
            )
        actor = Actor(PolicySpec.from_dict(ckpt["policy_spec"])).to(args.device)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        obs, _ = env.reset()
        history = reset_history(obs["actor"], actor.spec.history)
        steps = max(1, round(args.seconds / env.step_dt))
        falls = total_r = velocity_error = height = completed = 0.0
        root_error = body_error = valid_steps = 0.0
        if args.video:
            import imageio.v2 as imageio

            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video, fps=round(1 / env.step_dt))
        with torch.no_grad():
            for _ in range(steps):
                if motion:
                    from .kinematics import read_state as tracking_state

                    goal = motion.library.goal(motion.indices, tracking_state(env))
                else:
                    goal = env.command_manager.get_command("twist")
                obs, r, term, timeout, _ = env.step(
                    actor(history, goal, "motion" if motion else "velocity")
                )
                history = push_history(history, obs["actor"], term | timeout)
                s = tracking_state(env) if motion else robot_state(env)
                cmd = env.command_manager.get_command("twist") if not motion else None
                falls += int(term.sum())
                total_r += float(r.sum())
                if motion:
                    reference = motion.library.frame(motion.indices)
                    keep = ~(term | timeout)
                    if keep.any():
                        root_error += float(
                            (s[keep, :3] - reference["state"][keep, :3])
                            .norm(dim=-1)
                            .sum()
                        )
                        body_error += float(
                            (
                                motion.actual_bodies()["pos"][keep]
                                - reference["pos"][keep]
                            )
                            .norm(dim=-1)
                            .mean(-1)
                            .sum()
                        )
                        valid_steps += int(keep.sum())
                    completed += int(
                        (
                            env.termination_manager.get_term("reference_end") & ~term
                        ).sum()
                    )
                    height += float(s[:, 2].mean())
                else:
                    velocity_error += float(
                        (s[:, 4:6] - cmd[:, :2]).norm(dim=-1).mean()
                    )
                    height += float(s[:, 0].mean())
                if writer:
                    writer.append_data(env.render())
        result = {
            "checkpoint": str(args.checkpoint),
            "task": ckpt["cfg"]["task"],
            "seed": args.seed,
            "num_envs": args.num_envs,
            "seconds_per_env": steps * env.step_dt,
            "falls": falls,
            "falls_per_robot_min": falls / (args.num_envs * steps * env.step_dt / 60),
            "reward_per_robot_second": total_r / (args.num_envs * steps * env.step_dt),
            "mean_planar_velocity_error": velocity_error / steps,
            "mean_base_height": height / steps,
        }
        if motion:
            result.pop("mean_planar_velocity_error")
            result["reference_failures"] = result.pop("falls")
            result["reference_failures_per_robot_min"] = result.pop(
                "falls_per_robot_min"
            )
            result.update(
                {
                    "training_motion_signature": ckpt["motion_metadata"]["signature"],
                    "mean_root_position_error_m": root_error / max(1, valid_steps),
                    "mean_global_body_position_error_m": body_error
                    / max(1, valid_steps),
                    "reference_completions": completed,
                    "motion": motion.library.metadata(),
                }
            )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if writer:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
