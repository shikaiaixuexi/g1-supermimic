"""Shared simulator/imagination interfaces. Flat ground, fixed physical parameters.

The actor has no privileged base velocity. The learned dynamics and critic do.
G1 joint layout, actuator configuration and flat-ground tasks are defined independently.
"""

from __future__ import annotations

import torch


def yaw_inverse(v, yaw):
    c, s = yaw.cos(), yaw.sin()
    return torch.stack(
        (c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]), -1
    )


def robot_state(env):
    d = env.scene["robot"].data
    return torch.cat(
        (
            d.root_link_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3],
            d.projected_gravity_b,
            yaw_inverse(d.root_link_lin_vel_w, d.heading_w),
            d.root_link_ang_vel_b,
            d.joint_pos - d.default_joint_pos,
            d.joint_vel,
        ),
        -1,
    )


def observation(s, previous_action, noise=False):
    # Paper Table A.1 order: q-q0, qdot, body angular velocity, gravity, a_prev.
    frame = torch.cat((s[:, 10:], s[:, 7:10], s[:, 1:4], previous_action), -1)
    if noise:
        nj = previous_action.shape[-1]
        widths = frame.new_tensor(
            [0.01] * nj + [0.5] * nj + [0.2] * 3 + [0.05] * 3 + [0.0] * nj
        )
        frame = frame + (2 * torch.rand_like(frame) - 1) * widths
    return frame


def actor_observation(env):
    return observation(robot_state(env), env.action_manager.action)


def reward(s, action, previous_action, command, default_q, limits):
    n = action.shape[-1]
    q_rel = s[:, 10 : 10 + n]
    q = q_rel + default_q
    tracking = 2.0 * torch.exp(-((s[:, 4:6] - command[:, :2]) ** 2).sum(-1) / 0.25)
    # Project body angular velocity onto world vertical for yaw-rate tracking.
    yaw_rate = -(s[:, 7:10] * s[:, 1:4]).sum(-1)
    turning = torch.exp(-(yaw_rate - command[:, 2]).square() / 0.5)
    upright = torch.exp(-s[:, 1:3].square().sum(-1) / 0.2)
    height = torch.exp(-(s[:, 0] - 0.76).square() / 0.01)
    pose = q_rel.square().mean(-1)
    limit = ((limits[..., 0] - q).clamp(min=0) + (q - limits[..., 1]).clamp(min=0)).sum(
        -1
    )
    smooth = (action - previous_action).square().sum(-1)
    return tracking + turning + upright + height - 0.5 * pose - limit - 0.05 * smooth


def simulator_reward(env):
    d = env.scene["robot"].data
    return reward(
        robot_state(env),
        env.action_manager.action,
        env.action_manager.prev_action,
        env.command_manager.get_command("twist"),
        d.default_joint_pos,
        d.soft_joint_pos_limits,
    )


def fallen(env):
    s = robot_state(env)
    return (s[:, 0] < 0.45) | (s[:, 3] > -0.5)


def failure_probability(s):
    low = torch.sigmoid((0.45 - s[:, 0]) / 0.025)
    tilt = torch.sigmoid((s[:, 3] + 0.5) / 0.025)
    return 1 - (1 - low) * (1 - tilt)


def make_env_cfg(
    task="stand",
    num_envs=64,
    seed=0,
    motion_file="",
    reference_reset="uniform",
    allow_test_motion=False,
):
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
    from mjlab.managers import (
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        TerminationTermCfg,
    )

    if task == "motion":
        from .tracking import configure

        base = make_env_cfg("stand", num_envs, seed)
        return configure(base, motion_file, reference_reset, allow_test_motion)
    cfg = unitree_g1_flat_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.events = {
        k: v for k, v in cfg.events.items() if k in ("reset_base", "reset_robot_joints")
    }
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.scene.sensors = ()
    cfg.episode_length_s = 10.0
    cmd = cfg.commands["twist"]
    cmd.heading_command = False
    cmd.ranges.heading = None
    cmd.rel_standing_envs = 1.0 if task == "stand" else 0.1
    cmd.rel_heading_envs = 0.0
    cmd.rel_forward_envs = 0.0
    cmd.rel_world_envs = 0.0
    # Held constant during each 10 s episode and all 32-step imagined rollouts.
    cmd.resampling_time_range = (1000.0, 1000.0)
    cmd.ranges.lin_vel_x = (0.0, 0.0) if task == "stand" else (0.0, 0.5)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0) if task == "stand" else (-0.3, 0.3)
    cmd.debug_vis = False
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"frame": ObservationTermCfg(func=actor_observation)},
            concatenate_terms=True,
            enable_corruption=False,
        )
    }
    cfg.rewards = {"task": RewardTermCfg(func=simulator_reward, weight=1.0)}
    cfg.terminations = {
        "time_out": cfg.terminations["time_out"],
        "fell_over": TerminationTermCfg(func=fallen),
    }
    return cfg
