"""Full reference tracking reward, reference-state reset and timed motion command.

Rewards follow BeyondMimic's relative body pose/global body velocity convention,
with pelvis as root per SuperMimic Appendix A. Root pose is tracked globally.
Self-collision reward penalty is absent; physical self-collisions remain enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers import CommandTerm, CommandTermCfg

from .kinematics import (
    G1Kinematics,
    TRACKED_BODIES,
    apply,
    matrix6d,
    quat_matrix,
    read_state,
    rotation_error_squared,
    yaw_matrix,
    proprioception,
)
from .motion import MotionLibrary


WEIGHTS = {
    "body_position": 1.0,
    "body_orientation": 1.0,
    "body_linear_velocity": 1.0,
    "body_angular_velocity": 1.0,
    "root_position": 0.5,
    "root_orientation": 0.5,
    "action_smoothness": -0.1,
    "joint_limits": -2.0,
}


def tracking_terms(state, bodies, reference, action, previous, limits):
    r = matrix6d(state[:, 3:9])
    desired = reference["state"]
    rr = matrix6d(desired[:, 3:9])
    # Match BeyondMimic: align reference body poses by root XY and heading,
    # preserving reference height. Global root rewards still anchor the trajectory.
    delta = yaw_matrix(r) @ yaw_matrix(rr).transpose(-1, -2)
    anchor = torch.cat((state[:, :2], desired[:, 2:3]), -1)
    target_pos = anchor[:, None] + apply(
        delta[:, None], reference["pos"] - desired[:, None, :3]
    )
    target_rot = delta[:, None] @ reference["rot"]
    q = state[:, 15:44]
    return {
        "body_position": torch.exp(
            -(bodies["pos"] - target_pos).square().sum(-1).mean(-1) / 0.3**2
        ),
        "body_orientation": torch.exp(
            -rotation_error_squared(target_rot, bodies["rot"]).mean(-1) / 0.4**2
        ),
        "body_linear_velocity": torch.exp(
            -(bodies["lin_vel"] - reference["lin_vel"]).square().sum(-1).mean(-1)
            / 1.0**2
        ),
        "body_angular_velocity": torch.exp(
            -(bodies["ang_vel"] - reference["ang_vel"]).square().sum(-1).mean(-1)
            / 3.14**2
        ),
        "root_position": torch.exp(
            -(state[:, :3] - desired[:, :3]).square().sum(-1) / 0.3**2
        ),
        "root_orientation": torch.exp(-rotation_error_squared(rr, r) / 0.4**2),
        "action_smoothness": (action - previous).square().sum(-1),
        "joint_limits": (
            (limits[..., 0] - q).clamp(min=0) + (q - limits[..., 1]).clamp(min=0)
        ).sum(-1),
    }


def total_reward(terms):
    return sum(WEIGHTS[k] * v for k, v in terms.items())


def deviation(state, reference_state):
    height = (state[:, 2] - reference_state[:, 2]).abs()
    # z-components of projected gravity in reference/robot root frames.
    tilt = (
        matrix6d(state[:, 3:9])[:, 2, 2] - matrix6d(reference_state[:, 3:9])[:, 2, 2]
    ).abs()
    return height, tilt


def hard_failure(state, reference_state):
    height, tilt = deviation(state, reference_state)
    return (height > 0.25) | (tilt > 0.8)


def soft_failure(state, reference_state):
    height, tilt = deviation(state, reference_state)
    a = torch.sigmoid((height - 0.25) / 0.025)
    b = torch.sigmoid((tilt - 0.8) / 0.025)
    return 1 - (1 - a) * (1 - b)


@dataclass(kw_only=True)
class TrackingCommandCfg(CommandTermCfg):
    motion_file: str
    reset_mode: str = "uniform"
    allow_test_motion: bool = False

    def build(self, env):
        return TrackingCommand(self, env)


class TrackingCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.library = MotionLibrary(
            cfg.motion_file,
            self.robot.joint_names,
            dt=env.step_dt,
            device=env.device,
            allow_test_motion=cfg.allow_test_motion,
        )
        self.kinematics = G1Kinematics.from_env(env)
        self.indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.body_ids = self.robot.find_bodies(TRACKED_BODIES, preserve_order=True)[0]
        self._reward_step = -1
        self._refresh_step = -1
        self._reward_terms = None

    @property
    def command(self):
        return self.library.goal(self.indices, read_state(self._env)).flatten(1)

    def _update_metrics(self):
        pass

    def _update_command(self, env_ids=None):
        pass

    def _resample_command(self, env_ids):
        indices = self.library.sample(len(env_ids), self.cfg.reset_mode)
        self.reset_to_frames(env_ids, indices)

    def reset_to_frames(self, env_ids, indices):
        ref = self.library.frame(indices)
        s = ref["state"]
        self.indices[env_ids] = indices
        root = torch.cat(
            (
                s[:, :3] + self._env.scene.env_origins[env_ids],
                ref["root_quat"],
                s[:, 9:15],
            ),
            -1,
        )
        self.robot.write_joint_state_to_sim(s[:, 15:44], s[:, 44:73], env_ids=env_ids)
        self.robot.write_root_state_to_sim(root, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)
        self._reward_step = -1
        self._refresh_step = -1

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        return {}

    def compute(self, dt, env_ids=None):
        # Auto-reset envs receive dt=0. Never advance them past their reset frame.
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids
        )
        ticks = torch.as_tensor(dt, device=self.device)
        if ticks.ndim:
            ids = ids[ticks[ids] > 0]
        elif ticks <= 0:
            return
        self.indices[ids] = self.library.advance(self.indices[ids])

    def actual_bodies(self):
        d = self.robot.data
        ids = self.body_ids
        return {
            "pos": d.body_link_pos_w[:, ids] - self._env.scene.env_origins[:, None],
            "rot": quat_matrix(d.body_link_quat_w[:, ids]),
            "lin_vel": d.body_link_lin_vel_w[:, ids],
            "ang_vel": d.body_link_ang_vel_w[:, ids],
        }

    def refresh_kinematics(self):
        # Refresh the post-control state; mjlab's fast path can leave stale transforms.
        if self._refresh_step != self._env._sim_step_counter:
            self._env.sim.forward()
            self._refresh_step = self._env._sim_step_counter

    def reward_terms_next(self):
        self.refresh_kinematics()
        step = self._env._sim_step_counter
        if self._reward_step != step:
            self.last_reward_indices = self.library.advance(self.indices).clone()
            self._reward_terms = tracking_terms(
                read_state(self._env),
                self.actual_bodies(),
                self.library.frame(self.last_reward_indices),
                self._env.action_manager.action,
                self._env.action_manager.prev_action,
                self.robot.data.soft_joint_pos_limits,
            )
            self._reward_step = step
        return self._reward_terms


def simulator_term(env, term_name):
    return env.command_manager.get_term("motion").reward_terms_next()[term_name]


def simulator_failed(env):
    term = env.command_manager.get_term("motion")
    term.refresh_kinematics()
    ref = term.library.data["state"][term.library.advance(term.indices)]
    return hard_failure(read_state(env), ref)


def simulator_finished(env):
    term = env.command_manager.get_term("motion")
    return term.library.ending(term.indices)


def actor_observation(env):
    return proprioception(
        read_state(env),
        env.action_manager.action,
        env.scene["robot"].data.default_joint_pos,
    )


def configure(env_cfg, motion_file, reset_mode="uniform", allow_test_motion=False):
    from mjlab.managers import (
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        TerminationTermCfg,
    )

    if not motion_file:
        raise ValueError(
            "--task motion requires --motion-file (NPZ or JSON clip manifest)"
        )
    env_cfg.commands = {
        "motion": TrackingCommandCfg(
            motion_file=motion_file,
            reset_mode=reset_mode,
            allow_test_motion=allow_test_motion,
            resampling_time_range=(1e9, 1e9),
        )
    }
    env_cfg.events = {}
    # Episodes follow clip duration, not the velocity baseline's 10 s limit.
    env_cfg.episode_length_s = int(1e9)
    env_cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"frame": ObservationTermCfg(func=actor_observation)},
            concatenate_terms=True,
            enable_corruption=False,
        )
    }
    env_cfg.rewards = {
        k: RewardTermCfg(func=simulator_term, weight=w, params={"term_name": k})
        for k, w in WEIGHTS.items()
    }
    env_cfg.terminations = {
        "time_out": env_cfg.terminations["time_out"],
        "reference_end": TerminationTermCfg(func=simulator_finished, time_out=True),
        "reference_deviation": TerminationTermCfg(func=simulator_failed),
    }
    return env_cfg
