"""uv run python -m g1_supermimic.train --task stand --num-envs 64."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn

from .learning import Critic, Replay, WorldModel, td_lambda
from .task import failure_probability, make_env_cfg, observation, reward, robot_state
from .policy import Actor, PolicySpec, reset_history, push_history
from .config import resolve_path, training_defaults


@dataclass
class TrainCfg:
    device: str = "cuda:0"
    task: str = "stand"
    num_envs: int = 64
    iterations: int = 2000
    buffer_len: int = 256
    warmup_iterations: int = 300
    collect_steps: int = 4
    action_noise: float = 0.1
    sensor_noise: int = 1
    reference_frame_dim: int = 68
    freeze_decoder: int = 0
    motion_file: str = ""
    reference_reset: str = "uniform"
    allow_test_motion: int = 0
    wm_window: int = 8
    wm_batch: int = 256
    wm_updates: int = 2
    wm_width: int = 2048
    horizon: int = 32
    imag_batch: int = 256
    critic_epochs: int = 5
    lr: float = 2e-4
    actor_lr_scale: float = 0.3
    actor_ramp: int = 500
    seed: int = 0
    save_interval: int = 500
    log_interval: int = 20
    run_dir: str = "logs/g1/stand"
    resume: str = ""


def step_optimizer(loss, module, optimizer, element_clip=None):
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite {type(module).__name__} loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(module.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise FloatingPointError(f"Non-finite {type(module).__name__} gradient")
    if element_clip is not None:
        for p in module.parameters():
            if p.grad is not None:
                p.grad.clamp_(-element_clip, element_clip)
    optimizer.step()
    return float(norm)


class Trainer:
    def __init__(self, cfg):
        from mjlab.envs import ManagerBasedRlEnv

        if min(cfg.iterations, cfg.num_envs, cfg.save_interval, cfg.log_interval) < 1:
            raise ValueError(
                "Iterations, environment count and logging/save intervals must be positive"
            )
        if cfg.task not in ("stand", "velocity", "motion"):
            raise ValueError("task must be stand, velocity or motion")
        if cfg.motion_file and cfg.task != "motion":
            raise ValueError("--motion-file requires --task motion")
        if cfg.buffer_len < cfg.wm_window:
            raise ValueError("buffer_len must be >= wm_window")
        if min(cfg.collect_steps, cfg.wm_updates, cfg.horizon, cfg.num_envs) < 1:
            raise ValueError("steps, updates, horizon and env count must be positive")
        if cfg.freeze_decoder and not cfg.resume:
            raise ValueError("Decoder freezing requires a trained policy checkpoint")
        if cfg.resume:
            metadata = torch.load(cfg.resume, map_location="cpu", weights_only=False)
            if metadata.get("format") not in (
                "g1-supermimic-v2-paper-policy",
                "g1-supermimic-v3-motion-tracking",
            ):
                raise ValueError(
                    "Checkpoint uses the earlier policy interface; start a fresh current-policy run"
                )
            if cfg.iterations <= metadata["iteration"]:
                raise ValueError("--iterations must exceed the checkpoint iteration")
            if cfg.freeze_decoder and "motion" not in metadata.get(
                "trained_interfaces", []
            ):
                raise ValueError(
                    "Paper downstream adaptation requires a motion-trained shared decoder"
                )
            if cfg.task == "motion" and not cfg.motion_file:
                cfg.motion_file = metadata["cfg"].get("motion_file", "")
            cfg.wm_width = metadata["cfg"]["wm_width"]
            cfg.reference_frame_dim = metadata["policy_spec"]["reference_frame_dim"]
        self.interface = "motion" if cfg.task == "motion" else "velocity"
        if cfg.task == "motion" and not cfg.motion_file:
            raise ValueError("--task motion requires --motion-file")
        self.trained_interfaces = (
            sorted(set(metadata.get("trained_interfaces", [])) | {self.interface})
            if cfg.resume
            else [self.interface]
        )
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.env = ManagerBasedRlEnv(
            make_env_cfg(
                cfg.task,
                cfg.num_envs,
                cfg.seed,
                cfg.motion_file,
                cfg.reference_reset,
                bool(cfg.allow_test_motion),
            ),
            device=cfg.device,
        )
        self.motion = (
            self.env.command_manager.get_term("motion")
            if self.interface == "motion"
            else None
        )
        self.state_schema = (
            "g1-root-rot6d-joints-v1" if self.motion else "g1-flat-68-v1"
        )
        self.device = self.env.device
        self.obs, _ = self.env.reset()
        self.s = self.read_state()
        self.default_q = self.env.scene["robot"].data.default_joint_pos[:1]
        if self.motion:
            cfg.reference_frame_dim = self.motion.library.feature_dim
        self.nj = self.env.action_manager.total_action_dim
        if self.nj != self.env.scene["robot"].data.joint_pos.shape[-1]:
            raise ValueError("Baseline requires all G1 joints to be controlled")
        action_term = self.env.action_manager.get_term("joint_pos")
        if list(action_term._target_names) != list(self.env.scene["robot"].joint_names):
            raise ValueError("Action order differs from state joint order")
        self.sd, self.od = self.s.shape[-1], self.obs["actor"].shape[-1]
        self.spec = PolicySpec(
            joints=self.nj, reference_frame_dim=cfg.reference_frame_dim
        )
        if self.od != self.spec.frame_dim:
            raise ValueError("Environment proprioception differs from policy schema")
        self.actor = Actor(self.spec).to(self.device)
        self.actor.train_interface(self.interface, train_decoder=not cfg.freeze_decoder)
        self.wm = WorldModel(
            self.sd, self.nj, cfg.wm_width, project_rotation=bool(self.motion)
        ).to(self.device)
        self.critic = Critic(
            self.sd + self.nj + (10 * cfg.reference_frame_dim + 1 if self.motion else 3)
        ).to(self.device)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
        self.opt_actor = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95)
        )
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.buf = Replay(
            cfg.num_envs,
            cfg.buffer_len,
            self.sd,
            self.nj,
            self.device,
            history_shape=(self.spec.history, self.od),
            command_dim=0 if self.motion else 3,
            reference_indices=bool(self.motion),
        )
        self.previous = torch.zeros(cfg.num_envs, self.nj, device=self.device)
        self.history = reset_history(
            self.frame(self.s, self.previous), self.spec.history
        )
        self.it = self.start = self.env_steps = self.falls = 0
        self.run_dir = Path(cfg.run_dir)
        if (self.run_dir / "metrics.jsonl").exists() and not cfg.resume:
            raise FileExistsError(f"Choose a fresh run-dir: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if cfg.resume:
            self.load(cfg.resume)
        (self.run_dir / "config.json").write_text(
            json.dumps(asdict(cfg), indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "joints": self.nj,
                    "state_dim": self.sd,
                    "actor_frame_dim": self.od,
                    "history_frames": self.spec.history,
                    "actor_history_dim": self.od * self.spec.history,
                    "latent_dim": self.spec.latent_dim,
                    "joint_names": self.env.scene["robot"].joint_names,
                    "step_dt": self.env.step_dt,
                    "state_schema": self.state_schema,
                    "motion": self.motion.library.metadata() if self.motion else None,
                }
            ),
            flush=True,
        )

    def read_state(self):
        if self.motion:
            from .kinematics import read_state

            return read_state(self.env)
        return robot_state(self.env)

    def frame(self, state, previous):
        if self.motion:
            from .kinematics import proprioception

            return proprioception(
                state, previous, self.default_q, noise=bool(self.cfg.sensor_noise)
            )
        return observation(state, previous, noise=bool(self.cfg.sensor_noise))

    def goal(self, state, index=None):
        if self.motion:
            return self.motion.library.goal(
                self.motion.indices if index is None else index, state
            )
        return self.env.command_manager.get_command("twist").clone()

    def critic_input(self, state, previous, goal, index=None):
        parts = [state, previous, goal.flatten(1)]
        if self.motion:
            # Remaining clip time disambiguates repeated final goals for the critic.
            remaining = (
                self.motion.library.end_at[index] - index
            ).float() * self.env.step_dt
            parts.append(remaining[:, None])
        return torch.cat(parts, -1)

    @torch.no_grad()
    def collect(self):
        returns = []
        tracking_rewards = {}
        for _ in range(self.cfg.collect_steps):
            index = self.motion.indices.clone() if self.motion else None
            goal = self.goal(self.s, index)
            command = self.s.new_empty(self.cfg.num_envs, 0) if self.motion else goal
            self.actor.update_stats(self.history[:, -1], goal, self.interface)
            a = self.actor(
                self.history, goal, self.interface
            ) + self.cfg.action_noise * torch.randn_like(self.previous)
            obs, r, terminated, truncated, _ = self.env.step(a)
            done = terminated | truncated
            nxt = self.read_state()
            self.buf.add(
                self.s,
                nxt,
                a,
                self.previous,
                command,
                done,
                history=self.history,
                reference_index=index,
            )
            keep = ~done
            if keep.any():
                self.wm.update_stats(
                    self.s[keep], a[keep], self.previous[keep], nxt[keep]
                )
            self.s, self.obs = nxt.clone(), obs
            self.previous = torch.where(done[:, None], torch.zeros_like(a), a)
            frame = self.frame(nxt, self.previous)
            self.history = push_history(self.history, frame, done)
            self.env_steps += self.cfg.num_envs
            self.falls += int(terminated.sum())
            returns.append(float(r.mean()))
            if self.motion:
                from .tracking import WEIGHTS

                for key, value in self.motion._reward_terms.items():
                    tracking_rewards.setdefault("reward/" + key, []).append(
                        float(value.mean()) * WEIGHTS[key] * self.env.step_dt
                    )
        self.last_tracking_rewards = {
            k: sum(v) / len(v) for k, v in tracking_rewards.items()
        }
        return sum(returns) / len(returns)

    def imagine(self):
        seed = self.buf.seeds(self.cfg.imag_batch)
        s, prev, cmd = seed["s"], seed["previous"], seed["command"]
        history = seed["history"].float()
        index = seed.get("reference_index")
        d = self.env.scene["robot"].data
        q0, lim = d.default_joint_pos[:1], d.soft_joint_pos_limits[:1]
        inputs, rewards, continuations = [], [], []
        survival = torch.ones(s.shape[0], device=self.device)
        ret = torch.zeros_like(survival)
        gamma = 0.99
        for t in range(self.cfg.horizon):
            goal = self.motion.library.goal(index, s) if self.motion else cmd
            inputs.append(self.critic_input(s, prev, goal, index))
            a = self.actor(history, goal, self.interface)
            nxt = self.wm(s, a, prev)
            if self.motion:
                from .tracking import tracking_terms, total_reward, soft_failure

                next_index = self.motion.library.advance(index)
                reference = self.motion.library.frame(next_index)
                terms = tracking_terms(
                    nxt, self.motion.kinematics(nxt), reference, a, prev, lim
                )
                r = (
                    total_reward(terms)
                    * self.env.step_dt
                    * (index < self.motion.library.end_at[index]).float()
                )
                cont = (1 - soft_failure(nxt, reference["state"])) * (
                    ~self.motion.library.ending(index)
                ).float()
                index = next_index
            else:
                r = reward(nxt, a, prev, cmd, q0, lim) * self.env.step_dt
                cont = 1 - failure_probability(nxt)
            ret = ret + gamma**t * survival * r
            survival = survival * cont
            rewards.append(r)
            continuations.append(cont)
            history = push_history(history, self.frame(nxt, a))
            s, prev = nxt, a
        goal = self.motion.library.goal(index, s) if self.motion else cmd
        inputs.append(self.critic_input(s, prev, goal, index))
        ret = ret + gamma**self.cfg.horizon * survival * self.critic(inputs[-1])
        return ret, {
            "inputs": torch.stack(inputs, 1).detach(),
            "rewards": torch.stack(rewards, 1).detach(),
            "continuation": torch.stack(continuations, 1).detach(),
        }

    def improve(self):
        for module in (self.wm, self.critic):
            module.requires_grad_(False)
        try:
            objective, data = self.imagine()
            ramp = min(
                1.0,
                (self.it - self.start - self.cfg.warmup_iterations)
                / max(1, self.cfg.actor_ramp),
            )
            for group in self.opt_actor.param_groups:
                group["lr"] = self.cfg.lr * self.cfg.actor_lr_scale * max(0.0, ramp)
            norm = step_optimizer(
                -objective.mean(), self.actor, self.opt_actor, element_clip=0.1
            )
        finally:
            for module in (self.wm, self.critic):
                module.requires_grad_(True)
        x = data["inputs"]
        self.critic.norm.update(x[:, :-1])
        target = td_lambda(self.critic, x, data["rewards"], data["continuation"])
        flat_x, flat_y = x[:, :-1].reshape(-1, x.shape[-1]), target.reshape(-1)
        for _ in range(self.cfg.critic_epochs):
            for ids in torch.randperm(flat_y.numel(), device=self.device).chunk(4):
                if ids.numel():
                    loss = (self.critic(flat_x[ids]) - flat_y[ids]).square().mean()
                    step_optimizer(loss, self.critic, self.opt_critic)
        return {
            "imagined_return": float(objective.mean().detach()),
            "actor_grad_norm": norm,
            "critic_loss": float(loss.detach()),
        }

    def save(self):
        path = self.run_dir / f"model_{self.it:06d}.pt"
        payload = {
            "format": "g1-supermimic-v3-motion-tracking",
            "state_schema": self.state_schema,
            "critic_input_schema": "tracking-clock-v1"
            if self.motion
            else "velocity-v1",
            "motion_metadata": self.motion.library.metadata() if self.motion else None,
            "policy_spec": self.spec.to_dict(),
            "trained_interfaces": self.trained_interfaces,
            "cfg": asdict(self.cfg),
            "iteration": self.it,
            "state_dim": self.sd,
            "obs_dim": self.od,
            "act_dim": self.nj,
            "joint_names": self.env.scene["robot"].joint_names,
            "env_steps": self.env_steps,
        }
        for name in ("actor", "wm", "critic", "opt_wm", "opt_actor", "opt_critic"):
            payload[name] = getattr(self, name).state_dict()
        torch.save(payload, path)
        return path

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("format") not in (
            "g1-supermimic-v2-paper-policy",
            "g1-supermimic-v3-motion-tracking",
        ):
            raise ValueError("Expected a v2/v3 checkpoint")
        if (
            ckpt["joint_names"] != self.env.scene["robot"].joint_names
            or ckpt["cfg"]["task"] != self.cfg.task
        ):
            raise ValueError("Checkpoint robot joint order/task mismatch")
        if self.motion and ckpt.get("critic_input_schema") != "tracking-clock-v1":
            raise ValueError(
                "Motion checkpoint critic schema is incompatible with finite-clip value targets"
            )
        if self.motion and (
            ckpt.get("state_schema") != self.state_schema
            or ckpt.get("motion_metadata", {}).get("signature")
            != self.motion.library.signature
        ):
            raise ValueError(
                "Checkpoint tracking state schema or motion dataset signature mismatch"
            )
        if PolicySpec.from_dict(ckpt["policy_spec"]) != self.spec:
            raise ValueError("Checkpoint policy specification mismatch")
        for name in ("actor", "wm", "critic", "opt_wm", "opt_actor", "opt_critic"):
            getattr(self, name).load_state_dict(ckpt[name])
        self.start = self.it = ckpt["iteration"]
        self.env_steps = ckpt["env_steps"]
        # Replay is intentionally rebuilt; rewarm WM before actor updates.

    def run(self):
        started = time.time()
        try:
            while self.buf.size < self.cfg.wm_window:
                self.collect()
            for self.it in range(self.start + 1, self.cfg.iterations + 1):
                real_reward = self.collect()
                for _ in range(self.cfg.wm_updates):
                    loss = self.wm.loss(
                        self.buf.windows(self.cfg.wm_batch, self.cfg.wm_window)
                    )
                    step_optimizer(loss, self.wm, self.opt_wm)
                stats = {
                    "iteration": self.it,
                    "env_steps": self.env_steps,
                    ("reference_failures" if self.motion else "falls"): self.falls,
                    "sim_reward": real_reward,
                    "wm_loss": float(loss.detach()),
                }
                stats.update(self.last_tracking_rewards)
                if self.it > self.start + self.cfg.warmup_iterations:
                    stats.update(self.improve())
                if (
                    self.it % self.cfg.log_interval == 0
                    or self.it == self.cfg.iterations
                ):
                    stats["elapsed_s"] = time.time() - started
                    stats["gpu_peak_mb"] = (
                        torch.cuda.max_memory_allocated(self.device) / 2**20
                        if torch.device(self.device).type == "cuda"
                        else 0
                    )
                    with (self.run_dir / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(stats) + "\n")
                    print(json.dumps(stats), flush=True)
                if (
                    self.it % self.cfg.save_interval == 0
                    or self.it == self.cfg.iterations
                ):
                    print(f"saved {self.save()}", flush=True)
        finally:
            self.env.close()


def parse_cfg(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument(
        "--config", help="JSON training profile; CLI options take precedence"
    )
    known, _ = preliminary.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    defaults = TrainCfg()
    values = asdict(defaults)
    if known.config:
        try:
            values = training_defaults(known.config, values)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    for f in fields(defaults):
        v = getattr(defaults, f.name)
        parser.add_argument(
            "--" + f.name.replace("_", "-"), type=type(v), default=values[f.name]
        )
    options = vars(parser.parse_args(argv))
    options.pop("config")
    try:
        for key in ("motion_file", "resume", "run_dir"):
            if options[key]:
                options[key] = str(resolve_path(options[key]))
    except ValueError as error:
        parser.error(str(error))
    return TrainCfg(**options)


def main():
    cfg = parse_cfg()
    torch.set_float32_matmul_precision("high")
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
