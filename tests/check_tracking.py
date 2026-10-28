"""GPU integration checks for reference reset, clock, native/Torch FK and reward."""

import argparse
import json

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from g1_supermimic.task import make_env_cfg
from g1_supermimic.kinematics import read_state, proprioception
from g1_supermimic.tracking import tracking_terms, total_reward, WEIGHTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion-file", required=True)
    p.add_argument("--allow-test-motion", type=int, default=0)
    args = p.parse_args()
    env = ManagerBasedRlEnv(
        make_env_cfg(
            "motion", 8, 9, args.motion_file, "start", bool(args.allow_test_motion)
        ),
        device="cuda:0",
    )
    try:
        obs, _ = env.reset()
        term = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        ids = torch.arange(8, device=env.device)
        root = robot.data.root_link_pose_w.clone()
        angles = torch.randn(8, 3, device=env.device) * 0.4
        quat = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
        root[:, 3:] = quat
        root[:, 2] = 1.2
        lin = torch.randn(8, 3, device=env.device) * 0.2
        ang = torch.randn_like(lin) * 0.3
        q = robot.data.default_joint_pos + torch.randn(8, 29, device=env.device) * 0.03
        qd = torch.randn_like(q) * 0.4
        robot.write_joint_state_to_sim(q, qd)
        robot.write_root_state_to_sim(torch.cat((root, lin, ang), -1))
        env.sim.forward()
        s = read_state(env)
        predicted = term.kinematics(s)
        actual = term.actual_bodies()
        fk_errors = {k: float((predicted[k] - actual[k]).abs().max()) for k in actual}
        assert max(fk_errors.values()) < 4e-5, fk_errors
        torch.testing.assert_close(s[:, 9:12], lin, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(s[:, 12:15], ang, atol=2e-5, rtol=2e-5)

        frames = term.library.starts[0] + torch.arange(8, device=env.device) * 2
        term.reset_to_frames(ids, frames)
        env.scene.write_data_to_sim()
        env.sim.forward()
        reference = term.library.frame(frames)
        s = read_state(env)
        torch.testing.assert_close(s, reference["state"], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            proprioception(
                s, torch.zeros(8, 29, device=env.device), robot.data.default_joint_pos
            ),
            env.observation_manager.compute(update_history=True)["actor"],
            atol=2e-5,
            rtol=2e-5,
        )
        # This also validates that the file's body labels use this G1 model/frame.
        for k, value in term.kinematics(s).items():
            torch.testing.assert_close(value, reference[k], atol=5e-5, rtol=5e-5)

        perturbed = s.clone()
        perturbed[:, 15 + 15] += 0.1
        perturbed.requires_grad_()
        a = torch.zeros(8, 29, device=env.device)
        terms = tracking_terms(
            perturbed,
            term.kinematics(perturbed),
            reference,
            a,
            a,
            robot.data.soft_joint_pos_limits,
        )
        (-total_reward(terms).mean()).backward()
        assert torch.isfinite(perturbed.grad).all()
        assert perturbed.grad[:, 15:44].abs().sum() > 0
        assert set(env.cfg.rewards) == set(WEIGHTS)
        assert "ee_body_pos" not in env.cfg.terminations
        assert env.sim.mj_model.geom_contype[robot.indexing.geom_ids.cpu()].max() > 0

        before = term.indices.clone()
        _, r, failed, timeout, _ = env.step(a)
        keep = ~(failed | timeout)
        assert keep.any()
        torch.testing.assert_close(
            term.indices[keep], term.library.advance(before)[keep]
        )
        torch.testing.assert_close(
            term.last_reward_indices, term.library.advance(before)
        )
        torch.testing.assert_close(
            r, total_reward(term._reward_terms) * env.step_dt, atol=2e-6, rtol=2e-5
        )
        # Full parity at the actual post-step state, not only equal formulas.
        pure = tracking_terms(
            read_state(env),
            term.kinematics(read_state(env)),
            term.library.frame(term.indices),
            env.action_manager.action,
            env.action_manager.prev_action,
            robot.data.soft_joint_pos_limits,
        )
        torch.testing.assert_close(
            r[keep], (total_reward(pure) * env.step_dt)[keep], atol=3e-6, rtol=3e-5
        )

        # Partial resets must not advance the other environments' reference clocks.
        before = term.indices.clone()
        before_state = read_state(env).clone()
        env.reset(env_ids=ids[:1])
        assert torch.isin(term.indices[:1], term.library.starts).all()
        torch.testing.assert_close(term.indices[1:], before[1:])
        torch.testing.assert_close(
            read_state(env)[1:], before_state[1:], atol=2e-5, rtol=2e-5
        )

        # Last reference transition ends an episode; reset stays on a clip start.
        final = term.library.ends[0] - 1
        term.reset_to_frames(ids, final.expand(8))
        env.scene.write_data_to_sim()
        env.sim.forward()
        _, _, _, timeout, _ = env.step(a)
        assert env.termination_manager.get_term("reference_end").all()
        assert timeout.all()
        assert torch.isin(term.indices, term.library.starts).all()
        torch.testing.assert_close(
            read_state(env),
            term.library.frame(term.indices)["state"],
            atol=2e-5,
            rtol=2e-5,
        )

        # Deviation from reference height, followed by reference-state auto-reset.
        root = robot.data.root_link_pose_w.clone()
        desired = term.library.frame(term.library.advance(term.indices))["state"]
        root[:, 2] = desired[:, 2] + env.scene.env_origins[:, 2] + 0.4
        robot.write_root_state_to_sim(
            torch.cat((root, torch.zeros(8, 6, device=env.device)), -1)
        )
        env.sim.forward()
        _, _, failed, _, _ = env.step(a)
        assert failed.all()
        assert torch.isin(term.indices, term.library.starts).all()
        print(
            json.dumps(
                {
                    "tracking_integration": "passed",
                    "fk_max_errors": fk_errors,
                    "reference_state_reset": True,
                    "reference_clock_and_clip_end": True,
                    "reward_sum_and_gradient": True,
                    "height_deviation_resets": int(failed.sum()),
                    "motion": term.library.metadata(),
                }
            ),
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
