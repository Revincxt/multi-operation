"""验证研究假设依赖的因果概率、目标口径、边界与可复现性。"""

from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from online_rl.environment import DecisionEnvironment, ObservationEncoder
from online_rl.evaluate import paired_statistics
from online_rl.policy import JointPolicy
from online_rl.ppo import advantages, clipped_policy_loss, update


class OnlineAlgorithmTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(81)
        self.config = dict(grid_count=3, history_steps=2, history_features=2, context_size=1,
            price_count=2, radius_count=3, hidden_size=16, memory_size=8, heads=2, layers=1)
        self.model = JointPolicy(**self.config).eval()
        self.states = torch.randn(4, 15)

    def test_sampled_joint_likelihood_equals_teacher_forced_likelihood(self):
        actions, logged, value = self.model.sample(self.states, torch.Generator().manual_seed(42))
        recomputed, entropy, again = self.model.evaluate_actions(self.states, actions)
        torch.testing.assert_close(logged, recomputed, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(value, again)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(((actions >= 0) & (actions < 6)).all())

    def test_own_and_future_actions_cannot_change_a_conditional_distribution(self):
        encoded = self.model.actor_encoder(self.states)
        actions = torch.tensor([[0, 1, 2]]).expand(4, -1).clone()
        reference = self.model.conditional_log_probs(encoded, actions)
        for grid in range(3):
            changed = actions.clone()
            changed[:, grid:] = (changed[:, grid:]+3)%6
            altered = self.model.conditional_log_probs(encoded, changed)
            torch.testing.assert_close(reference[:, :grid+1], altered[:, :grid+1], rtol=0, atol=1e-7)
        changed = actions.clone(); changed[:, 0] = 4
        altered = self.model.conditional_log_probs(encoded, changed)
        self.assertGreater(float((reference[:, 1:]-altered[:, 1:]).abs().max().detach()), 1e-6)

    def test_toy_joint_probability_sums_to_one_over_all_joint_actions(self):
        actions = torch.tensor(list(itertools.product(range(6), repeat=3)))
        states = self.states[:1].expand(len(actions), -1)
        logs = self.model.evaluate_actions(states, actions)[0]
        self.assertAlmostEqual(float(logs.exp().sum().detach()), 1., places=5)

    def test_factorized_control_is_invariant_to_all_action_prefixes(self):
        model = JointPolicy(**self.config, autoregressive=False).eval()
        encoded = model.actor_encoder(self.states)
        first = model.conditional_log_probs(encoded, torch.zeros((4, 3), dtype=torch.long))
        second = model.conditional_log_probs(encoded, torch.full((4, 3), 5, dtype=torch.long))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_gae_lambda_one_matches_undiscounted_episode_returns(self):
        rewards = np.array([[1., 2.], [4., 8.]], dtype="float32")
        values = np.array([[10., 20.], [40., 80.]], dtype="float32")
        advantage, returns = advantages(rewards, values, np.array([[0., 1.], [0., 1.]]), gae_lambda=1.)
        expected = np.array([[3., 2.], [12., 8.]])
        np.testing.assert_allclose(returns, expected)
        np.testing.assert_allclose(advantage, expected-values)
        _, extended = advantages(np.r_[rewards, [[100., 200.]]], np.r_[values, [[0., 0.]]],
                                 np.array([[0., 1.]]*3), gae_lambda=1.)
        np.testing.assert_allclose(extended[:2], returns)

    def test_joint_ratio_clips_beneficial_large_moves_and_penalizes_harmful_moves(self):
        logp = torch.full((2,), float(np.log(1.5)), requires_grad=True)
        loss, kl, fraction = clipped_policy_loss(logp, torch.zeros(2), torch.tensor([1., -1.]), .1)
        self.assertAlmostEqual(float(loss.detach()), .2, places=6)
        loss.backward()
        self.assertAlmostEqual(float(logp.grad[0]), 0.)
        self.assertGreater(float(logp.grad[1]), 0.)
        self.assertGreater(float(kl.detach()), 0.)
        self.assertEqual(float(fraction), 1.)

    def episodes(self):
        states = torch.randn(6, 15)
        actions, logp, value = self.model.sample(states, torch.Generator().manual_seed(51))
        return [{"states": states[i:i+3].numpy(), "actions": actions[i:i+3].numpy(),
            "log_probs": logp[i:i+3].numpy(), "values": value[i:i+3].numpy(),
            "rewards": np.array([100., 20., 80.], dtype="float32"),
            "terminals": np.array([0., 0., 1.], dtype="float32")} for i in (0, 3)]

    def settings(self):
        return dict(gamma=1., gae_lambda=.95, reward_scale=100., epochs=2, batch_size=4,
                    clip=.1, target_kl=.1, max_grad_norm=.5, value_weight=.5, entropy_weight=.001)

    def test_ppo_uses_finite_gradients_and_reuses_every_rollout_sample(self):
        episodes = self.episodes()
        before = deepcopy(self.model.state_dict())
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        result = update(self.model, optimizer, episodes, self.settings(), np.random.default_rng(81), "cpu")
        self.assertTrue(np.isfinite(list(result.values())).all())
        self.assertEqual(result["sample_visits"], 12)
        self.assertEqual(result["effective_epochs"], 2.)
        self.assertTrue(any(not torch.equal(before[key], value) for key, value in self.model.state_dict().items()))

    def test_complete_optimizer_and_rng_restore_matches_continuation(self):
        episodes = self.episodes()
        first = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        rng = np.random.default_rng(81)
        update(self.model, first, episodes, self.settings(), rng, "cpu")
        restored = JointPolicy(**self.config)
        restored.load_state_dict(deepcopy(self.model.state_dict()))
        second = torch.optim.Adam(restored.parameters(), lr=1e-4)
        second.load_state_dict(deepcopy(first.state_dict()))
        other_rng = np.random.default_rng(); other_rng.bit_generator.state = deepcopy(rng.bit_generator.state)
        result = update(self.model, first, episodes, self.settings(), rng, "cpu")
        repeated = update(restored, second, episodes, self.settings(), other_rng, "cpu")
        for key in result:
            self.assertAlmostEqual(result[key], repeated[key], places=6)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, self.model.state_dict()[key], rtol=0, atol=0)

    def test_crossed_statistics_keep_environment_and_training_uncertainty_separate(self):
        rows = []
        for seed in range(5):
            for name, gain in (("fixed_f3", 0), ("AR-PPO_seed81", 1), ("AR-PPO_seed82", 3),
                               ("Factorized-PPO_seed81", .5), ("Factorized-PPO_seed82", .5)):
                rows.append({"seed": seed, "policy": name, "platform_revenue": 10.+seed+gain})
        statistics = paired_statistics(pd.DataFrame(rows), ["fixed_f3"], draws=1000)
        combined = next(row for row in statistics if row["policy"] == "AR-PPO" and row["reference"] == "fixed_f3")
        self.assertEqual(combined["training_initializations"], 2)
        self.assertEqual(combined["mean_difference_usd"], 2.)
        self.assertGreater(combined["ci95_high"]-combined["ci95_low"], 0.)
        control = next(row for row in statistics if row["policy"] == "AR-PPO" and row["reference"] == "Factorized-PPO")
        self.assertEqual(control["mean_difference_usd"], 1.5)


@unittest.skipUnless(Path("my_data/grid6/orders_grid6_2015-05-18.pkl").exists(), "requires local grid6 simulation inputs")
class OnlineSimulatorTests(unittest.TestCase):
    def test_f3_rewards_observations_and_initial_conditions_match_existing_evaluator(self):
        from baseline.compare_policies import run_episode, sample_common_orders
        from data_pipeline.collect_trajectories import simulation_window
        from src.env.actions import uniform_grid_actions
        from src.env.simulator_pattern import SimulatorPattern
        torch.set_num_threads(1)
        config = json.loads(Path("experiments/ma_iql_comparison_20261103/protocol.json").read_text())["config"]
        metadata = json.loads(Path("datasets_grid6_low_mixed100k_20261103/offline_rl/manifest.json").read_text())
        pattern = SimulatorPattern(config["date"], data_dir=config["data_dir"])
        environment = DecisionEnvironment(config, metadata, pattern)
        scenario_seed = 1_100_000_081
        observation = environment.reset(scenario_seed)
        fixed_index = metadata["action_pairs"].index([config["fixed_price"], config["fixed_radius"]])
        states, rewards, terminal = [], [], False
        while not terminal:
            states.append(observation.copy())
            observation, reward, terminal = environment.step(np.full(6, fixed_index, dtype="int64"))
            rewards.append(reward)
        self.assertEqual(len(states), 56)
        self.assertFalse(observation.any())
        self.assertAlmostEqual(sum(rewards), environment.summary()["platform_revenue"], places=7)
        encoder = ObservationEncoder(metadata)
        seen = []

        def policy(periods, decision, time, interval, current_state):
            seen.append(encoder(periods, decision, time, interval, current_state))
            return uniform_grid_actions(6, config["fixed_price"], config["fixed_radius"])

        _, start, end = simulation_window(config["start_time"], config["simulation_steps"])
        requests, _ = sample_common_orders(pattern.request_all, start, end, config["order_sample_ratio"], scenario_seed)
        with tempfile.TemporaryDirectory() as temporary:
            reference = run_episode(config, "fixed_f3", scenario_seed, requests, pattern.driver_info,
                                    None, Path(temporary)/"reference", action_policy=policy)
        np.testing.assert_allclose(states, seen, rtol=0, atol=0)
        for field in ("platform_revenue", "initial_driver_sha256", "warmup_metrics_sha256", "request_rate",
                      "driver_accept_rate", "order_completion_rate", "mean_wait_to_match_seconds"):
            self.assertEqual(environment.summary()[field], reference[field])


if __name__ == "__main__":
    unittest.main()
