"""资源信息边界、整轮记忆、联合概率、完整序列更新与断点恢复。"""

from copy import deepcopy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from resource_rl.observation import ResourceObservation, release_seconds
from resource_rl.policy import ResourcePolicy
from resource_rl.ppo import advantages, update
from resource_rl.rollout import collate, observation_tensor


def toy_episode(model, seed=11, steps=7):
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    observations, actions, logp, values = [], [], [], []
    previous = np.full(3, 6, dtype="int64")
    actor_state = critic_state = None
    for time in range(steps):
        row = {"nodes": rng.normal(size=(3, 5)).astype("float32"),
            "edges": rng.normal(size=(3, 3, 2)).astype("float32"),
            "drivers": rng.normal(size=(4, 4)).astype("float32"),
            "orders": rng.normal(size=(time%4, 6)).astype("float32"),
            "previous_actions": previous.copy()}
        chosen, probability, value, actor_state, critic_state = model.step(
            observation_tensor(row), actor_state, critic_state, generator)
        observations.append(row); actions.append(chosen[0].numpy().copy())
        logp.append(float(probability[0])); values.append(float(value[0]))
        previous = actions[-1]
    result = {name: np.stack([row[name] for row in observations]) for name in ("nodes", "edges", "drivers", "previous_actions")}
    return {**result, "orders": np.concatenate([row["orders"] for row in observations]),
        "order_lengths": np.asarray([len(row["orders"]) for row in observations], dtype="int64"),
        "actions": np.asarray(actions), "log_probs": np.asarray(logp, dtype="float32"),
        "values": np.asarray(values, dtype="float32"), "rewards": rng.uniform(10, 100, size=steps).astype("float32"),
        "terminals": np.asarray([0.]*(steps-1)+[1.], dtype="float32"),
        "aux_targets": rng.uniform(0, 1, size=(steps, 15)).astype("float32")}


class ResourcePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(201)
        self.config = dict(grid_count=3, node_size=5, edge_size=2, driver_size=4, order_size=6,
            price_count=2, radius_count=3, hidden_size=16, heads=2, layers=1, window=4)
        self.settings = dict(warmup_steps=2, sequence_batch_size=2, reward_scale=100., gamma=1., gae_lambda=.95,
            actor_epochs=2, critic_epochs=3, clip=.1, entropy_weight=.001, target_kl=.1,
            max_grad_norm=.5, value_weight=.5, auxiliary_weight=.1)

    def model(self, variant="flow"):
        return ResourcePolicy(**self.config, variant=variant).eval()

    def optimizers(self, model):
        return (torch.optim.Adam(model.actor_parameters(), lr=1e-4, eps=1e-5),
                torch.optim.Adam(model.critic_parameters(), lr=1e-4, eps=1e-5))

    def test_streaming_and_sequence_likelihoods_values_match_for_both_structures(self):
        for variant in ("flow", "window"):
            model = self.model(variant)
            episodes = [toy_episode(model, seed) for seed in (11, 12)]
            data = collate(episodes, "cpu")
            with torch.no_grad():
                logp, entropy = model.actor_sequence(data)
                value, auxiliary = model.critic_sequence(data)
            torch.testing.assert_close(logp, data["log_probs"], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(value, data["values"], rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.isfinite(entropy).all() and torch.isfinite(auxiliary).all())

    def test_future_observations_cannot_change_past_policy_or_value(self):
        model = self.model()
        data = collate([toy_episode(model)], "cpu")
        changed = deepcopy(data)
        for name in ("nodes", "edges", "drivers", "orders"):
            changed[name][:, 4:] += 10.
        first = model.actor_sequence(data)[0]
        second = model.actor_sequence(changed)[0]
        torch.testing.assert_close(first[:, :4], second[:, :4], rtol=0, atol=0)
        first = model.critic_sequence(data)[0]
        second = model.critic_sequence(changed)[0]
        torch.testing.assert_close(first[:, :4], second[:, :4], rtol=0, atol=0)

    def test_flow_remembers_beyond_window_while_control_forgets(self):
        for variant in ("flow", "window"):
            model = self.model(variant)
            data = collate([toy_episode(model)], "cpu")
            changed = deepcopy(data); changed["nodes"][:, 0] += 7.
            first = model.actor_sequence(data)[0][:, -1]
            second = model.actor_sequence(changed)[0][:, -1]
            if variant == "flow":
                self.assertGreater(float((first-second).abs().max().detach()), 1e-8)
            else:
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_actor_cannot_read_privileged_tokens_and_value_cannot_read_current_action(self):
        model = self.model()
        data = collate([toy_episode(model)], "cpu")
        changed = deepcopy(data); changed["drivers"] += 10.; changed["orders"] += 10.
        torch.testing.assert_close(model.actor_sequence(data)[0], model.actor_sequence(changed)[0], rtol=0, atol=0)
        changed = deepcopy(data); changed["actions"] = (changed["actions"]+2)%6
        value, prediction = model.critic_sequence(data)
        other_value, other_prediction = model.critic_sequence(changed)
        torch.testing.assert_close(value, other_value, rtol=0, atol=0)
        self.assertGreater(float((prediction-other_prediction).abs().max().detach()), 1e-5)
        actor = {id(p) for p in model.actor_parameters()}
        critic = {id(p) for p in model.critic_parameters()}
        self.assertFalse(actor & critic)
        self.assertEqual(actor | critic, {id(p) for p in model.parameters()})

    def test_joint_probability_normalization_and_action_causality(self):
        model = self.model()
        actions = torch.tensor(list(itertools.product(range(6), repeat=3)))
        encoded = torch.randn(1, 3, 16).expand(len(actions), -1, -1)
        logs = model.decoder(encoded, actions)
        probability = logs.gather(-1, actions[..., None]).squeeze(-1).sum(-1).exp().sum()
        self.assertAlmostEqual(float(probability.detach()), 1., places=5)
        changed = actions.clone(); changed[:, 1:] = (changed[:, 1:]+1)%6
        other = model.decoder(encoded, changed)
        torch.testing.assert_close(logs[:, :2], other[:, :2], rtol=0, atol=1e-7)

    def test_gae_terminal_and_no_cross_episode_bootstrap(self):
        rewards = np.array([[1., 2.], [4., 8.]], dtype="float32")
        values = rewards*10
        _, targets = advantages(rewards, values, np.array([[0., 1.], [0., 1.]]), gae_lambda=1.)
        np.testing.assert_allclose(targets, [[3., 2.], [12., 8.]])

    def test_critic_updates_continue_after_actor_KL_stop(self):
        model = self.model()
        episodes = [toy_episode(model, seed) for seed in (11, 12, 13, 14)]
        settings = dict(self.settings, target_kl=0.)
        result = update(model, *self.optimizers(model), episodes, settings, np.random.default_rng(91), "cpu")
        self.assertTrue(np.isfinite(list(result.values())).all())
        self.assertEqual(result["actor_epochs"], 1)
        self.assertEqual(result["critic_effective_epochs"], 3.)
        self.assertEqual(result["critic_sample_visits"], 4*5*3)

    def test_warmup_reward_does_not_enter_training_objectives(self):
        model = self.model()
        other = self.model(); other.load_state_dict(deepcopy(model.state_dict()))
        episodes = [toy_episode(model, seed) for seed in (11, 12)]
        changed = deepcopy(episodes)
        for episode in changed:
            episode["rewards"][:2] += 100000.
        first = update(model, *self.optimizers(model), episodes, self.settings, np.random.default_rng(91), "cpu")
        second = update(other, *self.optimizers(other), changed, self.settings, np.random.default_rng(91), "cpu")
        self.assertEqual(first, second)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)

    def test_both_optimizers_and_rng_restore_exact_sequence_continuation(self):
        model = self.model()
        actor, critic = self.optimizers(model)
        rng = np.random.default_rng(91)
        update(model, actor, critic, [toy_episode(model, seed) for seed in (11, 12)], self.settings, rng, "cpu")
        other = self.model(); other.load_state_dict(deepcopy(model.state_dict()))
        a2, c2 = self.optimizers(other)
        a2.load_state_dict(deepcopy(actor.state_dict())); c2.load_state_dict(deepcopy(critic.state_dict()))
        rng2 = np.random.default_rng(); rng2.bit_generator.state = deepcopy(rng.bit_generator.state)
        episodes = [toy_episode(model, seed) for seed in (13, 14)]
        first = update(model, actor, critic, episodes, self.settings, rng, "cpu")
        second = update(other, a2, c2, episodes, self.settings, rng2, "cpu")
        self.assertEqual(first, second)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)


class ResourceObservationTests(unittest.TestCase):
    def test_release_time_includes_delivery_after_pickup_and_minute_rounding(self):
        drivers = pd.DataFrame(dict(status=[2, 1, 0, 4], remaining_time=[61., 61., -60., 60.],
                                    _pending_trip_time=[121., 999., 0., 0.]))
        np.testing.assert_array_equal(release_seconds(drivers), [300., 120., 0., 60.])

    def test_new_state_separates_equal_counts_with_different_task_release_times(self):
        c = json.loads(Path("experiments/online_ppo_100000_20261104/protocol.json").read_text())["config"]
        adjacency = pd.read_csv(Path(c["data_dir"])/"grid_adjacency.csv").to_numpy()
        encoder = ResourceObservation(c, adjacency)
        drivers = pd.DataFrame(dict(grid_id=[0], target_grid_id=[1], status=[1], remaining_time=[60.],
            _pending_trip_time=[0.], total_idle_time=[0.], lng=[-74.], lat=[40.75],
            target_loc_lng=[-73.98], target_loc_lat=[40.77]))
        columns = ["origin_grid_id", "dest_grid_id", "origin_lng", "origin_lat", "dest_lng", "dest_lat", "wait_time",
                   "maximum_wait_time", "quoted_fare", "price_multiplier", "trip_distance", "trip_time"]
        queue = pd.DataFrame(columns=columns)
        sim = SimpleNamespace(driver_table=drivers, wait_requests=queue, grid_num=6, current_step=20, delta_t=60.)
        before_drivers, before_queue = drivers.copy(deep=True), queue.copy(deep=True)
        first = encoder.capture(sim, None, np.zeros(6, dtype="int64"))
        pd.testing.assert_frame_equal(drivers, before_drivers); pd.testing.assert_frame_equal(queue, before_queue)
        drivers.loc[0, "remaining_time"] = 600.
        second = encoder.capture(sim, None, np.zeros(6, dtype="int64"))
        self.assertEqual(first["nodes"][0, 2], second["nodes"][0, 2])
        self.assertNotEqual(first["edges"][0, 1, 3], second["edges"][0, 1, 3])

    def test_capture_preserves_random_stream_and_ignores_future_and_private_draws(self):
        from resource_rl.environment import ResourceEnvironment
        c = json.loads(Path("experiments/online_ppo_100000_20261104/protocol.json").read_text())["config"]
        env = ResourceEnvironment(c)
        env.reset(1_050_000_102)
        before_numpy = deepcopy(np.random.get_state())
        before_rng = deepcopy(env.sim._ensure_rng().get_state())
        original = env.observation()
        env.sim.request_databases = object()
        for column in env.sim.driver_table.columns:
            if "draw" in column or "_profile" in column:
                env.sim.driver_table[column] = 999999.
        changed = env.observation()
        for name in original:
            np.testing.assert_array_equal(original[name], changed[name])
        for first, second in zip(before_numpy, np.random.get_state()):
            np.testing.assert_array_equal(first, second)
        for first, second in zip(before_rng, env.sim._ensure_rng().get_state()):
            np.testing.assert_array_equal(first, second)


class ResourceStatisticsTests(unittest.TestCase):
    def test_crossed_paired_statistics_cancel_common_scenario_effects(self):
        from resource_rl.evaluate import METRICS, paired_statistics
        rows = []
        for seed in range(8):
            for variant, train_seed, gain in (("flow", 201, 2.), ("flow", 202, 4.),
                ("window", 201, 0.), ("window", 202, 0.), ("fixed_f3", -1, -1.), ("original_ar_ppo", -1, 1.)):
                rows.append(dict(seed=seed, training_seed=train_seed, variant=variant,
                                 **{name: 1000.+seed+gain for name in METRICS}))
        result = paired_statistics(pd.DataFrame(rows))
        primary = next(row for row in result if row["metric"] == "platform_revenue" and row["reference"] == "window")
        self.assertEqual(primary["mean_difference"], 3.)
        self.assertEqual(primary["per_initialization_differences"], [2., 4.])
        self.assertEqual(primary["training_initializations"], 2)
        self.assertEqual(primary["environment_seeds"], 8)

    def test_sealed_test_rejects_incomplete_training(self):
        from unittest.mock import patch
        from resource_rl.evaluate import run
        from resource_rl.protocol import sha256
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            protocol = root/"protocol.json"; protocol.write_text("{}")
            folder = root/"models/flow_seed201"; folder.mkdir(parents=True)
            torch.save({"protocol_sha256": sha256(protocol), "episodes": 64}, folder/"latest.pt")
            plan = {"training_seeds": [201], "variants": ["flow"], "training_episodes": 100000}
            with patch("resource_rl.evaluate.load_protocol", return_value=plan):
                with self.assertRaises(ValueError):
                    run(SimpleNamespace(protocol=str(protocol), models=str(root/"models"), split="test",
                                        output=str(root/"results"), workers=1))


@unittest.skipUnless(Path("my_data/grid6/orders_grid6_2015-05-18.pkl").exists(), "requires local simulator inputs")
class ResourceSimulatorTests(unittest.TestCase):
    def test_fixed_F3_matches_original_simulation_and_warmup_exactly(self):
        from online_rl.environment import DecisionEnvironment
        from resource_rl.environment import ResourceEnvironment
        torch.set_num_threads(1)
        old = json.loads(Path("experiments/online_ppo_100000_20261104/protocol.json").read_text())
        c, seed = old["config"], 1_050_000_101
        legacy = DecisionEnvironment(c, old["observation_metadata"])
        legacy.reset(seed)
        action = legacy.pairs.index([c["fixed_price"], c["fixed_radius"]])
        terminal = False
        while not terminal:
            _, _, terminal = legacy.step(np.full(6, action, dtype="int64"))
        expected = legacy.summary()
        env = ResourceEnvironment(c, legacy.pattern)
        observation = env.reset(seed)
        self.assertEqual(len(env.warmup), 4)
        self.assertTrue(all(np.isfinite(value).all() for value in observation.values()))
        rewards, steps, terminal = [], 0, False
        while not terminal:
            _, reward, terminal = env.step(np.full(6, action, dtype="int64"))
            rewards.append(reward); steps += 1
        actual = env.summary()
        self.assertEqual(steps, 56)
        self.assertAlmostEqual(sum(rewards), actual["platform_revenue"], places=7)
        pd.testing.assert_frame_equal(env.minutes, legacy.minutes, check_exact=True)
        pd.testing.assert_frame_equal(env.periods.drop(columns="run_id"), legacy.periods.drop(columns="run_id"), check_exact=True)
        for key in ("platform_revenue", "request_rate", "driver_accept_rate", "order_completion_rate",
                    "mean_wait_to_match_seconds", "initial_driver_sha256", "warmup_metrics_sha256", "sampled_orders_sha256"):
            self.assertEqual(actual[key], expected[key], key)


if __name__ == "__main__":
    unittest.main()
