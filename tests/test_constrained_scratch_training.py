"""从零训练的共同初始点、断点身份和续训可重复性检查。"""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from constrained_rl.policy import ConstrainedPolicy
from constrained_rl.ppo import update
from constrained_rl.train_scratch import (create_member, critic_diagnostics, random_initial,
                                          restore_member, save_member)
from offline_rl.data import sha256


class ConstrainedScratchTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = {
            "training_seed": 412,
            "model": dict(grid_count=6, history_steps=2, history_features=2, context_size=1,
                price_count=11, radius_count=10, hidden_size=16, memory_size=8, heads=2, layers=1),
            "algorithm": dict(gamma=1., gae_lambda=.95, reward_scale=7000., epochs=2, batch_size=4,
                learning_rate=3e-4, clip=.1, target_kl=.1, max_grad_norm=.5, value_weight=.5,
                entropy_weight=.001, cost_value_weight=.5, dual_learning_rate=5.,
                value_scale_floor=.05, cost_scale_floor=.01)}
        (self.root / "protocol.json").write_text(json.dumps(self.plan))
        self.initial_path = self.root / "initial.pt"

    def initial(self):
        return random_initial(self.plan, self.initial_path, "cpu")

    def episodes(self, model, seed=51):
        states = torch.randn(8, model.state_size, generator=torch.Generator().manual_seed(seed))
        model.eval()
        with torch.no_grad():
            actions, logp, value = model.sample(states, torch.Generator().manual_seed(seed + 1))
            costs = model.cost_values(states)
        return [{"states": states[i:i+4].numpy(), "actions": actions[i:i+4].numpy(),
                 "log_probs": logp[i:i+4].numpy(), "values": value[i:i+4].numpy(),
                 "cost_values": costs[i:i+4].numpy(),
                 "rewards": np.array([90., 100., 110., 120.], dtype="float32"),
                 "costs": np.array([[.02, -.01], [.01, .01], [-.01, .01], [.03, -.02]], dtype="float32"),
                 "terminals": np.array([0., 0., 0., 1.], dtype="float32")}
                for i in (0, 4)]

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_random_initial_is_deterministic_near_uniform_and_never_imports_weights(self):
        with patch.object(ConstrainedPolicy, "load_original", side_effect=AssertionError):
            first = self.initial()
            torch.manual_seed(999)
            torch.randn(23)
            repeated = self.initial()
            separate = random_initial(self.plan, self.root / "second_initial.pt", "cpu")
        self.assert_nested_equal(first.state_dict(), repeated.state_dict())
        self.assert_nested_equal(first.state_dict(), separate.state_dict())
        states = torch.randn(12, first.state_size, generator=torch.Generator().manual_seed(19))
        actions = torch.zeros((len(states), first.grids), dtype=torch.long)
        with torch.no_grad():
            logs = first.conditional_log_probs(first.actor_encoder(states), actions)
            entropy = -(logs.exp() * logs).sum(-1).sum(-1)
            costs = first.cost_values(states)
            values = first.value(states)
        self.assertLess(float((6 * math.log(110) - entropy).abs().max()), .01)
        self.assertLess(float((logs.exp() - 1 / 110).abs().max()), .001)
        torch.testing.assert_close(costs, torch.zeros_like(costs), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(any(value.abs().sum() > 0 for value in first.value_head.parameters()))

    def test_random_initial_rejects_wrong_protocol_or_trained_initial_weights(self):
        self.initial()
        original = torch.load(self.initial_path, map_location="cpu", weights_only=True)
        for field, changed in (("initialization", "warm_start"), ("training_seed", 413),
                               ("protocol_sha256", "wrong")):
            with self.subTest(field=field):
                invalid = deepcopy(original)
                invalid[field] = changed
                torch.save(invalid, self.initial_path)
                with self.assertRaises(ValueError):
                    self.initial()
        invalid = deepcopy(original)
        invalid["model_state_dict"]["price_head.bias"][0] += .01
        torch.save(invalid, self.initial_path)
        with self.assertRaises(ValueError):
            self.initial()

    def test_members_start_identically_with_independent_models_and_empty_optimizers(self):
        initial = self.initial()
        first = create_member("revenue_control", initial, self.plan)
        second = create_member("constrained", initial, self.plan)
        self.assert_nested_equal(first["model"].state_dict(), initial.state_dict())
        self.assert_nested_equal(first["model"].state_dict(), second["model"].state_dict())
        for original, left, right in zip(initial.parameters(), first["model"].parameters(),
                                         second["model"].parameters()):
            self.assertNotEqual(original.data_ptr(), left.data_ptr())
            self.assertNotEqual(left.data_ptr(), right.data_ptr())
        for member in (first, second):
            self.assertFalse(member["optimizer"].state)
            self.assertEqual(member["optimizer"].param_groups[0]["lr"], 3e-4)
            self.assertEqual(member["episodes"], 0)
            self.assertEqual(member["iteration"], 0)
            self.assertEqual(member["history"], [])
            np.testing.assert_array_equal(member["multipliers"], [0., 0.])
        self.assert_nested_equal(first["rng"].bit_generator.state, second["rng"].bit_generator.state)

    def test_disk_resume_restores_full_state_and_reproduces_next_update(self):
        initial = self.initial()
        first = create_member("constrained", initial, self.plan)
        update(first["model"], first["optimizer"], self.episodes(first["model"]),
               self.plan["algorithm"], first["rng"], "cpu", first["multipliers"])
        first.update(episodes=2, iteration=1,
                     history=[{"method": "constrained", "iteration": 1, "episodes": 2, "revenue_mean_usd": 420.}])
        protocol_hash, initial_hash = sha256(self.root / "protocol.json"), sha256(self.initial_path)
        save_member(first, self.root, protocol_hash, initial_hash)
        restored = create_member("constrained", initial, self.plan)
        restore_member(restored, self.root, protocol_hash, initial_hash)
        self.assertTrue(first["optimizer"].state)
        self.assertGreater(first["multipliers"][0], 0.)
        self.assert_nested_equal(first["model"].state_dict(), restored["model"].state_dict())
        self.assert_nested_equal(first["optimizer"].state_dict(), restored["optimizer"].state_dict())
        self.assert_nested_equal(first["rng"].bit_generator.state, restored["rng"].bit_generator.state)
        for key in ("multipliers", "episodes", "iteration", "history"):
            self.assert_nested_equal(first[key], restored[key])
        episodes = self.episodes(first["model"], seed=74)
        uninterrupted = update(first["model"], first["optimizer"], episodes, self.plan["algorithm"],
                               first["rng"], "cpu", first["multipliers"])
        resumed = update(restored["model"], restored["optimizer"], episodes, self.plan["algorithm"],
                         restored["rng"], "cpu", restored["multipliers"])
        self.assert_nested_equal(uninterrupted, resumed)
        self.assert_nested_equal(first["model"].state_dict(), restored["model"].state_dict())
        self.assert_nested_equal(first["optimizer"].state_dict(), restored["optimizer"].state_dict())
        self.assert_nested_equal(first["rng"].bit_generator.state, restored["rng"].bit_generator.state)
        np.testing.assert_array_equal(first["multipliers"], restored["multipliers"])

    def test_restore_rejects_wrong_initial_or_protocol_identity_before_loading(self):
        initial = self.initial()
        member = create_member("constrained", initial, self.plan)
        protocol_hash, initial_hash = sha256(self.root / "protocol.json"), sha256(self.initial_path)
        save_member(member, self.root, protocol_hash, initial_hash)
        checkpoint = self.root / "constrained" / "latest.pt"
        original = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for field, changed in (("protocol_sha256", "wrong"), ("initial_sha256", "wrong"),
                               ("method", "revenue_control"), ("initialization", "warm_start")):
            with self.subTest(field=field):
                invalid = deepcopy(original)
                invalid[field] = changed
                invalid["model_state_dict"]["price_head.bias"][0] += 1.
                torch.save(invalid, checkpoint)
                restored = create_member("constrained", initial, self.plan)
                with self.assertRaises(ValueError):
                    restore_member(restored, self.root, protocol_hash, initial_hash)
                self.assert_nested_equal(initial.state_dict(), restored["model"].state_dict())

    def test_restore_rejects_history_that_disagrees_with_progress(self):
        initial = self.initial()
        member = create_member("constrained", initial, self.plan)
        protocol_hash, initial_hash = sha256(self.root / "protocol.json"), sha256(self.initial_path)
        save_member(member, self.root, protocol_hash, initial_hash)
        checkpoint = self.root / "constrained" / "latest.pt"
        original = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for history in ([], [{"episodes": 4}]):
            with self.subTest(history=history):
                invalid = deepcopy(original)
                invalid.update(iteration=1, episodes=2, history=history)
                torch.save(invalid, checkpoint)
                with self.assertRaises(ValueError):
                    restore_member(create_member("constrained", initial, self.plan),
                                   self.root, protocol_hash, initial_hash)

    def test_critic_diagnostics_compare_recorded_predictions_with_undiscounted_returns(self):
        episodes = self.episodes(self.initial())
        for row in episodes:
            row["values"] = np.cumsum((row["rewards"] / 7000.)[::-1])[::-1].copy()
            row["cost_values"] = np.cumsum(row["costs"][::-1], axis=0)[::-1].copy()
        result = critic_diagnostics(episodes, 7000.)
        for name in ("revenue", "service", "waiting"):
            self.assertEqual(result[name + "_mc_mse"], 0.)
            self.assertEqual(result[name + "_mc_explained_variance"], 1.)


if __name__ == "__main__":
    unittest.main()
