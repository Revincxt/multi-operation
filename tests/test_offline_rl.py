"""离线 RL 的时间边界、局部观测和 IQL 更新公式。"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from offline_rl.data import Replay, VERSION, encode_state, state_context_indices
from offline_rl.iql import Actor, IQL, advantage_weights, bellman_target, expectile_loss
from offline_rl.train import atomic_save, evaluation_output


class OfflineRLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = dict(state_size=4, grid_count=2, price_count=2, radius_count=2,
                           local_indices=[[0, 1], [2, 3]], hidden_size=8, embedding_size=3)

    def test_expectile_advantage_weights_and_terminal_bellman_formula(self):
        loss = expectile_loss(torch.tensor([-2., 2.]), 0.7)
        self.assertAlmostEqual(float(loss), 2.0)
        weights = advantage_weights(torch.tensor([0., 1., 100.], requires_grad=True), 3.)
        np.testing.assert_allclose(weights.numpy(), [1, np.exp(3), 100], rtol=1e-6)
        self.assertFalse(weights.requires_grad)
        target = bellman_target(torch.tensor([2., 3.]), torch.tensor([1., 0.]),
                                torch.tensor([10., 1000.]), 0.99)
        torch.testing.assert_close(target, torch.tensor([11.9, 3.]))

    def test_six_agent_structure_has_independent_two_heads_and_local_information(self):
        actor = Actor(state_size=12, grid_count=6, price_count=11, radius_count=10,
                      local_indices=[[g * 2, g * 2 + 1] for g in range(6)], hidden_size=8)
        states = torch.randn(3, 12)
        p, r = actor(states)
        self.assertEqual(len(actor.agents), 6)
        self.assertEqual(p.shape, (3, 6, 11))
        self.assertEqual(r.shape, (3, 6, 10))
        changed = states.clone()
        changed[:, 2:] += 100
        p2, r2 = actor(changed)
        torch.testing.assert_close(p[:, 0], p2[:, 0])
        torch.testing.assert_close(r[:, 0], r2[:, 0])
        actions = torch.randint(110, (3, 6))
        (-actor.log_prob(states, actions).mean()).backward()
        for agent in actor.agents:
            self.assertGreater(float(agent.price_head.weight.grad.abs().sum()), 0)
            self.assertGreater(float(agent.radius_head.weight.grad.abs().sum()), 0)
        constrained = actor.act(states, [1], [0])
        self.assertTrue(constrained.eq(10).all())

    def test_current_actions_are_excluded_and_grid_order_is_explicit(self):
        processor = SimpleNamespace(context_columns=["decision_start_sim_time", "price_multiplier",
                                                     "matching_radius_km", "start_idle_driver_num",
                                                     "joint_price_grid_0", "joint_radius_grid_0"])
        indices = state_context_indices(processor)
        self.assertEqual(indices, [0, 3])
        history = np.arange(8).reshape(1, 2, 2, 2)
        context = np.arange(12).reshape(1, 2, 6).astype(float)
        before = encode_state(history, context, indices)
        context[..., [1, 2, 4, 5]] += 10000
        np.testing.assert_array_equal(encode_state(history, context, indices), before)

    def test_transitions_do_not_cross_trajectories_and_keep_final_rewards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train").mkdir()
            metadata = {"schema_version": VERSION, "status": "complete", "periods_per_run": 3,
                        "splits": {"train": {"transitions": 6}}}
            (root / "manifest.json").write_text(json.dumps(metadata))
            states = np.arange(12, dtype="float32").reshape(6, 2)
            np.save(root / "train/states.npy", states)
            np.save(root / "train/actions.npy", np.zeros((6, 2), dtype="int16"))
            np.save(root / "train/rewards.npy", np.arange(6, dtype="float32"))
            batch = Replay(root, "train").batch([1, 2, 3, 5])
            np.testing.assert_array_equal(batch["masks"], [1, 0, 1, 0])
            np.testing.assert_array_equal(batch["next_states"], [states[2], [0, 0], states[4], [0, 0]])
            np.testing.assert_array_equal(batch["rewards"], [1, 2, 3, 5])

    def test_full_optimizer_checkpoint_resumes_same_updates(self):
        first = IQL(self.config)
        rng = np.random.default_rng(42)
        batch = {"states": rng.normal(size=(8, 4)).astype("float32"),
                 "actions": rng.integers(4, size=(8, 2), dtype="int64"),
                 "rewards": np.ones(8, dtype="float32"),
                 "next_states": rng.normal(size=(8, 4)).astype("float32"),
                 "masks": np.ones(8, dtype="float32")}
        first.update(batch)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.pt"
            atomic_save(path, {"learner": first.state_dict(),
                               "numpy_rng": rng.bit_generator.state})
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)["learner"]
        second = IQL(self.config)
        second.load_state_dict(checkpoint)
        torch.testing.assert_close(first.update(batch), second.update(batch))
        for name in ("actor", "bc", "q", "target_q", "value"):
            for a, b in zip(getattr(first, name).parameters(), getattr(second, name).parameters()):
                torch.testing.assert_close(a, b)
                self.assertTrue(torch.isfinite(a).all())
        self.assertFalse(any(p.requires_grad for p in first.target_q.parameters()))

    def test_resumed_evaluation_keeps_previous_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step005000"
            self.assertEqual(evaluation_output(path), path)
            path.mkdir()
            path.with_name("step005000_retry001").mkdir()
            self.assertEqual(evaluation_output(path), path.with_name("step005000_retry002"))
            self.assertTrue(path.is_dir())


if __name__ == "__main__":
    unittest.main()
