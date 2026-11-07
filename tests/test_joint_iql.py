"""跨区域动作前缀、完整联合似然及成对IQL更新的科学对照。"""

from copy import deepcopy
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from offline_rl.coordination import make_actor
from offline_rl.joint_policy import JointActor, PairedIQL
from offline_rl.train import atomic_save
from offline_rl.train_joint import hierarchical_statistics


class JointIQLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = dict(state_size=4, grid_count=2, price_count=2, radius_count=2,
                           price_values=[1., 2.], radius_values=[1., 2.],
                           hidden_size=8, embedding_size=3)

    def test_autoregressive_likelihood_normalizes_over_all_joint_actions(self):
        actor = JointActor(**self.config, autoregressive=True)
        actions = torch.tensor(list(product(range(4), repeat=2)))
        states = torch.randn(1, 4).expand(len(actions), -1)
        self.assertAlmostEqual(float(actor.log_prob(states, actions).exp().sum().detach()), 1., places=5)
        (-actor.log_prob(states, actions).mean()).backward()
        for grid in range(2):
            self.assertGreater(float(actor.price_head.weight.grad[grid].abs().sum()), 0)
            self.assertGreater(float(actor.radius_head.weight.grad[grid].abs().sum()), 0)

    def test_only_previous_regions_reach_each_prefix_and_no_current_target_leak(self):
        actor = JointActor(**self.config, autoregressive=True)
        states = torch.randn(1, 4)
        a = torch.tensor([[0, 0]])
        b = torch.tensor([[0, 3]])
        torch.testing.assert_close(actor._features(states, a, 0), actor._features(states, b, 0))
        torch.testing.assert_close(actor._features(states, a, 1), actor._features(states, b, 1))
        b = torch.tensor([[3, 0]])
        self.assertFalse(torch.equal(actor._features(states, a, 1), actor._features(states, b, 1)))
        actor.autoregressive = False
        torch.testing.assert_close(actor._features(states, a, 1), actor._features(states, b, 1))

    def test_inference_consumes_its_own_chosen_prefix(self):
        actor = JointActor(**self.config, autoregressive=True)
        with patch.object(actor, "_features", wraps=actor._features) as features:
            choice = actor.act(torch.zeros(2, 4))
            states, selected, grid = features.call_args.args
            prefix = actor._features(states, selected, grid)[:, self.config["state_size"]:].reshape(2, 2, 3)
        torch.testing.assert_close(prefix[:, 0, 0], actor.price_codes[choice[:, 0] // 2])
        torch.testing.assert_close(prefix[:, 0, 1], actor.radius_codes[choice[:, 0] % 2])
        self.assertTrue(prefix[:, 0, 2].eq(1).all())
        self.assertTrue(prefix[:, 1].eq(0).all())
        with self.assertRaises(ValueError):
            actor.act(torch.zeros(1, 4), allowed_prices=[0])

    def test_same_parameter_count_initialization_and_information(self):
        learner = PairedIQL(self.config)
        independent, autoregressive = learner.actors["independent"], learner.actors["autoregressive"]
        self.assertEqual(sum(p.numel() for p in independent.parameters()),
                         sum(p.numel() for p in autoregressive.parameters()))
        for a, b in zip(independent.parameters(), autoregressive.parameters()):
            torch.testing.assert_close(a, b)
        states = torch.randn(3, 4)
        before = independent.region_logits(states, torch.zeros(3, 2, dtype=torch.long), 0)[0]
        states[:, -1] += 100
        after = independent.region_logits(states, torch.zeros(3, 2, dtype=torch.long), 0)[0]
        self.assertFalse(torch.equal(before, after))

    def test_batched_regions_equal_individual_regional_computation(self):
        for autoregressive in (False, True):
            actor = JointActor(**self.config, autoregressive=autoregressive)
            states, actions = torch.randn(7, 4), torch.randint(4, (7, 2))
            prices, radii = actor.logits(states, actions)
            for grid in range(2):
                p, r = actor.region_logits(states, actions, grid, actions[:, grid] // 2)
                torch.testing.assert_close(p, prices[:, grid])
                torch.testing.assert_close(r, radii[:, grid])

    def test_joint_q_value_updates_do_not_depend_on_actor_architecture(self):
        first = PairedIQL(self.config)
        second = deepcopy(first)
        second.actors["autoregressive"].autoregressive = False
        batch = {"states": torch.randn(8, 4), "actions": torch.randint(4, (8, 2)),
                 "rewards": torch.ones(8), "next_states": torch.randn(8, 4), "masks": torch.ones(8)}
        first.update(batch)
        second.update(batch)
        for name in ("q", "target_q", "value"):
            for a, b in zip(getattr(first, name).parameters(), getattr(second, name).parameters()):
                torch.testing.assert_close(a, b)

    def test_safe_checkpoint_exact_resume_and_factory_loading(self):
        first = PairedIQL(self.config)
        batch = {"states": torch.randn(8, 4), "actions": torch.randint(4, (8, 2)),
                 "rewards": torch.ones(8), "next_states": torch.randn(8, 4), "masks": torch.ones(8)}
        first.update(batch)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latest.pt"
            atomic_save(path, first.state_dict())
            saved = torch.load(path, weights_only=True, map_location="cpu")
        second = PairedIQL(self.config)
        second.load_state_dict(saved)
        torch.testing.assert_close(first.update(batch), second.update(batch))
        actor = make_actor(self.config, "autoregressive")
        actor.load_state_dict(first.actors["autoregressive"].state_dict())
        torch.testing.assert_close(actor.act(batch["states"]), first.actors["autoregressive"].act(batch["states"]))

    def test_hierarchical_interval_retains_training_seed_variation(self):
        rows = []
        for training_seed, delta in ((42, -10.), (43, 10.)):
            for seed in range(20):
                rows.extend([dict(training_seed=training_seed, seed=seed, policy="f3", platform_revenue=100.),
                             dict(training_seed=training_seed, seed=seed, policy="ar", platform_revenue=100.+delta)])
        stats = hierarchical_statistics(pd.DataFrame(rows), "ar", "f3", draws=1000)
        self.assertEqual(stats["training_initializations"], 2)
        self.assertEqual(stats["paired_simulations"], 40)
        self.assertEqual(stats["mean_difference_usd"], 0.)
        self.assertEqual(stats["ci95_low"], -10.)
        self.assertEqual(stats["ci95_high"], 10.)


if __name__ == "__main__":
    unittest.main()
