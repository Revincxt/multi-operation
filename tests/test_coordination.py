"""状态广播、条件动作、离线局部信用与采用门槛的回归测试。"""

from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from offline_rl.coordination import ConditionalActor, OMIGA, broadcast_indices, make_actor, omiga_value_loss
from offline_rl.optimize import passes_gate
from offline_rl.train import atomic_save


class CoordinationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.metadata = {
            "grid_count": 2,
            "preprocessor": {"history_steps": 2, "history_features": ["x", "price_multiplier", "matching_radius_km"]},
            "state_context_columns": ["decision_start_sim_time", "interval_minutes", "start_waiting_request_num",
                                      "start_idle_driver_num", "start_busy_driver_num", "start_repositioning_driver_num"],
            "local_state_indices": [list(range(6)) + list(range(12, 18)),
                                    list(range(6, 12)) + list(range(18, 24))],
        }
        self.config = dict(state_size=24, grid_count=2, price_count=3, radius_count=2,
                           local_indices=broadcast_indices(self.metadata), hidden_size=8, embedding_size=3)

    def test_other_region_broadcast_reaches_policy_without_current_actions(self):
        indices = self.config["local_indices"]
        self.assertEqual(indices[0][-6:], [20, 21, 22, 23, 10, 11])
        actor = make_actor(self.config)
        first = torch.zeros(2, 24)
        second = first.clone()
        second[:, 20:24] = 10
        p1, r1 = actor(first)
        p2, r2 = actor(second)
        self.assertGreater(float((p1[:, 0] - p2[:, 0]).abs().sum()), 0)
        self.assertEqual(p1.shape, (2, 2, 3))
        self.assertEqual(r1.shape, (2, 2, 2))

    def test_conditional_inference_uses_the_price_chosen_after_mask(self):
        actor = ConditionalActor(**self.config)
        with torch.no_grad():
            for agent in actor.agents:
                for parameter in agent.parameters():
                    parameter.zero_()
                agent.price_head.bias[0] = 10
                agent.price_embedding.weight[1] = 1
                agent.radius_head.weight[0, 8:] = -1
                agent.radius_head.weight[1, 8:] = 1
        states = torch.zeros(2, 24)
        self.assertTrue(actor.act(states).eq(0).all())
        self.assertTrue(actor.act(states, allowed_prices=[1]).eq(3).all())
        self.assertTrue(actor.act(states, allowed_prices=[1], allowed_radii=[0]).eq(2).all())

    def test_conditional_joint_likelihood_is_normalized_and_has_two_head_gradients(self):
        config = {**self.config, "grid_count": 1, "local_indices": [self.config["local_indices"][0]]}
        actor = ConditionalActor(**config)
        states = torch.randn(1, 24).expand(6, -1)
        actions = torch.arange(6)[:, None]
        logp = actor.log_prob(states, actions)
        self.assertAlmostEqual(float(logp.exp().sum().detach()), 1., places=5)
        (-logp.mean()).backward()
        self.assertGreater(float(actor.agents[0].price_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(actor.agents[0].radius_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(actor.agents[0].price_embedding.weight.grad.abs().sum()), 0)

    def test_omiga_assigns_distinct_local_weights_and_correct_value_gradient(self):
        q = torch.tensor([[2., -2.]])
        v = torch.zeros(1, 2, requires_grad=True)
        w = torch.tensor([[1., 2.]])
        loss, weights = omiga_value_loss(q, v, w, 1.)
        loss.backward()
        expected = w / 2 * np.exp(-2) * (1 - (w * q).exp())
        torch.testing.assert_close(v.grad, expected)
        self.assertFalse(weights.requires_grad)
        self.assertGreater(float(weights[0, 0]), float(weights[0, 1]))

    def test_omiga_updates_and_safe_checkpoint_roundtrip(self):
        learner = OMIGA(self.config)
        batch = {"states": torch.randn(8, 24), "actions": torch.randint(6, (8, 2)),
                 "rewards": torch.ones(8), "next_states": torch.randn(8, 24), "masks": torch.ones(8)}
        learner.update(batch)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latest.pt"
            atomic_save(path, learner.state_dict())
            saved = torch.load(path, map_location="cpu", weights_only=True)
        second = OMIGA(self.config)
        second.load_state_dict(saved)
        torch.testing.assert_close(learner.update(batch), second.update(batch))
        for module in (learner.actor, learner.q, learner.value, learner.mixer):
            self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))
        self.assertFalse(any(p.requires_grad for p in learner.target_q.parameters()))

    def test_adoption_requires_validation_gain_and_positive_test_interval(self):
        val = {"relative_percent": .7}
        test = {"mean_difference_usd": 50., "ci95_low": 10.}
        self.assertTrue(passes_gate(val, test))
        self.assertFalse(passes_gate({"relative_percent": .4}, test))
        self.assertFalse(passes_gate(val, {**test, "ci95_low": -1.}))
        self.assertFalse(passes_gate(val, {**test, "mean_difference_usd": -1.}))


if __name__ == "__main__":
    unittest.main()
