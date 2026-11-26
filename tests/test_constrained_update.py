"""约束成本、价值单位、梯度隔离和恢复可重复性的检查。"""

from copy import deepcopy
import unittest

import numpy as np
import torch

from constrained_rl.policy import ConstrainedPolicy
from constrained_rl.ppo import constraint_costs, update, update_multipliers, warm_cost_heads
from online_rl.policy import JointPolicy


class ConstrainedUpdateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        self.model_config = dict(grid_count=2, history_steps=2, history_features=2, context_size=1,
            price_count=2, radius_count=2, hidden_size=16, memory_size=8, heads=2, layers=1)
        self.model = ConstrainedPolicy(**self.model_config).eval()
        self.config = dict(gamma=1., gae_lambda=.95, reward_scale=7000., epochs=2, batch_size=4,
            clip=.1, target_kl=.1, max_grad_norm=.5, value_weight=.5, entropy_weight=.001,
            cost_value_weight=.5, dual_learning_rate=5., cost_warmup_epochs=25,
            cost_warmup_learning_rate=.002, value_scale_floor=.05, cost_scale_floor=.01)

    def episodes(self, model=None, seed=51):
        model = self.model if model is None else model
        states = torch.randn(8, 10, generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            actions, logp, value = model.sample(states, torch.Generator().manual_seed(seed + 1))
            costs = model.cost_values(states)
        return [{"states": states[i:i+4].numpy(), "actions": actions[i:i+4].numpy(),
                 "log_probs": logp[i:i+4].numpy(), "values": value[i:i+4].numpy(),
                 "cost_values": costs[i:i+4].numpy(), "rewards": np.array([90., 100., 110., 120.], dtype="float32"),
                 "costs": np.array([[.02, -.01], [.01, .01], [-.01, .01], [.03, -.02]], dtype="float32"),
                 "terminals": np.array([0., 0., 0., 1.], dtype="float32")}
                for i in (0, 4)]

    def test_original_policy_and_dollar_values_are_preserved_when_importing(self):
        original = JointPolicy(**self.model_config).eval()
        self.model.load_original(original.state_dict(), reward_scale=7000.)
        states = torch.randn(7, 10)
        first = original.sample(states, torch.Generator().manual_seed(73))
        second = self.model.sample(states, torch.Generator().manual_seed(73))
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
        torch.testing.assert_close(first[2] * 100., second[2] * 7000., rtol=1e-5, atol=1e-4)
        with self.assertRaises(ValueError):
            self.model.load_original(self.model.state_dict())

    def test_costs_sum_to_global_constraints_and_do_not_depend_on_period_partition(self):
        values = dict(service_floor=.6, wait_limit_seconds_per_demand=90., demand_scale=100., wait_scale_seconds=100.)
        costs = constraint_costs([50., 25., 25.], [10., 20., 25.], [6000., 2000., 1500.], **values)
        np.testing.assert_allclose(costs.sum(0), [.05, .05], atol=1e-7)
        np.testing.assert_allclose(costs.sum(0), constraint_costs(100., 55., 9500., **values), atol=1e-7)
        # 有积压时本期匹配可以大于本期新增需求；不能用逐期率代替可加量。
        self.assertTrue(np.isfinite(constraint_costs(0., 5., 300., **values)).all())

    def test_multiplier_increases_on_violation_decreases_on_slack_and_projects(self):
        multipliers = np.array([.1, .2])
        update_multipliers(multipliers, [[.2, -.3], [.4, -.1]], 2.)
        np.testing.assert_allclose(multipliers, [.7, 0.])
        update_multipliers(multipliers, [[1., 1.]], 10., constrained=False)
        np.testing.assert_array_equal(multipliers, [0., 0.])

    def test_cost_value_loss_has_no_actor_gradient(self):
        self.model.zero_grad(set_to_none=True)
        value = self.model.cost_values(torch.randn(4, 10))
        (value - 1.).square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in self.model.actor_parameters()))
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in self.model.cost_head.parameters()))
        actor = {id(value) for value in self.model.actor_parameters()}
        critic = {id(value) for value in self.model.critic_parameters()}
        self.assertFalse(actor & critic)
        self.assertEqual(actor | critic, {id(value) for value in self.model.parameters()})

    def test_cost_warmup_fits_heads_without_changing_original_weights(self):
        episodes = self.episodes()
        before = deepcopy(self.model.state_dict())
        result = warm_cost_heads(self.model, episodes, self.config, np.random.default_rng(71), "cpu")
        self.assertLess(result["warm_cost_loss_after"], result["warm_cost_loss_before"])
        for name, value in before.items():
            if not name.startswith("cost_head."):
                torch.testing.assert_close(value, self.model.state_dict()[name], rtol=0, atol=0)

    def test_finite_update_uses_fixed_multipliers_then_dual_update(self):
        episodes = self.episodes()
        multipliers = np.array([.5, .5])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        result = update(self.model, optimizer, episodes, self.config, np.random.default_rng(81), "cpu", multipliers)
        self.assertTrue(np.isfinite(list(result.values())).all())
        self.assertEqual(result["service_lambda_used"], .5)
        self.assertEqual(result["waiting_lambda_used"], .5)
        self.assertAlmostEqual(result["service_lambda_next"], .75, places=6)
        self.assertAlmostEqual(result["waiting_lambda_next"], .45, places=6)
        self.assertEqual(result["effective_epochs"], 2.)
        self.assertEqual(result["sample_visits"], 16)

    def test_zero_multiplier_control_has_identical_first_network_update(self):
        other = deepcopy(self.model)
        episodes = self.episodes()
        first = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        second = torch.optim.Adam(other.parameters(), lr=1e-4)
        multipliers, control = np.zeros(2), np.zeros(2)
        update(self.model, first, episodes, self.config, np.random.default_rng(81), "cpu", multipliers, True)
        update(other, second, episodes, self.config, np.random.default_rng(81), "cpu", control, False)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)
        self.assertGreater(multipliers[0], 0.)
        np.testing.assert_array_equal(control, [0., 0.])

    def test_constraint_advantage_reduces_probability_of_costly_action(self):
        states = torch.zeros(256, self.model.state_size)
        with torch.no_grad():
            actions, logp, _ = self.model.sample(states, torch.Generator().manual_seed(42))
            before = self.model.conditional_log_probs(self.model.actor_encoder(states[:1]),
                                                     torch.zeros((1, 2), dtype=torch.long)).exp()[0, 0, 0]
        terminal = np.zeros(256, dtype="float32")
        terminal[-1] = 1.
        costs = np.zeros((256, 2), dtype="float32")
        costs[:, 0] = (actions[:, 0].numpy() == 0).astype("float32")
        episode = {"states": states.numpy(), "actions": actions.numpy(), "log_probs": logp.numpy(),
                   "values": np.zeros(256, dtype="float32"), "rewards": np.zeros(256, dtype="float32"),
                   "cost_values": np.zeros_like(costs), "costs": costs, "terminals": terminal}
        config = dict(self.config, gae_lambda=0., batch_size=256, epochs=1,
                      entropy_weight=0., value_weight=0., cost_value_weight=0.)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        update(self.model, optimizer, [episode], config, np.random.default_rng(81), "cpu", np.array([1., 0.]))
        with torch.no_grad():
            after = self.model.conditional_log_probs(self.model.actor_encoder(states[:1]),
                                                    torch.zeros((1, 2), dtype=torch.long)).exp()[0, 0, 0]
        self.assertLess(float(after), float(before))

    def test_resume_restores_optimizer_rng_and_multipliers_exactly(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        rng, multipliers = np.random.default_rng(81), np.array([.5, .5])
        update(self.model, optimizer, self.episodes(), self.config, rng, "cpu", multipliers)
        other = deepcopy(self.model)
        second = torch.optim.Adam(other.parameters(), lr=1e-4)
        second.load_state_dict(deepcopy(optimizer.state_dict()))
        other_rng = np.random.default_rng()
        other_rng.bit_generator.state = deepcopy(rng.bit_generator.state)
        second_multipliers = multipliers.copy()
        fresh = self.episodes(seed=74)
        result = update(self.model, optimizer, fresh, self.config, rng, "cpu", multipliers)
        repeated = update(other, second, fresh, self.config, other_rng, "cpu", second_multipliers)
        for name in result:
            self.assertAlmostEqual(result[name], repeated[name], places=7)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)
        np.testing.assert_array_equal(multipliers, second_multipliers)


if __name__ == "__main__":
    unittest.main()
