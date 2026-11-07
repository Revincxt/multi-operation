"""有界联合价值提取的数值一致性、概率和完整动作支持。"""

import unittest
from unittest.mock import patch

import torch

from offline_rl.value_guided import ValueGuidedActor


class ValueGuidedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = dict(state_size=4, grid_count=2, price_count=2, radius_count=2,
            price_values=[1., 2.], radius_values=[1., 2.], hidden_size=8, embedding_size=3)
        self.actor = ValueGuidedActor(self.config, ensemble_size=2)

    def test_cached_q_matches_original_network(self):
        state = torch.randn(1, 4)
        actions = torch.randint(4, (17, 2))
        networks, biases = self.actor._prepare(state)
        actual = self.actor.cached_values(actions, networks, biases)
        expected = torch.stack([q.minimum(state.expand(len(actions), -1), actions) for q in self.actor.critics])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_cached_behavior_preserves_joint_conditional_likelihood(self):
        states = torch.randn(3, 4)
        prior = self.actor.behavior_log_probs(states)
        torch.testing.assert_close(prior.exp().sum(-1), torch.ones(3, 2))
        actions = torch.randint(4, (3, 2))
        actual = prior.gather(-1, actions[..., None]).squeeze(-1).sum(-1)
        torch.testing.assert_close(actual, self.actor.behavior.log_prob(states, actions))

    def test_all_legal_actions_and_repeated_grid_actions_can_be_selected(self):
        self.actor.behavior_weight = 0.
        with patch.object(self.actor, "cached_values", side_effect=lambda actions, *args:
                          actions.float().sum(1)[None].expand(2, -1)):
            action = self.actor.act(torch.zeros(1, 4))
        self.assertTrue(action.eq(3).all())
        with self.assertRaises(ValueError):
            self.actor.act(torch.zeros(1, 4), allowed_prices=[0])

    def test_search_does_not_reduce_its_declared_objective(self):
        state = torch.randn(1, 4)
        chosen = self.actor.act(state)
        prior = self.actor.behavior_log_probs(state)[0]
        baseline = prior.argmax(-1)[None]
        networks, biases = self.actor._prepare(state)
        delta = self.actor.cached_values(chosen, networks, biases) - self.actor.cached_values(baseline, networks, biases)
        value = delta.mean() - delta.std(unbiased=False) + self.actor.behavior_weight * (
            prior.gather(-1, chosen.T).sum() - prior.gather(-1, baseline.T).sum())
        self.assertGreaterEqual(float(value.detach()), -1e-6)


if __name__ == "__main__":
    unittest.main()
