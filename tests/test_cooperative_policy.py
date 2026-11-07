"""轻量联合策略学习的概率、缓存数值及冻结价值约束。"""

import unittest

import torch

from offline_rl.cooperative_policy import CooperativeActor, CooperativeDistillation, RefinedCooperativeActor, behavior_log_probs, ensemble_values
from offline_rl.coordination import make_actor
from offline_rl.value_guided import ValueGuidedActor


class CooperativePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.behavior = dict(state_size=20, grid_count=2, price_count=2, radius_count=2,
            price_values=[1., 2.], radius_values=[1., 2.], hidden_size=8, embedding_size=3)
        self.config = dict(state_size=20, grid_count=2, history_steps=4, history_features=2,
            context_size=2, price_values=[1., 2.], radius_values=[1., 2.], hidden_size=16, memory_size=8)

    def test_full_joint_probabilities_and_temporal_gradients(self):
        actor = CooperativeActor(**self.config)
        probabilities = actor.log_probs(torch.randn(5, 20))
        torch.testing.assert_close(probabilities.exp().sum(-1), torch.ones(5, 2))
        (-probabilities[:, :, 0].mean()).backward()
        self.assertTrue(all(p.grad is None or p.grad.isfinite().all() for p in actor.parameters()))
        self.assertGreater(float(actor.memory.weight_ih_l0.grad.abs().sum()), 0.)

    def test_batched_value_cache_matches_explicit_critic(self):
        model = ValueGuidedActor(self.behavior, ensemble_size=2)
        states, actions = torch.randn(3, 20), torch.randint(4, (3, 7, 2))
        actual = ensemble_values(model, states, actions)
        expected = torch.stack([q.minimum(states[:, None].expand(-1, 7, -1).reshape(-1, 20),
            actions.reshape(-1, 2)).reshape(3, 7) for q in model.critics])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_behavior_probability_cache_is_exact(self):
        model = ValueGuidedActor(self.behavior, ensemble_size=2)
        states = torch.randn(3, 20)
        torch.testing.assert_close(behavior_log_probs(model.behavior, states), model.behavior_log_probs(states))

    def test_policy_updates_without_modifying_frozen_critics(self):
        model = ValueGuidedActor(self.behavior, ensemble_size=2)
        actor = CooperativeActor(**self.config)
        learner = CooperativeDistillation(actor, model)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = learner.update(torch.randn(5, 20), grid=0)
        self.assertTrue(result.isfinite().all())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    @torch.no_grad()
    def test_refined_export_and_batch_actions_preserve_base_improvement(self):
        base = ValueGuidedActor(self.behavior, ensemble_size=2)
        refined = RefinedCooperativeActor(proposal_config=self.config, behavior_config=self.behavior, ensemble_size=2)
        refined.load_state_dict(base.state_dict(), strict=False)
        restored = make_actor(dict(proposal_config=self.config, behavior_config=self.behavior, ensemble_size=2), "cooperative_refined")
        restored.load_state_dict(refined.state_dict())
        states = torch.randn(3, 20)
        original, improved = base.act(states), restored.act(states)
        torch.testing.assert_close(improved, torch.cat([restored.act(state[None]) for state in states]))
        for index, state in enumerate(states):
            prior = base.behavior_log_probs(state[None])[0]
            baseline = prior.argmax(-1)
            values = ensemble_values(base, state[None], torch.stack((baseline, original[index], improved[index]))[None])
            difference = values[:, 0] - values[:, 0, :1]
            score = difference.mean(0) - base.uncertainty_weight * difference.std(0, unbiased=False)
            score += base.behavior_weight * prior[torch.arange(2), torch.stack((baseline, original[index], improved[index]))].sum(-1)
            self.assertGreaterEqual(float(score[2]), float(score[1])-1e-6)


if __name__ == "__main__":
    unittest.main()
