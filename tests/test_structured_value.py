"""时空联合价值的缓存、交互、梯度与断点一致性。"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from offline_rl.coordination import make_actor
from offline_rl.structured_value import StructuredIQL, StructuredTwinQ, StructuredValueActor
from offline_rl.train_structured import BatchPrefetch
from offline_rl.compact_replay import CompactReplay
from offline_rl.joint_starts import JointStartActor


class StructuredValueTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(52)
        self.config = dict(state_size=20, grid_count=2, history_steps=4, history_features=2,
            context_size=2, price_values=[1., 2.], radius_values=[1., 2.], hidden_size=16,
            memory_size=8, basis_size=4, rank=3)
        self.behavior = dict(state_size=20, grid_count=2, price_count=2, radius_count=2,
            price_values=[1., 2.], radius_values=[1., 2.], hidden_size=8, embedding_size=3)

    def test_cached_candidates_equal_repeated_forward_and_receive_temporal_gradients(self):
        model = StructuredTwinQ(**self.config)
        states, actions = torch.randn(3, 20), torch.randint(4, (3, 7, 2))
        cached = model.cached(model.prepare(states), actions)
        direct = model(states[:, None].expand(-1, 7, -1).reshape(-1, 20), actions.reshape(-1, 2)).reshape(2, 3, 7)
        torch.testing.assert_close(cached, direct, atol=1e-6, rtol=1e-5)
        cached.square().mean().backward()
        self.assertGreater(float(model.encoder.memory.weight_ih_l0.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None or p.grad.isfinite().all() for p in model.parameters()))

    @torch.no_grad()
    def test_joint_value_has_nonadditive_action_interactions(self):
        model = StructuredTwinQ(**self.config)
        state = torch.randn(1, 20)
        actions = torch.tensor([[[0, 0], [0, 3], [3, 0], [3, 3]]])
        values = model.cached(model.prepare(state), actions)
        cross = values[:, :, 3] - values[:, :, 2] - values[:, :, 1] + values[:, :, 0]
        self.assertGreater(float(cross.abs().max()), 1e-6)

    def test_finite_horizon_terminal_reward_does_not_bootstrap(self):
        learner = StructuredIQL(self.config)
        states = torch.randn(4, 20)
        batch = dict(states=states, actions=torch.randint(4, (4, 2)), rewards=torch.full((4,), 200.),
                     next_states=torch.randn(4, 20)*100, masks=torch.zeros(4))
        result = learner.update(batch)
        self.assertAlmostEqual(float(result[-1]), 2.)
        self.assertEqual(learner.discount, 1.)
        self.assertTrue(result.isfinite().all())
        self.assertTrue(all(not p.requires_grad for p in learner.target_q.parameters()))

    def test_complete_checkpoint_resumes_same_value_updates(self):
        first, second = StructuredIQL(self.config), StructuredIQL(self.config)
        batch = dict(states=torch.randn(5, 20), actions=torch.randint(4, (5, 2)), rewards=torch.ones(5),
                     next_states=torch.randn(5, 20), masks=torch.tensor([1., 1., 0., 1., 0.]))
        first.update(batch)
        from copy import deepcopy
        second.load_state_dict(deepcopy(first.state_dict()))
        torch.testing.assert_close(first.update(batch), second.update(batch))
        for name in ("q", "target_q", "value"):
            for a, b in zip(getattr(first, name).parameters(), getattr(second, name).parameters()):
                torch.testing.assert_close(a, b)

    @torch.no_grad()
    def test_exported_policy_uses_all_legal_actions_and_cached_critics(self):
        config = dict(behavior_config=self.behavior, critic_config=self.config, ensemble_size=2)
        first = StructuredValueActor(**config)
        restored = make_actor(config, "structured_value")
        restored.load_state_dict(first.state_dict())
        states = torch.randn(2, 20)
        torch.testing.assert_close(first.act(states), restored.act(states))
        self.assertTrue(((restored.act(states) >= 0) & (restored.act(states) < 4)).all())
        state, actions = states[:1], torch.randint(4, (7, 2))
        prepared, unused = restored._prepare(state)
        expected = torch.stack([q.minimum(state.expand(7, -1), actions) for q in restored.critics])
        torch.testing.assert_close(restored.cached_values(actions, prepared, unused), expected, atol=1e-6, rtol=1e-5)
        with self.assertRaises(ValueError):
            restored.act(states, [0], [0])

    def test_prefetch_preserves_sampling_and_resume_rng_exactly(self):
        class ReplayStub:
            count = 100

            def batch(self, indices):
                return {"indices": indices.copy()}
        replay, actual, expected = ReplayStub(), np.random.default_rng(62), np.random.default_rng(62)
        for _ in range(2):
            loader = BatchPrefetch(replay, actual, 8, workers=2, depth=4)
            for _ in range(5):
                indices, values = loader.next()
                np.testing.assert_array_equal(indices, expected.integers(100, size=8))
                np.testing.assert_array_equal(values["indices"], indices)
            loader.close()
            self.assertEqual(actual.bit_generator.state, expected.bit_generator.state)

    def test_compact_history_is_bitwise_identical_and_keeps_terminal_boundaries(self):
        rng = np.random.default_rng(52)
        frames = rng.normal(size=(3, 7, 2, 3)).astype("float32")
        context = rng.normal(size=(3, 4, 4)).astype("float32")
        states = np.stack([np.concatenate((frames[r, t:t+4].transpose(1, 0, 2).flatten(), context[r, t]))
                           for r in range(3) for t in range(4)])
        source = SimpleNamespace(periods=4, count=12, metadata={"grid_count": 2, "state_size": 28,
            "preprocessor": {"history_steps": 4, "history_features": ["a", "b", "c"]}},
            arrays={"states": states, "actions": np.zeros((12, 2), dtype="int16"), "rewards": np.arange(12, dtype="float32")})
        compact = CompactReplay(source, chunk_runs=2)
        np.testing.assert_array_equal(compact.states(np.arange(12)), states)
        batch = compact.batch(np.array([2, 3, 4, 11]))
        np.testing.assert_array_equal(batch["masks"], [1, 0, 1, 0])
        np.testing.assert_array_equal(batch["next_states"], np.stack((states[3], np.zeros(28), states[5], np.zeros(28))))
        self.assertLess(compact.bytes, sum(array.nbytes for array in source.arrays.values()))
        source.arrays["states"][1, 0] += 1
        with self.assertRaises(ValueError):
            CompactReplay(source)

    @torch.no_grad()
    def test_joint_start_search_preserves_original_and_uniform_objective_bounds(self):
        config = dict(behavior_config=self.behavior, critic_config=self.config, ensemble_size=2)
        original = StructuredValueActor(**config)
        joint = make_actor(config, "structured_joint_starts")
        joint.load_state_dict(original.state_dict())
        self.assertIsInstance(joint, JointStartActor)
        for state in torch.randn(4, 20):
            state = state[None]
            chosen, improved = original.act(state)[0], joint.act(state)[0]
            uniform = torch.arange(4)[:, None].expand(-1, 2)
            prior = joint.behavior_log_probs(state)[0]
            baseline = prior.argmax(-1)
            candidates = torch.cat((baseline[None], chosen[None], improved[None], uniform), 0)
            prepared, unused = joint._prepare(state)
            values = joint.cached_values(candidates, prepared, unused)
            difference = values-values[:, :1]
            scores = difference.mean(0)-joint.uncertainty_weight*difference.std(0, unbiased=False)
            scores += joint.behavior_weight*prior[torch.arange(2), candidates].sum(-1)
            self.assertGreaterEqual(float(scores[2]), float(scores[1])-1e-6)
            self.assertGreaterEqual(float(scores[2]), float(scores[3:].max())-1e-6)


if __name__ == "__main__":
    unittest.main()
