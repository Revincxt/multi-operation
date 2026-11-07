"""保守策略迭代的目标、反事实梯度、终止边界与恢复一致性。"""

from copy import deepcopy
import unittest

import torch

from offline_rl.conservative_policy import ConservativeActorCritic, conservative_gap, counterfactual_actions, masked_auxiliary_loss
from offline_rl.coordination import make_actor


class ConservativePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(72)
        self.actor = dict(state_size=20, grid_count=2, history_steps=4, history_features=2,
            context_size=2, price_values=[1., 2.], radius_values=[1., 2.], hidden_size=16, memory_size=8)
        self.critic = dict(**self.actor, basis_size=4, rank=3)
        self.batch = dict(states=torch.randn(5, 20), next_states=torch.randn(5, 20),
            actions=torch.randint(4, (5, 2)), rewards=torch.arange(5).float()*100,
            masks=torch.tensor([1., 0., 1., 1., 0.]))

    def learner(self):
        return ConservativeActorCritic(self.actor, self.critic, [16, 17])

    def test_counterfactual_retains_every_teammate_and_enumerates_own_support(self):
        actions = torch.tensor([[1, 3], [2, 1]])
        candidates = counterfactual_actions(actions, 0, 4)
        torch.testing.assert_close(candidates[:, :, 0], torch.arange(4)[None].expand(2, -1))
        torch.testing.assert_close(candidates[:, :, 1], actions[:, 1:2].expand(-1, 4))
        torch.testing.assert_close(actions, torch.tensor([[1, 3], [2, 1]]))

    def test_conservative_gradient_penalizes_unobserved_high_action_and_rewards_record(self):
        other = torch.tensor([[[0., 5., 0.]]], requires_grad=True)
        recorded = torch.zeros((1, 1), requires_grad=True)
        loss = conservative_gap(other, recorded)
        loss.backward()
        self.assertGreater(float(other.grad[0, 0, 1]), .99)
        self.assertLess(float(recorded.grad), 0.)

    def test_td_target_uses_deployed_target_actor_and_terminal_never_bootstraps(self):
        learner = self.learner()
        with torch.no_grad():
            actions = learner.target_actor.act(self.batch["next_states"])
            expected = self.batch["rewards"]/100+self.batch["masks"]*learner.target_q.minimum(self.batch["next_states"], actions)
        torch.testing.assert_close(learner.bellman_target(self.batch), expected)
        terminal = {**self.batch, "masks": torch.zeros(5)}
        torch.testing.assert_close(learner.bellman_target(terminal), self.batch["rewards"]/100)
        self.assertFalse(learner.bellman_target(self.batch).requires_grad)

    def test_terminal_auxiliary_keeps_reward_and_ignores_unavailable_next_state(self):
        predicted = torch.tensor([[100., 100., 0.]], requires_grad=True)
        loss = masked_auxiliary_loss(predicted, torch.zeros_like(predicted), torch.zeros(1))
        loss.backward()
        torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted))
        self.assertEqual(float(loss.detach()), 0.)

    def test_frozen_policy_evaluation_changes_q_but_never_actor_or_behavior(self):
        learner = self.learner()
        before = {name: deepcopy(getattr(learner, name).state_dict()) for name in ("q", "actor", "target_actor", "behavior")}
        output = learner.update(self.batch, 0, improve=False)
        self.assertTrue(output.isfinite().all())
        for name in ("actor", "target_actor", "behavior"):
            for key, value in getattr(learner, name).state_dict().items():
                torch.testing.assert_close(value, before[name][key])
        self.assertTrue(any(not torch.equal(value, before["q"][key]) for key, value in learner.q.state_dict().items()))

    def test_resumed_updates_match_and_export_matches_the_policy_used_in_td(self):
        first, second = self.learner(), self.learner()
        first.warmup(self.batch)
        first.finish_warmup()
        first.update(self.batch, 0)
        second.load_state_dict(deepcopy(first.state_dict()))
        torch.testing.assert_close(first.update(self.batch, 1), second.update(self.batch, 1))
        for name in first.state_dict()["models"]:
            for a, b in zip(getattr(first, name).parameters(), getattr(second, name).parameters()):
                torch.testing.assert_close(a, b)
        exported = make_actor(self.actor, "cooperative_distill")
        exported.load_state_dict(first.target_actor.state_dict())
        torch.testing.assert_close(exported.act(self.batch["states"]), first.target_actor.act(self.batch["states"]))
        for model in (first.target_actor, first.target_q, first.behavior):
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))

    def test_joint_regularization_dual_tracks_constraint_violation_in_both_directions(self):
        high = ConservativeActorCritic(self.actor, self.critic, [16, 17], policy_gap_limit=.05)
        low = ConservativeActorCritic(self.actor, self.critic, [16, 17], policy_gap_limit=.05)
        initial = float(high.log_joint_weight.detach())
        for _ in range(20):
            high.update_dual(torch.tensor(.5))
            low.update_dual(torch.tensor(-.5))
        self.assertGreater(float(high.log_joint_weight.detach()), initial)
        self.assertLess(float(low.log_joint_weight.detach()), initial)

    def test_joint_penalty_and_dual_resume_identically(self):
        first = ConservativeActorCritic(self.actor, self.critic, [16, 17], policy_gap_limit=.05)
        second = ConservativeActorCritic(self.actor, self.critic, [16, 17], policy_gap_limit=.05)
        first.update(self.batch, 0)
        second.load_state_dict(deepcopy(first.state_dict()))
        torch.testing.assert_close(first.update(self.batch, 1), second.update(self.batch, 1))
        torch.testing.assert_close(first.log_joint_weight, second.log_joint_weight)


if __name__ == "__main__":
    unittest.main()
