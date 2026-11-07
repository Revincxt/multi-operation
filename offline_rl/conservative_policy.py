"""当前策略评估、反事实保守Q和逐区域策略改进；保持全部合法动作。"""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from offline_rl.cooperative_policy import CooperativeActor
from offline_rl.iql import mlp
from offline_rl.structured_value import StructuredTwinQ


def counterfactual_actions(actions, grid, action_count):
    """只替换一个网格，队友动作保持输入值；不是枚举全城联合空间。"""
    candidates = actions[:, None].expand(-1, action_count, -1).clone()
    candidates[:, :, grid] = torch.arange(action_count, device=actions.device)
    return candidates


def conservative_gap(values, recorded, temperature=.1):
    # 减去log(A)仅去掉常数，不改变有限动作CQL的梯度。
    return (temperature*(torch.logsumexp(values/temperature, -1)-math.log(values.shape[-1]))-recorded).mean()


def masked_auxiliary_loss(predictions, targets, masks):
    weights = masks[:, None].expand_as(targets).clone()
    weights[:, -1] = 1.  # 终止转移的即时收益仍然有效。
    errors = F.smooth_l1_loss(predictions, targets, reduction="none")
    return (errors*weights).sum()/weights.sum().clamp_min(1.)


class DynamicsTwinQ(StructuredTwinQ):
    """沿用非可加联合Q，增加动作条件化的物理动态辅助任务。"""

    def __init__(self, auxiliary_size, **config):
        super().__init__(**config)
        self.auxiliary = mlp(config["grid_count"]*(config["hidden_size"]+5), 128, auxiliary_size)

    def prepare_with_auxiliary(self, states, actions):
        tokens, basis = self.encoder(states), self.basis()
        prepared = [head.prepare(tokens, basis) for head in self.heads]
        prediction = self.auxiliary(torch.cat((tokens.flatten(1), self.numeric_actions[actions].flatten(1)), -1))
        return prepared, prediction


class ConservativeActorCritic:
    """CFCQL式训练正则与确定性策略评估；不是原论文的逐行复现。"""

    metrics = ("td_loss", "cf_gap", "q_auxiliary_loss", "actor_loss", "actor_auxiliary_loss",
               "actor_kl", "target_mean", "td_mae", "policy_q_gap", "joint_conservative_weight")

    def __init__(self, actor_config, critic_config, auxiliary_indices, device="cpu",
                 discount=1., reward_scale=100., conservative_weight=.05, conservative_temperature=.1,
                 behavior_weight=.0001, entropy_weight=.002, auxiliary_weight=.02,
                 critic_learning_rate=1e-4, actor_learning_rate=3e-4, target_tau=.005,
                 policy_gap_limit=None, dual_learning_rate=.003):
        self.actor = CooperativeActor(**actor_config).to(device)
        self.behavior = deepcopy(self.actor).requires_grad_(False).eval()
        self.target_actor = deepcopy(self.actor).requires_grad_(False).eval()
        self.q = DynamicsTwinQ(len(auxiliary_indices)+1, **critic_config).to(device)
        self.target_q = deepcopy(self.q).requires_grad_(False).eval()
        self.actor_auxiliary = mlp(actor_config["grid_count"]*(actor_config["hidden_size"]+5), 128,
                                   len(auxiliary_indices)+1).to(device)
        self.optimizers = {"actor": torch.optim.Adam(list(self.actor.parameters())+list(self.actor_auxiliary.parameters()), lr=actor_learning_rate),
                           "q": torch.optim.Adam(self.q.parameters(), lr=critic_learning_rate)}
        self.auxiliary_indices = auxiliary_indices
        self.discount, self.reward_scale = discount, reward_scale
        self.conservative_weight, self.conservative_temperature = conservative_weight, conservative_temperature
        self.behavior_weight, self.entropy_weight = behavior_weight, entropy_weight
        self.auxiliary_weight, self.target_tau = auxiliary_weight, target_tau
        self.action_count = len(actor_config["price_values"])*len(actor_config["radius_values"])
        self.policy_gap_limit = policy_gap_limit
        if policy_gap_limit is not None:
            self.log_joint_weight = nn.Parameter(torch.tensor(math.log(.05), device=device))
            self.optimizers["dual"] = torch.optim.Adam([self.log_joint_weight], lr=dual_learning_rate)

    def auxiliary_targets(self, batch):
        return torch.cat((batch["next_states"][:, self.auxiliary_indices], batch["rewards"][:, None]/self.reward_scale), -1)

    def warmup(self, batch):
        """仅用训练数据初始化行为先验与策略表示。"""
        states, actions = batch["states"], batch["actions"]
        loss = -self.actor.log_probs(states).gather(-1, actions[..., None]).squeeze(-1).mean()
        auxiliary = self.actor_auxiliary(torch.cat((self.actor.encode(states).transpose(0, 1).flatten(1),
                                                  self.q.numeric_actions[actions].flatten(1)), -1))
        aux_loss = masked_auxiliary_loss(auxiliary, self.auxiliary_targets(batch), batch["masks"])
        self._step("actor", loss+self.auxiliary_weight*aux_loss)
        return torch.stack((loss.detach(), aux_loss.detach()))

    def finish_warmup(self):
        self.behavior.load_state_dict(self.actor.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

    @torch.no_grad()
    def bellman_target(self, batch):
        # 目标与最终导出的target_actor使用相同的联合MAP执行方式。
        actions = self.target_actor.act(batch["next_states"])
        future = self.target_q.minimum(batch["next_states"], actions)
        return batch["rewards"]/self.reward_scale+self.discount*batch["masks"]*future

    def update(self, batch, grid, improve=True):
        states, actions = batch["states"], batch["actions"]
        target = self.bellman_target(batch)
        prepared, auxiliary = self.q.prepare_with_auxiliary(states, actions)
        candidates = counterfactual_actions(actions, grid, self.action_count)
        extra = actions[:, None, :][:, :0]
        if self.policy_gap_limit is not None:
            with torch.no_grad():
                extra = torch.stack((self.actor.act(states), self.target_actor.act(states)), 1)
        values = self.q.cached(prepared, torch.cat((actions[:, None], candidates, extra), 1))
        recorded, other = values[:, :, 0], values[:, :, 1:1+self.action_count]
        errors = recorded-target[None]
        td_loss = errors.square().mean(1).sum()
        cf_loss = conservative_gap(other, recorded, self.conservative_temperature)
        aux_loss = masked_auxiliary_loss(auxiliary, self.auxiliary_targets(batch), batch["masks"])
        joint_gap = joint_weight = states.new_zeros(())
        if self.policy_gap_limit is not None:
            joint_gap = (values[:, :, 1+self.action_count:]-recorded[:, :, None]).mean()
            joint_weight = self.log_joint_weight.exp().detach()
        objective = td_loss+self.conservative_weight*cf_loss+self.auxiliary_weight*aux_loss
        if self.policy_gap_limit is not None:
            objective = objective+joint_weight*(joint_gap-self.policy_gap_limit)
        self._step("q", objective)
        if self.policy_gap_limit is not None:
            self.update_dual(joint_gap.detach())
        actor_loss = actor_auxiliary = kl = states.new_zeros(())
        if improve:
            with torch.no_grad():
                # 每个更新读取最新队友策略，以全城Q而非本区即时收益作改进。
                teammates = self.actor.act(states)
                candidates = counterfactual_actions(teammates, grid, self.action_count)
                scores = self.q.cached(self.q.prepare(states), candidates).min(0).values
                prior = self.behavior.log_probs(states)[:, grid]
                logits = (scores-scores.mean(-1, keepdim=True)+self.behavior_weight*prior)/(self.entropy_weight+self.behavior_weight)
                policy_target = F.softmax(logits, -1)
            log_policy = self.actor.log_probs(states)[:, grid]
            actor_loss = -(policy_target*log_policy).sum(-1).mean()
            kl = (log_policy.exp()*(log_policy-prior)).sum(-1).mean().detach()
            auxiliary = self.actor_auxiliary(torch.cat((self.actor.encode(states).transpose(0, 1).flatten(1),
                                                       self.q.numeric_actions[actions].flatten(1)), -1))
            actor_auxiliary = masked_auxiliary_loss(auxiliary, self.auxiliary_targets(batch), batch["masks"])
            self._step("actor", actor_loss+self.auxiliary_weight*actor_auxiliary)
        with torch.no_grad():
            for old, new in zip(self.target_q.parameters(), self.q.parameters()):
                old.lerp_(new, self.target_tau)
            if improve:
                for old, new in zip(self.target_actor.parameters(), self.actor.parameters()):
                    old.lerp_(new, self.target_tau)
        return torch.stack((td_loss.detach(), cf_loss.detach(), aux_loss.detach(), actor_loss.detach(),
                            actor_auxiliary.detach(), kl, target.mean(), errors.detach().abs().mean(),
                            joint_gap.detach(), joint_weight))

    def update_dual(self, gap):
        self.optimizers["dual"].zero_grad(set_to_none=True)
        loss = -self.log_joint_weight.exp()*(gap.detach()-self.policy_gap_limit)
        loss.backward()
        self.optimizers["dual"].step()
        with torch.no_grad():
            self.log_joint_weight.clamp_(-10., math.log(100.))

    def _step(self, name, loss):
        parameters = list(self.q.parameters()) if name == "q" else list(self.actor.parameters())+list(self.actor_auxiliary.parameters())
        self.optimizers[name].zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 10.)
        self.optimizers[name].step()

    def state_dict(self):
        result = {"models": {name: getattr(self, name).state_dict() for name in
                ("actor", "behavior", "target_actor", "q", "target_q", "actor_auxiliary")},
                "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}}
        if self.policy_gap_limit is not None:
            result["log_joint_weight"] = self.log_joint_weight.detach().clone()
        return result

    def load_state_dict(self, state):
        for name, values in state["models"].items():
            getattr(self, name).load_state_dict(values)
        for name, values in state["optimizers"].items():
            self.optimizers[name].load_state_dict(values)
        if self.policy_gap_limit is not None:
            with torch.no_grad():
                self.log_joint_weight.copy_(state["log_joint_weight"])
