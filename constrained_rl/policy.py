"""在冻结 AR-PPO 结构上增加两个独立输出的约束价值头。"""

import math

import torch
from torch import nn

from online_rl.policy import JointPolicy


class ConstrainedPolicy(JointPolicy):
    """保留联合动作分布；收益和约束价值共享独立于 Actor 的编码器。"""

    def __init__(self, **config):
        super().__init__(**config)
        hidden = self.value_head[0].in_features
        self.cost_head = nn.Sequential(nn.Linear(hidden, 128), nn.Tanh(), nn.Linear(128, 2))
        nn.init.zeros_(self.cost_head[-1].weight)
        nn.init.zeros_(self.cost_head[-1].bias)

    def critic_values(self, states):
        encoded = self.critic_encoder(states).flatten(1)
        return self.value_head(encoded).squeeze(-1), self.cost_head(encoded)

    def cost_values(self, states):
        return self.cost_head(self.critic_encoder(states).flatten(1))

    def evaluate_all(self, states, actions):
        logs = self.conditional_log_probs(self.actor_encoder(states), actions)
        selected = logs.gather(-1, actions[..., None]).squeeze(-1)
        entropy = -(logs.exp() * logs).sum(-1).sum(-1)
        value, costs = self.critic_values(states)
        return selected.sum(-1), entropy, value, costs

    def actor_parameters(self):
        excluded = {id(parameter) for parameter in self.critic_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in excluded]

    def critic_parameters(self):
        return [parameter for module in (self.critic_encoder, self.value_head, self.cost_head)
                for parameter in module.parameters()]

    @torch.no_grad()
    def load_original(self, state_dict, reward_scale=100., original_reward_scale=100.):
        """导入原版权重；转换价值单位，但保持动作分布和美元价值预测不变。"""
        if not all(math.isfinite(value) and value > 0 for value in (reward_scale, original_reward_scale)):
            raise ValueError
        expected = set(self.state_dict())
        additions = {name for name in expected if name.startswith("cost_head.")}
        if set(state_dict) != expected - additions:
            raise ValueError
        result = self.load_state_dict(state_dict, strict=False)
        if set(result.missing_keys) != additions or result.unexpected_keys:
            raise ValueError
        factor = original_reward_scale / reward_scale
        self.value_head[-1].weight.mul_(factor)
        self.value_head[-1].bias.mul_(factor)

