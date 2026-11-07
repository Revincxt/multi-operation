"""集成联合Q、行为正则和有界坐标改进的离线策略提取。"""

import torch
from torch import nn
from torch.nn import functional as F

from offline_rl.iql import TwinQ
from offline_rl.joint_policy import JointActor


class ValueGuidedActor(nn.Module):
    """保留全部合法动作；用全城联合价值评分，不枚举全城笛卡尔积。"""

    def __init__(self, behavior_config, ensemble_size=3, uncertainty_weight=1.,
                 behavior_weight=.001, sweeps=2, multistart=True):
        super().__init__()
        self.behavior = JointActor(**behavior_config)
        self.grid_count = behavior_config["grid_count"]
        self.action_count = behavior_config["price_count"] * behavior_config["radius_count"]
        config = {key: behavior_config[key] for key in (
            "state_size", "grid_count", "price_count", "radius_count", "hidden_size", "embedding_size")}
        self.critics = nn.ModuleList([TwinQ(**config) for _ in range(ensemble_size)])
        self.uncertainty_weight, self.behavior_weight = uncertainty_weight, behavior_weight
        self.sweeps, self.multistart = sweeps, multistart
        self._codebooks = None

    def _prepare(self, states):
        networks = [head for critic in self.critics for head in (critic.q1, critic.q2)]
        if self._codebooks is None:
            self._codebooks = []
            for head in networks:
                first = head.network[0]
                embedding = head.embedding.weight.reshape(self.grid_count, self.action_count, -1)
                weights = first.weight[:, states.shape[-1]:].reshape(first.out_features, self.grid_count, -1)
                self._codebooks.append(torch.einsum("gae,hge->gah", embedding, weights))
        biases = [F.linear(states, head.network[0].weight[:, :states.shape[-1]], head.network[0].bias)
                  for head in networks]
        return networks, biases

    def cached_values(self, actions, networks, biases):
        grids = torch.arange(self.grid_count, device=actions.device)
        values = []
        for head, codes, bias in zip(networks, self._codebooks, biases):
            hidden = F.relu(codes[grids, actions].sum(1) + bias)
            hidden = F.relu(F.linear(hidden, head.network[2].weight, head.network[2].bias))
            values.append(F.linear(hidden, head.network[4].weight, head.network[4].bias).squeeze(-1))
        values = torch.stack(values).reshape(len(self.critics), 2, len(actions))
        return values.min(1).values

    def behavior_log_probs(self, states):
        prices, radii = self.behavior.price_count, self.behavior.radius_count
        repeated = states.repeat_interleave(prices, dim=0)
        labels = torch.arange(prices, device=states.device).repeat(len(states))[:, None].expand(-1, self.grid_count) * radii
        p, r = self.behavior.logits(repeated, labels)
        p = F.log_softmax(p, -1).reshape(len(states), prices, self.grid_count, prices)[:, 0]
        r = F.log_softmax(r, -1).reshape(len(states), prices, self.grid_count, radii).transpose(1, 2)
        return (p[..., None] + r).flatten(2)

    def extra_starts(self, states):
        return []

    @torch.no_grad()
    def act(self, states, allowed_prices=None, allowed_radii=None):
        if allowed_prices is not None or allowed_radii is not None:
            raise ValueError
        if len(states) != 1:
            return torch.cat([self.act(state[None]) for state in states], dim=0)
        networks, biases = self._prepare(states)
        prior = self.behavior_log_probs(states)[0]
        grids = torch.arange(self.grid_count, device=states.device)
        baseline = prior.argmax(-1)
        base_values = self.cached_values(baseline[None], networks, biases)
        base_prior = prior[grids, baseline].sum()

        def score(actions):
            differences = self.cached_values(actions, networks, biases) - base_values
            confidence = differences.mean(0) - self.uncertainty_weight * differences.std(0, unbiased=False)
            log_prior = prior[grids, actions].sum(-1) - base_prior
            return confidence + self.behavior_weight * log_prior

        starts = [baseline.clone()]
        if self.multistart:
            candidates = baseline.repeat(self.grid_count * self.action_count, 1)
            candidates[torch.arange(len(candidates)), grids.repeat_interleave(self.action_count)] = torch.arange(self.action_count).repeat(self.grid_count)
            starts.append(score(candidates).reshape(self.grid_count, self.action_count).argmax(-1))
        starts.extend(self.extra_starts(states))
        best_actions, best_value = baseline.clone(), states.new_tensor(0.)
        for start in starts:
            current = start.clone()
            for sweep in range(self.sweeps):
                order = range(self.grid_count) if sweep % 2 == 0 else reversed(range(self.grid_count))
                for grid in order:
                    candidates = current.repeat(self.action_count, 1)
                    candidates[:, grid] = torch.arange(self.action_count, device=states.device)
                    values = score(candidates)
                    index = values.argmax()
                    current = candidates[index].clone()
                    if values[index] > best_value:
                        best_value, best_actions = values[index], current.clone()
        return best_actions[None]
