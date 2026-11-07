"""轻量时序/空间策略与集成Q引导的熵正则联合策略学习。"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from offline_rl.joint_policy import RegionalLinear
from offline_rl.value_guided import ValueGuidedActor


class CooperativeActor(nn.Module):
    """四期窗口LSTM、六网格注意力及价格条件半径头；不跨轨迹保存隐藏状态。"""

    def __init__(self, state_size, grid_count, history_steps, history_features, context_size,
                 price_values, radius_values, hidden_size=64, memory_size=48):
        super().__init__()
        self.grid_count, self.history_steps = grid_count, history_steps
        self.history_features, self.context_size = history_features, context_size
        self.price_count, self.radius_count = len(price_values), len(radius_values)
        if state_size != grid_count * (history_steps * history_features + context_size):
            raise ValueError
        self.memory = nn.LSTM(history_features, memory_size, batch_first=True)
        self.grid_embedding = nn.Embedding(grid_count, 8)
        self.projection = nn.Linear(memory_size + context_size + 8, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, 4, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(hidden_size), nn.LayerNorm(hidden_size)
        self.feedforward = nn.Sequential(nn.Linear(hidden_size, hidden_size * 2), nn.ReLU(),
                                         nn.Linear(hidden_size * 2, hidden_size))
        self.price_head = RegionalLinear(grid_count, hidden_size, self.price_count)
        self.price_embedding = nn.Parameter(torch.randn(grid_count, self.price_count, 8) * .1)
        self.radius_head = RegionalLinear(grid_count, hidden_size + 9, self.radius_count)
        values = torch.tensor(price_values, dtype=torch.float32)
        self.register_buffer("price_codes", 2 * (values-values.min()) / (values.max()-values.min()) - 1)

    def encode(self, states):
        batch = len(states)
        length = self.grid_count * self.history_steps * self.history_features
        history = states[:, :length].reshape(batch * self.grid_count, self.history_steps, self.history_features)
        _, (hidden, _) = self.memory(history.contiguous())
        hidden = hidden[-1].reshape(batch, self.grid_count, -1)
        context = states[:, length:].reshape(batch, self.grid_count, self.context_size)
        identities = self.grid_embedding.weight[None].expand(batch, -1, -1)
        tokens = F.relu(self.projection(torch.cat((hidden, context, identities), -1)))
        attention, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attention)
        return self.norm2(tokens + self.feedforward(tokens)).transpose(0, 1)

    def log_probs(self, states):
        hidden = self.encode(states)
        prices = F.log_softmax(self.price_head(hidden), -1)
        expanded = hidden[:, :, None].expand(-1, -1, self.price_count, -1)
        embedding = self.price_embedding[:, None].expand(-1, len(states), -1, -1)
        code = self.price_codes[None, None, :, None].expand(self.grid_count, len(states), -1, -1)
        inputs = torch.cat((expanded, embedding, code), -1)
        radii = self.radius_head(inputs.flatten(1, 2)).reshape(self.grid_count, len(states), self.price_count, self.radius_count)
        return (prices[..., None] + F.log_softmax(radii, -1)).flatten(2).transpose(0, 1)

    @torch.no_grad()
    def act(self, states, allowed_prices=None, allowed_radii=None):
        if allowed_prices is not None or allowed_radii is not None:
            raise ValueError
        return self.log_probs(states).argmax(-1)


class RefinedCooperativeActor(ValueGuidedActor):
    """学习策略提供第三个联合起点，继续按同一全城价值目标作有界改进。"""

    def __init__(self, proposal_config, **kwargs):
        super().__init__(**kwargs)
        self.proposal = CooperativeActor(**proposal_config)

    def extra_starts(self, states):
        return [self.proposal.act(states)[0]]


def behavior_log_probs(model, states):
    """每区域仅编码一次状态，再精确计算11×10个自身联合动作的概率。"""
    hidden = torch.cat((states, states.new_zeros(len(states), model.grid_count * 3)), -1)
    hidden = hidden[None].expand(model.grid_count, -1, -1)
    for layer in model.encoder:
        hidden = F.relu(layer(hidden))
    prices = F.log_softmax(model.price_head(hidden), -1)
    expanded = hidden[:, :, None].expand(-1, -1, model.price_count, -1)
    embedding = model.price_embedding[:, None].expand(-1, len(states), -1, -1)
    values = torch.cat((expanded, embedding), -1).flatten(1, 2)
    radii = model.radius_head(values).reshape(model.grid_count, len(states), model.price_count, model.radius_count)
    return (prices[..., None] + F.log_softmax(radii, -1)).flatten(2).transpose(0, 1)


def ensemble_values(model, states, actions):
    """batch×candidate×grid的联合Q，缓存第一层中的状态/动作可分部分。"""
    networks, biases = model._prepare(states)
    grids = torch.arange(model.grid_count, device=states.device)
    batch, candidates = actions.shape[:2]
    values = []
    for head, code, bias in zip(networks, model._codebooks, biases):
        hidden = F.relu(code[grids, actions].sum(-2) + bias[:, None])
        hidden = F.relu(F.linear(hidden, head.network[2].weight, head.network[2].bias))
        values.append(F.linear(hidden, head.network[4].weight, head.network[4].bias).squeeze(-1))
    return torch.stack(values).reshape(len(model.critics), 2, batch, candidates).min(1).values


class CooperativeDistillation:
    """冻结价值网络，学习软行为正则、熵正则的逐区域价值改进目标。"""

    def __init__(self, actor, evaluator, learning_rate=3e-4, entropy_weight=.002,
                 behavior_weight=.0001, uncertainty_weight=.25, target_tau=.02):
        self.actor = actor
        self.target_actor = deepcopy(actor).requires_grad_(False).eval()
        self.evaluator = evaluator.requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
        self.entropy_weight, self.behavior_weight = entropy_weight, behavior_weight
        self.uncertainty_weight, self.target_tau = uncertainty_weight, target_tau

    def update(self, states, actions=None, grid=None):
        log_probs = self.actor.log_probs(states)
        if grid is None:
            loss = -log_probs.gather(-1, actions[..., None]).squeeze(-1).sum(-1).mean()
            entropy = states.new_tensor(0.)
        else:
            with torch.no_grad():
                prior = behavior_log_probs(self.evaluator.behavior, states)
                baseline = prior.argmax(-1)
                joint = self.target_actor.act(states)
                candidates = joint[:, None].expand(-1, self.evaluator.action_count, -1).clone()
                candidates[:, :, grid] = torch.arange(self.evaluator.action_count, device=states.device)
                values = ensemble_values(self.evaluator, states, candidates)
                base = ensemble_values(self.evaluator, states, baseline[:, None])
                differences = values - base
                conservative = differences.mean(0) - self.uncertainty_weight * differences.std(0, unbiased=False)
                logits = (conservative + self.behavior_weight * prior[:, grid]) / (self.entropy_weight + self.behavior_weight)
                target = F.softmax(logits, -1)
                entropy = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
            loss = -(target * log_probs[:, grid]).sum(-1).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
        self.optimizer.step()
        with torch.no_grad():
            for target, online in zip(self.target_actor.parameters(), self.actor.parameters()):
                target.lerp_(online, self.target_tau)
        return torch.stack((loss.detach(), entropy.detach()))

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "target_actor": self.target_actor.state_dict(),
                "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.target_actor.load_state_dict(state["target_actor"])
        self.optimizer.load_state_dict(state["optimizer"])
