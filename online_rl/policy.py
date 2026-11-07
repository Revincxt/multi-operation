"""时序状态编码、因果动作解码和独立的全城状态价值网络。"""

import torch
from torch import nn
from torch.nn import functional as F


class StateEncoder(nn.Module):
    def __init__(self, grids, steps, features, context, hidden, memory, heads, layers):
        super().__init__()
        self.grids, self.steps, self.features, self.context = grids, steps, features, context
        self.memory = nn.LSTM(features, memory, batch_first=True)
        self.identity = nn.Embedding(grids, 8)
        self.projection = nn.Linear(memory+context+8, hidden)
        block = nn.TransformerEncoderLayer(hidden, heads, hidden*2, dropout=0., batch_first=True)
        self.attention = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)

    def forward(self, states):
        boundary = self.grids*self.steps*self.features
        history = states[:, :boundary].reshape(-1, self.steps, self.features)
        _, (hidden, _) = self.memory(history.contiguous())
        hidden = hidden[-1].reshape(len(states), self.grids, -1)
        context = states[:, boundary:].reshape(len(states), self.grids, self.context)
        identity = self.identity.weight[None].expand(len(states), -1, -1)
        return self.attention(F.gelu(self.projection(torch.cat((hidden, context, identity), -1))))


class JointPolicy(nn.Module):
    """P(a|H)=∏i P(pi|H,a<i)P(ri|H,a<i,pi)，固定区域顺序。"""

    def __init__(self, grid_count, history_steps, history_features, context_size, price_count, radius_count,
                 hidden_size=64, memory_size=32, heads=4, layers=1, autoregressive=True):
        super().__init__()
        self.grids, self.prices, self.radii = grid_count, price_count, radius_count
        self.autoregressive = autoregressive
        self.state_size = grid_count*(history_steps*history_features+context_size)
        args = grid_count, history_steps, history_features, context_size, hidden_size, memory_size, heads, layers
        # 价值回归不直接改变策略的状态编码，便于解释PPO的策略变化。
        self.actor_encoder, self.critic_encoder = StateEncoder(*args), StateEncoder(*args)
        self.action_embedding = nn.Embedding(price_count*radius_count+1, hidden_size)
        self.region_embedding = nn.Embedding(grid_count, hidden_size)
        block = nn.TransformerDecoderLayer(hidden_size, heads, hidden_size*2, dropout=0., batch_first=True)
        self.decoder = nn.TransformerDecoder(block, layers)
        self.price_head = nn.Linear(hidden_size, price_count)
        self.price_embedding = nn.Embedding(price_count, 8)
        self.radius_head = nn.Linear(hidden_size+8, radius_count)
        self.value_head = nn.Sequential(nn.Linear(grid_count*hidden_size, 128), nn.Tanh(), nn.Linear(128, 1))
        self.register_buffer("causal_mask", torch.triu(torch.ones(grid_count, grid_count, dtype=torch.bool), diagonal=1))
        nn.init.orthogonal_(self.price_head.weight, gain=.01)
        nn.init.orthogonal_(self.radius_head.weight, gain=.01)
        nn.init.zeros_(self.price_head.bias); nn.init.zeros_(self.radius_head.bias)

    def conditional_log_probs(self, encoded, actions):
        # 输入右移一位：区域i永远看不到自己的实际动作和后续区域动作。
        previous = torch.full_like(actions, self.prices*self.radii)
        if self.autoregressive:
            previous[:, 1:] = actions[:, :-1]
        query = self.action_embedding(previous)+self.region_embedding.weight[None]
        hidden = self.decoder(query, encoded, tgt_mask=self.causal_mask)
        log_prices = F.log_softmax(self.price_head(hidden), -1)
        expanded = hidden[:, :, None].expand(-1, -1, self.prices, -1)
        price_codes = self.price_embedding.weight[None, None].expand(len(hidden), self.grids, -1, -1)
        log_radii = F.log_softmax(self.radius_head(torch.cat((expanded, price_codes), -1)), -1)
        return (log_prices[..., None]+log_radii).flatten(2)

    def value(self, states):
        return self.value_head(self.critic_encoder(states).flatten(1)).squeeze(-1)

    def evaluate_actions(self, states, actions):
        logs = self.conditional_log_probs(self.actor_encoder(states), actions)
        selected = logs.gather(-1, actions[..., None]).squeeze(-1)
        # 条件熵对已采样前缀求均值；不声称是新策略全联合熵的精确枚举。
        entropy = -(logs.exp()*logs).sum(-1).sum(-1)
        return selected.sum(-1), entropy, self.value(states)

    @torch.no_grad()
    def sample(self, states, generator, deterministic=False):
        encoded = self.actor_encoder(states)
        actions = torch.zeros((len(states), self.grids), dtype=torch.long, device=states.device)
        logs = []
        for grid in range(self.grids):
            part = self.conditional_log_probs(encoded, actions)[:, grid]
            chosen = part.argmax(-1) if deterministic else torch.multinomial(part.exp(), 1, generator=generator).squeeze(-1)
            actions[:, grid] = chosen
            logs.append(part.gather(-1, chosen[:, None]).squeeze(-1))
        return actions, torch.stack(logs).sum(0), self.value(states)
