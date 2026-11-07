"""IQL 的六智能体适配：独立局部策略、并行双动作头、集中式双 Q。"""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_size, hidden_size, output_size):
    return nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                         nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                         nn.Linear(hidden_size, output_size))


def expectile_loss(difference, expectile):
    return (torch.where(difference > 0, expectile, 1 - expectile) * difference.square()).mean()


def bellman_target(rewards, masks, next_values, discount):
    return rewards + discount * masks * next_values


def advantage_weights(advantage, temperature, maximum=100.0):
    return (advantage * temperature).clamp(max=math.log(maximum)).exp().detach()


class GridActor(nn.Module):
    """一个区域一个独立网络，价格和半径分别输出离散 logits。"""

    def __init__(self, input_size, hidden_size, price_count, radius_count):
        super().__init__()
        self.encoder = nn.Sequential(mlp(input_size, hidden_size, hidden_size), nn.ReLU())
        self.price_head = nn.Linear(hidden_size, price_count)
        self.radius_head = nn.Linear(hidden_size, radius_count)

    def forward(self, observation):
        hidden = self.encoder(observation)
        return self.price_head(hidden), self.radius_head(hidden)


class Actor(nn.Module):
    """六个智能体各自两个并行输出头，参数不共享；支持分散执行。"""

    def __init__(self, state_size, grid_count, price_count, radius_count,
                 local_indices, hidden_size=256, embedding_size=16):
        super().__init__()
        self.grid_count, self.price_count, self.radius_count = grid_count, price_count, radius_count
        if len(local_indices) != grid_count or len({len(i) for i in local_indices}) != 1:
            raise ValueError
        self.register_buffer("local_indices", torch.as_tensor(local_indices, dtype=torch.long))
        self.agents = nn.ModuleList([GridActor(len(local_indices[g]), hidden_size, price_count, radius_count)
                                    for g in range(grid_count)])

    def forward(self, states):
        prices, radii = [], []
        for grid, agent in enumerate(self.agents):
            p, r = agent(states[:, self.local_indices[grid]])
            prices.append(p)
            radii.append(r)
        return torch.stack(prices, dim=1), torch.stack(radii, dim=1)

    def log_prob(self, states, actions):
        price_logits, radius_logits = self(states)
        prices, radii = actions // self.radius_count, actions % self.radius_count
        price_log = F.log_softmax(price_logits, dim=-1).gather(-1, prices[..., None]).squeeze(-1)
        radius_log = F.log_softmax(radius_logits, dim=-1).gather(-1, radii[..., None]).squeeze(-1)
        # 联合行为的对数概率，各网格共用一个全城优势权重。
        return (price_log + radius_log).sum(dim=-1)

    @torch.no_grad()
    def act(self, states, allowed_prices=None, allowed_radii=None):
        price_logits, radius_logits = self(states)
        if allowed_prices is not None:
            mask = torch.ones(self.price_count, dtype=torch.bool, device=states.device)
            mask[allowed_prices] = False
            price_logits = price_logits.masked_fill(mask, -torch.inf)
        prices = price_logits.argmax(dim=-1)
        if allowed_radii is not None:
            mask = torch.ones(self.radius_count, dtype=torch.bool, device=states.device)
            mask[allowed_radii] = False
            radius_logits = radius_logits.masked_fill(mask, -torch.inf)
        return prices * self.radius_count + radius_logits.argmax(dim=-1)


class Critic(nn.Module):
    """全城状态和完整六网格动作共同预测累计收益，保留动作间交互。"""

    def __init__(self, state_size, grid_count, action_count, hidden_size, embedding_size):
        super().__init__()
        self.action_count = action_count
        self.embedding = nn.Embedding(grid_count * action_count, embedding_size)
        self.register_buffer("offsets", torch.arange(grid_count) * action_count)
        self.network = mlp(state_size + grid_count * embedding_size, hidden_size, 1)

    def forward(self, states, actions):
        encoded = self.embedding(actions + self.offsets).flatten(1)
        return self.network(torch.cat((states, encoded), dim=1)).squeeze(-1)


class TwinQ(nn.Module):
    def __init__(self, state_size, grid_count, price_count, radius_count, local_indices=None, hidden_size=256, embedding_size=16):
        super().__init__()
        args = state_size, grid_count, price_count * radius_count, hidden_size, embedding_size
        self.q1, self.q2 = Critic(*args), Critic(*args)

    def forward(self, states, actions):
        return self.q1(states, actions), self.q2(states, actions)

    def minimum(self, states, actions):
        return torch.minimum(*self(states, actions))


class IQL:
    """价值 expectile、双 Q Bellman 回归与优势加权行为克隆。"""

    def __init__(self, model_config, device="cpu", learning_rate=3e-4, discount=0.99,
                 expectile=0.7, temperature=3.0, target_tau=0.005, reward_scale=100.0):
        if (not 0 <= discount <= 1 or not 0 < expectile < 1 or temperature <= 0
                or not 0 < target_tau <= 1 or reward_scale <= 0):
            raise ValueError
        self.device = torch.device(device)
        self.actor = Actor(**model_config).to(device)
        self.bc = deepcopy(self.actor)
        self.q = TwinQ(**model_config).to(device)
        self.target_q = deepcopy(self.q).requires_grad_(False)
        self.value = mlp(model_config["state_size"], model_config["hidden_size"], 1).to(device)
        self.optimizers = {name: torch.optim.Adam(model.parameters(), lr=learning_rate)
                           for name, model in (("actor", self.actor), ("bc", self.bc),
                                               ("q", self.q), ("value", self.value))}
        self.discount, self.expectile = discount, expectile
        self.temperature, self.target_tau, self.reward_scale = temperature, target_tau, reward_scale

    def update(self, batch):
        values = {name: torch.as_tensor(value, device=self.device) for name, value in batch.items()}
        states, actions = values["states"], values["actions"]
        with torch.no_grad():
            target_q = self.target_q.minimum(states, actions)
        v = self.value(states).squeeze(-1)
        value_loss = expectile_loss(target_q - v, self.expectile)
        self._step("value", value_loss)
        with torch.no_grad():
            advantage = target_q - self.value(states).squeeze(-1)
            weights = advantage_weights(advantage, self.temperature)
            target = bellman_target(values["rewards"] / self.reward_scale, values["masks"],
                                    self.value(values["next_states"]).squeeze(-1), self.discount)
        actor_loss = -(weights * self.actor.log_prob(states, actions)).mean()
        self._step("actor", actor_loss)
        bc_loss = -self.bc.log_prob(states, actions).mean()
        self._step("bc", bc_loss)
        q1, q2 = self.q(states, actions)
        q_loss = ((q1 - target).square() + (q2 - target).square()).mean()
        self._step("q", q_loss)
        with torch.no_grad():
            for online, frozen in zip(self.q.parameters(), self.target_q.parameters()):
                frozen.lerp_(online, self.target_tau)
        return torch.stack([q_loss.detach(), value_loss.detach(), actor_loss.detach(),
                            bc_loss.detach(), advantage.mean(), weights.mean(), target_q.mean()])

    def _step(self, name, loss):
        self.optimizers[name].zero_grad(set_to_none=True)
        loss.backward()
        self.optimizers[name].step()

    def state_dict(self):
        return {"models": {name: getattr(self, name).state_dict() for name in
                           ("actor", "bc", "q", "target_q", "value")},
                "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}}

    def load_state_dict(self, state):
        for name, values in state["models"].items():
            getattr(self, name).load_state_dict(values)
        for name, values in state["optimizers"].items():
            self.optimizers[name].load_state_dict(values)
