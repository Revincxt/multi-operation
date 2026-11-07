"""相同全城观测下的独立/自回归双头策略，用于联合策略投影消融。"""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from offline_rl.iql import TwinQ, advantage_weights, bellman_target, expectile_loss, mlp


class RegionalLinear(nn.Module):
    """把独立区域参数放在首维，一次batched GEMM完成同一层计算。"""

    def __init__(self, regions, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(regions, output_size, input_size))
        self.bias = nn.Parameter(torch.empty(regions, output_size))
        for region in range(regions):
            nn.init.kaiming_uniform_(self.weight[region], a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1 / math.sqrt(input_size), 1 / math.sqrt(input_size))

    def forward(self, values):
        return torch.bmm(values, self.weight.transpose(1, 2)) + self.bias[:, None, :]

    def region(self, values, grid):
        return F.linear(values, self.weight[grid], self.bias[grid])


class JointActor(nn.Module):
    """六个独立参数的区域网络；自回归版额外读取已选前缀，不改变动作支持集。"""

    def __init__(self, state_size, grid_count, price_count, radius_count,
                 price_values, radius_values, hidden_size=256, embedding_size=16,
                 autoregressive=False, order=None):
        super().__init__()
        if len(price_values) != price_count or len(radius_values) != radius_count:
            raise ValueError
        self.state_size, self.grid_count = state_size, grid_count
        self.price_count, self.radius_count = price_count, radius_count
        self.autoregressive = autoregressive
        self.order = tuple(range(grid_count)) if order is None else tuple(order)
        if sorted(self.order) != list(range(grid_count)):
            raise ValueError
        for name, values in (("price_codes", price_values), ("radius_codes", radius_values)):
            values = torch.tensor(values, dtype=torch.float32)
            scale = (values.max() - values.min()).clamp_min(1.)
            self.register_buffer(name, 2 * (values - values.min()) / scale - 1)
        preceding = torch.zeros(grid_count, grid_count, 1)
        for position, grid in enumerate(self.order):
            preceding[grid, list(self.order[:position])] = 1
        self.register_buffer("preceding", preceding)
        # 两版同一网络及参数量；独立版将前缀输入固定为零。
        size = state_size + grid_count * 3
        self.encoder = nn.ModuleList([RegionalLinear(grid_count, source, hidden_size)
                                      for source in (size, hidden_size, hidden_size)])
        self.price_head = RegionalLinear(grid_count, hidden_size, price_count)
        self.price_embedding = nn.Parameter(torch.randn(grid_count, price_count, 16))
        self.radius_head = RegionalLinear(grid_count, hidden_size + 16, radius_count)

    def _features(self, states, actions, grid):
        if not self.autoregressive:
            prefix = states.new_zeros(len(states), self.grid_count * 3)
        else:
            prices = self.price_codes[actions // self.radius_count]
            radii = self.radius_codes[actions % self.radius_count]
            prefix = torch.stack((prices, radii, torch.ones_like(prices)), -1)
            prefix = (prefix * self.preceding[grid]).flatten(1)
        return torch.cat((states, prefix), -1)

    def logits(self, states, actions):
        if self.autoregressive:
            prices = self.price_codes[actions // self.radius_count]
            radii = self.radius_codes[actions % self.radius_count]
            prefix = torch.stack((prices, radii, torch.ones_like(prices)), -1)
            prefix = (prefix[None] * self.preceding[:, None]).flatten(2)
        else:
            prefix = states.new_zeros(self.grid_count, len(states), self.grid_count * 3)
        hidden = torch.cat((states[None].expand(self.grid_count, -1, -1), prefix), -1)
        for layer in self.encoder:
            hidden = F.relu(layer(hidden))
        price_logits = self.price_head(hidden)
        grids = torch.arange(self.grid_count, device=states.device)[:, None]
        embedding = self.price_embedding[grids, (actions // self.radius_count).T]
        radius_logits = self.radius_head(torch.cat((hidden, embedding), -1))
        return price_logits.transpose(0, 1), radius_logits.transpose(0, 1)

    def region_logits(self, states, actions, grid, prices=None):
        hidden = self._features(states, actions, grid)
        for layer in self.encoder:
            hidden = F.relu(layer.region(hidden, grid))
        price_logits = self.price_head.region(hidden, grid)
        prices = price_logits.argmax(-1) if prices is None else prices
        embedding = self.price_embedding[grid, prices]
        radius_logits = self.radius_head.region(torch.cat((hidden, embedding), -1), grid)
        return price_logits, radius_logits

    def log_prob(self, states, actions):
        p, r = self.logits(states, actions)
        price_log = F.log_softmax(p, -1).gather(-1, (actions // self.radius_count)[..., None])
        radius_log = F.log_softmax(r, -1).gather(-1, (actions % self.radius_count)[..., None])
        return (price_log + radius_log).squeeze(-1).sum(-1)

    @torch.no_grad()
    def act(self, states, allowed_prices=None, allowed_radii=None):
        # 正式消融始终保留全部合法动作，不接受推理时人为截断。
        if allowed_prices is not None or allowed_radii is not None:
            raise ValueError
        selected = torch.zeros(len(states), self.grid_count, dtype=torch.long, device=states.device)
        for grid in self.order:
            prices, radii = self.region_logits(states, selected, grid)
            selected[:, grid] = prices.argmax(-1) * self.radius_count + radii.argmax(-1)
        return selected


class PairedIQL:
    """同一个新训练的联合Q/V，向两种策略提供完全相同的IQL优势权重。"""

    names = ("independent", "autoregressive", "bc_independent", "bc_autoregressive")
    metrics = ("q_loss", "value_loss", *[f"{name}_loss" for name in names],
               "advantage_mean", "weight_mean", "weight_ess_fraction")

    def __init__(self, config, device="cpu", discount=.99, expectile=.7, temperature=3.,
                 learning_rate=3e-4, reward_scale=100., target_tau=.005, critic_hidden_size=None):
        self.config, self.device = config, torch.device(device)
        self.discount, self.expectile, self.temperature = discount, expectile, temperature
        self.reward_scale, self.target_tau = reward_scale, target_tau
        independent = JointActor(**config).to(device)
        autoregressive = deepcopy(independent)
        autoregressive.autoregressive = True
        self.actors = nn.ModuleDict({"independent": independent, "autoregressive": autoregressive,
                                    "bc_independent": deepcopy(independent),
                                    "bc_autoregressive": deepcopy(autoregressive)})
        critic_config = {key: config[key] for key in (
            "state_size", "grid_count", "price_count", "radius_count", "hidden_size", "embedding_size")}
        critic_config["hidden_size"] = critic_hidden_size or config["hidden_size"]
        self.q = TwinQ(**critic_config).to(device)
        self.target_q = deepcopy(self.q).requires_grad_(False)
        self.value = mlp(config["state_size"], critic_config["hidden_size"], 1).to(device)
        self.optimizers = {name: torch.optim.Adam(model.parameters(), lr=learning_rate)
            for name, model in {**dict(self.actors.items()), "q": self.q, "value": self.value}.items()}

    def _step(self, name, loss):
        self.optimizers[name].zero_grad(set_to_none=True)
        loss.backward()
        self.optimizers[name].step()

    def update(self, batch):
        s, a = batch["states"], batch["actions"]
        with torch.no_grad():
            q_target = self.target_q.minimum(s, a)
        value_loss = expectile_loss(q_target - self.value(s).squeeze(-1), self.expectile)
        self._step("value", value_loss)
        with torch.no_grad():
            advantage = q_target - self.value(s).squeeze(-1)
            weights = advantage_weights(advantage, self.temperature)
            target = bellman_target(batch["rewards"] / self.reward_scale, batch["masks"],
                self.value(batch["next_states"]).squeeze(-1), self.discount)
        actor_losses = []
        for name, actor in self.actors.items():
            logp = actor.log_prob(s, a)
            loss = -logp.mean() if name.startswith("bc_") else -(weights * logp).mean()
            self._step(name, loss)
            actor_losses.append(loss.detach())
        q1, q2 = self.q(s, a)
        q_loss = ((q1 - target).square() + (q2 - target).square()).mean()
        self._step("q", q_loss)
        with torch.no_grad():
            for online, frozen in zip(self.q.parameters(), self.target_q.parameters()):
                frozen.lerp_(online, self.target_tau)
        ess = weights.sum().square() / (len(weights) * weights.square().sum().clamp_min(1e-12))
        return torch.stack([q_loss.detach(), value_loss.detach(), *actor_losses,
                            advantage.mean(), weights.mean(), ess])

    def state_dict(self):
        return {"actors": self.actors.state_dict(), "q": self.q.state_dict(),
                "target_q": self.target_q.state_dict(), "value": self.value.state_dict(),
                "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}}

    def load_state_dict(self, state):
        for name in ("actors", "q", "target_q", "value"):
            getattr(self, name).load_state_dict(state[name])
        for name, value in state["optimizers"].items():
            self.optimizers[name].load_state_dict(value)
