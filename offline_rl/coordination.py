"""城市状态广播、价格条件半径头，以及 OMIGA 的离散策略适配。"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from offline_rl.iql import Actor, mlp


def broadcast_indices(metadata):
    """广播其他区域的四项当前计数和上周期动作，保留网格身份及历史锁价。"""
    preprocessor = metadata["preprocessor"]
    features = preprocessor["history_features"]
    per_history = preprocessor["history_steps"] * len(features)
    history_size = metadata["grid_count"] * per_history
    context = metadata["state_context_columns"]
    counts = [context.index(f"start_{name}") for name in (
        "waiting_request_num", "idle_driver_num", "busy_driver_num", "repositioning_driver_num")]
    previous = [(preprocessor["history_steps"] - 1) * len(features) + features.index(name)
                for name in ("price_multiplier", "matching_radius_km")]
    indices = []
    for grid, own in enumerate(metadata["local_state_indices"]):
        extra = []
        for other in range(metadata["grid_count"]):
            if other != grid:
                extra.extend(history_size + other * len(context) + i for i in counts)
                extra.extend(other * per_history + i for i in previous)
        indices.append(own + extra)
    return indices


class ConditionalGridActor(nn.Module):
    """仍为两个动作头；第二个头接收本智能体实际选择的价格。"""

    def __init__(self, input_size, hidden_size, price_count, radius_count):
        super().__init__()
        self.encoder = nn.Sequential(mlp(input_size, hidden_size, hidden_size), nn.ReLU())
        self.price_head = nn.Linear(hidden_size, price_count)
        self.price_embedding = nn.Embedding(price_count, 16)
        self.radius_head = nn.Linear(hidden_size + 16, radius_count)

    def forward(self, observations, prices=None, allowed_prices=None):
        hidden = self.encoder(observations)
        logits = self.price_head(hidden)
        if allowed_prices is not None:
            mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            mask[allowed_prices] = False
            logits = logits.masked_fill(mask, -torch.inf)
        if prices is None:
            prices = logits.argmax(-1)
        radius = self.radius_head(torch.cat((hidden, self.price_embedding(prices)), dim=-1))
        return logits, radius


class ConditionalActor(Actor):
    def __init__(self, state_size, grid_count, price_count, radius_count, local_indices,
                 hidden_size=256, embedding_size=16):
        # 保留与原模型相同的独立智能体布局，避免复用并丢弃一套父类参数。
        nn.Module.__init__(self)
        self.grid_count, self.price_count, self.radius_count = grid_count, price_count, radius_count
        self.register_buffer("local_indices", torch.as_tensor(local_indices, dtype=torch.long))
        self.agents = nn.ModuleList([ConditionalGridActor(len(indices), hidden_size, price_count, radius_count)
                                    for indices in local_indices])

    def forward(self, states, prices=None, allowed_prices=None):
        outputs = [agent(states[:, self.local_indices[g]], None if prices is None else prices[:, g], allowed_prices)
                   for g, agent in enumerate(self.agents)]
        return torch.stack([x[0] for x in outputs], 1), torch.stack([x[1] for x in outputs], 1)

    def log_probs(self, states, actions):
        # 正确的联合似然：P(price|obs) × P(radius|obs, recorded_price)。
        prices, radii = actions // self.radius_count, actions % self.radius_count
        p, r = self(states, prices)
        return (F.log_softmax(p, -1).gather(-1, prices[..., None]).squeeze(-1)
                + F.log_softmax(r, -1).gather(-1, radii[..., None]).squeeze(-1))

    def log_prob(self, states, actions):
        return self.log_probs(states, actions).sum(-1)

    @torch.no_grad()
    def act(self, states, allowed_prices=None, allowed_radii=None):
        p, r = self(states, allowed_prices=allowed_prices)
        if allowed_radii is not None:
            mask = torch.ones(self.radius_count, dtype=torch.bool, device=states.device)
            mask[allowed_radii] = False
            r = r.masked_fill(mask, -torch.inf)
        return p.argmax(-1) * self.radius_count + r.argmax(-1)


def make_actor(config, architecture="parallel"):
    if architecture == "structured_joint_starts":
        from offline_rl.joint_starts import JointStartActor
        return JointStartActor(**config)
    if architecture == "structured_value":
        from offline_rl.structured_value import StructuredValueActor
        return StructuredValueActor(**config)
    if architecture in ("cooperative_distill", "cooperative_refined"):
        from offline_rl.cooperative_policy import CooperativeActor, RefinedCooperativeActor
        return (CooperativeActor if architecture == "cooperative_distill" else RefinedCooperativeActor)(**config)
    if architecture == "value_guided":
        from offline_rl.value_guided import ValueGuidedActor
        return ValueGuidedActor(**config)
    if architecture in ("global_conditional", "autoregressive"):
        from offline_rl.joint_policy import JointActor
        return JointActor(**config, autoregressive=architecture == "autoregressive")
    if architecture not in ("parallel", "conditional"):
        raise ValueError
    return (Actor if architecture == "parallel" else ConditionalActor)(**config)


class LocalValue(nn.Module):
    """局部价值网络共享底层参数并带区域 ID；策略仍是六个独立网络。"""

    def __init__(self, config, with_actions=False):
        super().__init__()
        self.with_actions = with_actions
        self.register_buffer("indices", torch.as_tensor(config["local_indices"], dtype=torch.long))
        self.register_buffer("identity", torch.eye(config["grid_count"]))
        self.actions = nn.Embedding(config["price_count"] * config["radius_count"], 16) if with_actions else None
        size = len(config["local_indices"][0]) + config["grid_count"] + (16 if with_actions else 0)
        self.network = mlp(size, config["hidden_size"], 1)

    def forward(self, states, actions=None):
        features = [states[:, self.indices], self.identity[None].expand(len(states), -1, -1)]
        if self.with_actions:
            features.append(self.actions(actions))
        return self.network(torch.cat(features, -1)).squeeze(-1)


class Mixer(nn.Module):
    def __init__(self, state_size, grid_count, hidden_size):
        super().__init__()
        self.network = mlp(state_size, hidden_size, grid_count + 1)

    def forward(self, states):
        values = self.network(states)
        return values[:, :-1].abs(), values[:, -1]


def omiga_value_loss(local_q, local_v, weights, alpha):
    """论文局部隐式价值目标，沿用官方实现的指数数值稳定处理。"""
    z = (weights * (local_q - local_v) / alpha).clamp(-10, 10)
    shift = z.max().clamp_min(-1).detach()
    loss = ((z - shift).exp() + (-shift).exp() * weights * local_v / alpha).mean()
    return loss, z.exp().detach()


class OMIGA:
    """全局加权 Q/V 分解与局部优势权重；离散双头适配，不声称原任务复现。"""

    def __init__(self, config, device="cpu", alpha=.3, discount=.99, reward_scale=100.,
                 learning_rate=3e-4, target_tau=.005):
        self.device = torch.device(device)
        self.config, self.alpha, self.discount = config, alpha, discount
        self.reward_scale, self.target_tau = reward_scale, target_tau
        self.actor = ConditionalActor(**config).to(device)
        self.q, self.value = LocalValue(config, True).to(device), LocalValue(config).to(device)
        self.mixer = Mixer(config["state_size"], config["grid_count"], config["hidden_size"]).to(device)
        self.target_q = deepcopy(self.q).requires_grad_(False)
        self.target_v = deepcopy(self.value).requires_grad_(False)
        self.target_mixer = deepcopy(self.mixer).requires_grad_(False)
        self.optimizers = {
            "actor": torch.optim.Adam(self.actor.parameters(), lr=learning_rate),
            "q": torch.optim.Adam(list(self.q.parameters()) + list(self.mixer.parameters()), lr=learning_rate),
            "value": torch.optim.Adam(self.value.parameters(), lr=learning_rate),
        }

    def update(self, batch):
        s, a, nxt = batch["states"], batch["actions"], batch["next_states"]
        with torch.no_grad():
            w_target, _ = self.target_mixer(s)
            local_q = self.target_q(s, a)
            w_next, b_next = self.target_mixer(nxt)
            future = (w_next * self.target_v(nxt)).sum(-1) + b_next
            target = batch["rewards"] / self.reward_scale + self.discount * batch["masks"] * future
        local_v = self.value(s)
        value_loss, weights = omiga_value_loss(local_q, local_v, w_target, self.alpha)
        actor_loss = -(weights * self.actor.log_probs(s, a)).mean()
        w, offset = self.mixer(s)
        total_q = (w * self.q(s, a)).sum(-1) + offset
        q_loss = (total_q - target).square().mean()
        for name, loss, parameters in (("actor", actor_loss, self.actor.parameters()),
                                        ("q", q_loss, list(self.q.parameters()) + list(self.mixer.parameters())),
                                        ("value", value_loss, self.value.parameters())):
            self.optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 10.)
            self.optimizers[name].step()
        with torch.no_grad():
            for online, frozen in ((self.q, self.target_q), (self.value, self.target_v),
                                    (self.mixer, self.target_mixer)):
                for p, target_p in zip(online.parameters(), frozen.parameters()):
                    target_p.lerp_(p, self.target_tau)
        return torch.stack([q_loss.detach(), value_loss.detach(), actor_loss.detach(),
                            weights.mean(), total_q.detach().mean()])

    def state_dict(self):
        return {"models": {name: getattr(self, name).state_dict() for name in (
                    "actor", "q", "value", "mixer", "target_q", "target_v", "target_mixer")},
                "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}}

    def load_state_dict(self, state):
        for name, values in state["models"].items():
            getattr(self, name).load_state_dict(values)
        for name, values in state["optimizers"].items():
            self.optimizers[name].load_state_dict(values)
