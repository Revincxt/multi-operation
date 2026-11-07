"""有限时域IQL：时空状态编码、数值动作基与非可加联合价值。"""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from offline_rl.iql import expectile_loss, mlp
from offline_rl.joint_policy import JointActor
from offline_rl.value_guided import ValueGuidedActor


class TemporalEncoder(nn.Module):
    """四期窗口LSTM和六网格注意力；状态布局与现有660维replay一致。"""

    def __init__(self, state_size, grid_count, history_steps, history_features,
                 context_size, hidden_size=64, memory_size=32):
        super().__init__()
        if state_size != grid_count * (history_steps * history_features + context_size):
            raise ValueError
        self.grid_count, self.history_steps = grid_count, history_steps
        self.history_features, self.context_size = history_features, context_size
        self.memory = nn.LSTM(history_features, memory_size, batch_first=True)
        self.identity = nn.Embedding(grid_count, 8)
        self.project = nn.Linear(memory_size + context_size + 8, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, 4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU())

    def forward(self, states):
        batch = len(states)
        boundary = self.grid_count * self.history_steps * self.history_features
        sequence = states[:, :boundary].reshape(batch*self.grid_count, self.history_steps, self.history_features)
        _, (hidden, _) = self.memory(sequence.contiguous())
        hidden = hidden[-1].reshape(batch, self.grid_count, -1)
        context = states[:, boundary:].reshape(batch, self.grid_count, self.context_size)
        identity = self.identity.weight[None].expand(batch, -1, -1)
        tokens = F.silu(self.project(torch.cat((hidden, context, identity), -1)))
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return self.output(self.norm(tokens + attended))


class JointValueHead(nn.Module):
    """状态基值、自身动作项、成对交互和高阶联合残差；不限制单调性。"""

    def __init__(self, grids, hidden, basis, rank):
        super().__init__()
        self.grids, self.rank, self.basis_size = grids, rank, basis
        self.baseline = mlp(grids*hidden, 128, 1)
        self.local = nn.Linear(hidden, basis)
        self.factors = nn.Linear(hidden, basis*rank)
        self.gates = nn.Linear(grids*hidden, grids*(grids-1)//2)
        self.residual = mlp(grids*rank, 64, 1)
        pairs = torch.triu_indices(grids, grids, offset=1)
        self.register_buffer("pair_i", pairs[0])
        self.register_buffer("pair_j", pairs[1])

    def prepare(self, tokens, basis):
        batch = len(tokens)
        flat = tokens.flatten(1)
        local = torch.einsum("bgk,ak->bga", self.local(tokens), basis)
        coefficients = self.factors(tokens).reshape(batch, self.grids, self.basis_size, self.rank)
        factors = torch.einsum("bgkr,ak->bgar", coefficients, basis)
        return self.baseline(flat).squeeze(-1), local, factors, self.gates(flat).tanh()

    def cached(self, prepared, actions):
        # 支持batch×candidate×grid；状态相关的所有动作编码仅计算一次。
        baseline, local, factors, gates = prepared
        indices = torch.arange(len(actions), device=actions.device)[:, None, None]
        grids = torch.arange(self.grids, device=actions.device)[None, None]
        own = local[indices, grids, actions].sum(-1)
        chosen = factors[indices, grids, actions]
        paired = (chosen[:, :, self.pair_i] * chosen[:, :, self.pair_j]).sum(-1) / math.sqrt(self.rank)
        interactions = (paired * gates[:, None]).sum(-1)
        nonlinear = self.residual(chosen.flatten(2)).squeeze(-1)
        return baseline[:, None] + own + interactions + nonlinear

    def selected(self, tokens, basis, actions):
        # TD训练只需要已记录动作，不展开batch×grid×110个中间向量。
        chosen_basis = basis[actions]
        own = (self.local(tokens)*chosen_basis).sum((-1, -2))
        coefficients = self.factors(tokens).reshape(len(tokens), self.grids, self.basis_size, self.rank)
        chosen = (coefficients*chosen_basis[..., None]).sum(-2)
        flat = tokens.flatten(1)
        paired = (chosen[:, self.pair_i]*chosen[:, self.pair_j]).sum(-1)/math.sqrt(self.rank)
        interactions = (paired*self.gates(flat).tanh()).sum(-1)
        return self.baseline(flat).squeeze(-1) + own + interactions + self.residual(chosen.flatten(1)).squeeze(-1)


class StructuredTwinQ(nn.Module):
    """两个独立联合价值头共用时空编码；不同集成成员完全独立。"""

    def __init__(self, state_size, grid_count, history_steps, history_features, context_size,
                 price_values, radius_values, hidden_size=64, memory_size=32, basis_size=16, rank=8):
        super().__init__()
        self.grid_count = grid_count
        self.encoder = TemporalEncoder(state_size, grid_count, history_steps, history_features,
                                       context_size, hidden_size, memory_size)
        p = torch.tensor(price_values, dtype=torch.float32)
        r = torch.tensor(radius_values, dtype=torch.float32)
        pi = torch.arange(len(p)).repeat_interleave(len(r))
        ri = torch.arange(len(r)).repeat(len(p))
        pc = 2*(p[pi]-p.min())/(p.max()-p.min()).clamp_min(1e-6)-1
        rc = 2*(r[ri]-r.min())/(r.max()-r.min()).clamp_min(1e-6)-1
        self.register_buffer("price_indices", pi)
        self.register_buffer("radius_indices", ri)
        self.register_buffer("numeric_actions", torch.stack((pc, rc, pc*rc, pc.square(), rc.square()), -1))
        self.price_embedding, self.radius_embedding = nn.Embedding(len(p), 8), nn.Embedding(len(r), 8)
        self.action_basis = nn.Sequential(nn.Linear(21, 32), nn.SiLU(), nn.Linear(32, basis_size))
        self.heads = nn.ModuleList([JointValueHead(grid_count, hidden_size, basis_size, rank) for _ in range(2)])

    def basis(self):
        basis = self.action_basis(torch.cat((self.numeric_actions,
            self.price_embedding(self.price_indices), self.radius_embedding(self.radius_indices)), -1))
        return basis - basis.mean(0, keepdim=True)

    def prepare(self, states):
        tokens, basis = self.encoder(states), self.basis()
        return [head.prepare(tokens, basis) for head in self.heads]

    def cached(self, prepared, actions):
        return torch.stack([head.cached(values, actions) for head, values in zip(self.heads, prepared)])

    def forward(self, states, actions):
        tokens, basis = self.encoder(states), self.basis()
        return torch.stack([head.selected(tokens, basis, actions) for head in self.heads])

    def minimum(self, states, actions):
        return self(states, actions).min(0).values


class TemporalValue(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TemporalEncoder(**{key: config[key] for key in (
            "state_size", "grid_count", "history_steps", "history_features", "context_size", "hidden_size", "memory_size")})
        self.output = mlp(config["grid_count"]*config["hidden_size"], 128, 1)

    def forward(self, states):
        return self.output(self.encoder(states).flatten(1)).squeeze(-1)


class StructuredIQL:
    """重训Q/V；有限时域使用gamma=1，策略提取由独立的联合搜索完成。"""

    def __init__(self, config, device="cpu", discount=1., expectile=.7, learning_rate=3e-4,
                 target_tau=.005, reward_scale=100.):
        self.q = StructuredTwinQ(**config).to(device)
        self.target_q = deepcopy(self.q).requires_grad_(False)
        self.value = TemporalValue(config).to(device)
        self.optimizers = {name: torch.optim.Adam(model.parameters(), lr=learning_rate)
                           for name, model in (("q", self.q), ("value", self.value))}
        self.discount, self.expectile = discount, expectile
        self.target_tau, self.reward_scale = target_tau, reward_scale

    def update(self, batch):
        states, actions = batch["states"], batch["actions"]
        with torch.no_grad():
            target_q = self.target_q.minimum(states, actions)
        value_loss = expectile_loss(target_q-self.value(states), self.expectile)
        self._step("value", value_loss)
        with torch.no_grad():
            target = batch["rewards"]/self.reward_scale + self.discount*batch["masks"]*self.value(batch["next_states"])
        predictions = self.q(states, actions)
        errors = predictions-target[None]
        q_loss = errors.square().mean(1).sum()
        self._step("q", q_loss)
        with torch.no_grad():
            for old, new in zip(self.target_q.parameters(), self.q.parameters()):
                old.lerp_(new, self.target_tau)
        return torch.stack((q_loss.detach(), value_loss.detach(), errors.detach().abs().mean(), target_q.mean(), target.mean()))

    def _step(self, name, loss):
        self.optimizers[name].zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(getattr(self, name).parameters(), 10.)
        self.optimizers[name].step()

    def state_dict(self):
        return {"models": {name: getattr(self, name).state_dict() for name in ("q", "target_q", "value")},
                "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}}

    def load_state_dict(self, state):
        for name, values in state["models"].items():
            getattr(self, name).load_state_dict(values)
        for name, values in state["optimizers"].items():
            self.optimizers[name].load_state_dict(values)


class StructuredValueActor(ValueGuidedActor):
    """沿用同一全动作搜索及软正则，仅替换价值模型。"""

    def __init__(self, behavior_config, critic_config, ensemble_size=3, uncertainty_weight=.25,
                 behavior_weight=.0001, sweeps=2, multistart=True):
        nn.Module.__init__(self)
        self.behavior = JointActor(**behavior_config)
        self.grid_count = behavior_config["grid_count"]
        self.action_count = behavior_config["price_count"]*behavior_config["radius_count"]
        self.critics = nn.ModuleList([StructuredTwinQ(**critic_config) for _ in range(ensemble_size)])
        self.uncertainty_weight, self.behavior_weight = uncertainty_weight, behavior_weight
        self.sweeps, self.multistart = sweeps, multistart

    def _prepare(self, states):
        return [critic.prepare(states) for critic in self.critics], None

    def cached_values(self, actions, prepared, unused):
        return torch.stack([critic.cached(values, actions[None]).min(0).values[0]
                            for critic, values in zip(self.critics, prepared)])
