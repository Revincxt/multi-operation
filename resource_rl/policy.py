"""独立策略/价值编码器：资源流图、整轮记忆、同信息窗口控制和因果动作解码。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class FlowAttention(nn.Module):
    def __init__(self, hidden, edges, heads):
        super().__init__()
        self.heads, self.width = heads, hidden//heads
        self.qkv = nn.Linear(hidden, hidden*3)
        self.edge_bias = nn.Sequential(nn.Linear(edges*2, hidden), nn.GELU(), nn.Linear(hidden, heads))
        self.projection = nn.Linear(hidden, hidden)
        self.first_norm, self.second_norm = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(), nn.Linear(hidden*2, hidden))

    def forward(self, nodes, edges):
        b, g, h = nodes.shape
        query, key, value = self.qkv(nodes).chunk(3, -1)
        def split(x):
            return x.reshape(b, g, self.heads, self.width).transpose(1, 2)
        score = split(query) @ split(key).transpose(-1, -2)/math.sqrt(self.width)
        # 同时提供两方向资源流，不把非邻接区域硬屏蔽。
        bias = self.edge_bias(torch.cat((edges, edges.transpose(1, 2)), -1)).permute(0, 3, 1, 2)
        message = ((score+bias).softmax(-1) @ split(value)).transpose(1, 2).reshape(b, g, h)
        nodes = self.first_norm(nodes+self.projection(message))
        return self.second_norm(nodes+self.feedforward(nodes))


class MicroStateEncoder(nn.Module):
    """价值侧读取当前个体集合；空队列保留一个可学习null token。"""
    def __init__(self, nodes, drivers, orders, hidden, heads, grids):
        super().__init__()
        self.query = nn.Linear(nodes, hidden)
        self.identity = nn.Embedding(grids, hidden)
        self.driver = nn.Sequential(nn.Linear(drivers, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.order = nn.Sequential(nn.Linear(orders, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.driver_attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.order_attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.null_order = nn.Parameter(torch.zeros(1, 1, hidden))
        self.output = nn.Sequential(nn.Linear(hidden*3, hidden), nn.GELU(), nn.LayerNorm(hidden))

    def forward(self, nodes, drivers, orders, order_mask):
        query = self.query(nodes)+self.identity.weight[None]
        driver = self.driver(drivers)
        order = torch.cat((self.null_order.expand(len(nodes), -1, -1), self.order(orders)), 1)
        mask = F.pad(order_mask, (1, 0), value=False)
        d = self.driver_attention(query, driver, driver, need_weights=False)[0]
        o = self.order_attention(query, order, order, key_padding_mask=mask, need_weights=False)[0]
        return self.output(torch.cat((query, d, o), -1))


class TemporalEncoder(nn.Module):
    def __init__(self, grids, nodes, edges, actions, hidden, heads, layers, variant, window, micro=False):
        super().__init__()
        self.grids, self.hidden, self.variant, self.window, self.micro = grids, hidden, variant, window, micro
        self.previous_action = nn.Embedding(actions+1, 16)
        self.identity = nn.Embedding(grids, 8)
        inputs = nodes+16+8+(hidden if micro else 0)
        if variant == "window":
            inputs += grids*edges*2
        self.project = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.memory = nn.LSTM(hidden, hidden, batch_first=True)
        if variant == "flow":
            self.graph = nn.ModuleList(FlowAttention(hidden, edges, heads) for _ in range(layers))
        else:
            self.current = nn.Linear(hidden*2+8, hidden)
            block = nn.TransformerEncoderLayer(hidden, heads, hidden*2, dropout=0., batch_first=True)
            self.spatial = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)

    def features(self, nodes, edges, previous_actions, micro=None):
        items = [nodes, self.previous_action(previous_actions), self.identity.weight[None].expand(len(nodes), -1, -1)]
        if micro is not None:
            items.append(micro)
        if self.variant == "window":
            items += [edges.flatten(-2), edges.transpose(-3, -2).flatten(-2)]
        x = self.project(torch.cat(items, -1))
        if self.variant == "flow":
            for layer in self.graph:
                x = layer(x, edges)
        return x

    def sequence(self, nodes, edges, previous_actions, micro=None):
        b, t, g, _ = nodes.shape
        x = self.features(nodes.flatten(0, 1), edges.flatten(0, 1), previous_actions.flatten(0, 1),
                          micro.flatten(0, 1) if micro is not None else None).reshape(b, t, g, self.hidden)
        if self.variant == "flow":
            value = self.memory(x.permute(0, 2, 1, 3).reshape(b*g, t, self.hidden))[0]
            return value.reshape(b, g, t, self.hidden).permute(0, 2, 1, 3)
        # 每个决策只用此前window个周期，当前信息经context分支进入；零填充只发生在episode开始。
        padded = F.pad(x, (0, 0, 0, 0, self.window, 0))
        history = torch.stack([padded[:, k:k+t] for k in range(self.window)], -2)
        value = self.memory(history.reshape(b*t*g, self.window, self.hidden))[0][:, -1]
        value = value.reshape(b*t, g, self.hidden)
        identity = self.identity.weight[None].expand(b*t, -1, -1)
        return self.spatial(self.current(torch.cat((value, x.flatten(0, 1), identity), -1))).reshape(b, t, g, self.hidden)

    def step(self, nodes, edges, previous_actions, state=None, micro=None):
        x = self.features(nodes, edges, previous_actions, micro)
        b, g, h = x.shape
        if self.variant == "flow":
            value, following = self.memory(x.reshape(b*g, 1, h), state)
            return value.reshape(b, g, h), following
        if state is None:
            state = x.new_zeros((b, self.window, g, h))
        value = self.memory(state.permute(0, 2, 1, 3).reshape(b*g, self.window, h))[0][:, -1].reshape(b, g, h)
        identity = self.identity.weight[None].expand(b, -1, -1)
        result = self.spatial(self.current(torch.cat((value, x, identity), -1)))
        return result, torch.cat((state[:, 1:], x[:, None]), 1)


class JointDecoder(nn.Module):
    def __init__(self, grids, prices, radii, hidden, heads):
        super().__init__()
        self.grids, self.prices, self.radii = grids, prices, radii
        self.action = nn.Embedding(prices*radii+1, hidden)
        self.identity = nn.Embedding(grids, hidden)
        block = nn.TransformerDecoderLayer(hidden, heads, hidden*2, dropout=0., batch_first=True)
        self.decoder = nn.TransformerDecoder(block, 1)
        self.price = nn.Linear(hidden, prices)
        self.price_code = nn.Embedding(prices, 8)
        self.radius = nn.Linear(hidden+8, radii)
        self.register_buffer("causal_mask", torch.triu(torch.ones(grids, grids, dtype=torch.bool), 1))
        for head in (self.price, self.radius):
            nn.init.orthogonal_(head.weight, gain=.01); nn.init.zeros_(head.bias)

    def forward(self, encoded, actions):
        previous = torch.full_like(actions, self.prices*self.radii)
        previous[:, 1:] = actions[:, :-1]
        x = self.decoder(self.action(previous)+self.identity.weight[None], encoded, tgt_mask=self.causal_mask)
        price = F.log_softmax(self.price(x), -1)
        code = self.price_code.weight[None, None].expand(len(x), self.grids, -1, -1)
        radius = F.log_softmax(self.radius(torch.cat((x[:, :, None].expand(-1, -1, self.prices, -1), code), -1)), -1)
        return (price[..., None]+radius).flatten(2)

    @torch.no_grad()
    def sample(self, encoded, generator, forced=None):
        actions = torch.zeros((len(encoded), self.grids), dtype=torch.long, device=encoded.device)
        selected = []
        for grid in range(self.grids):
            logs = self(encoded, actions)[:, grid]
            chosen = forced[:, grid] if forced is not None else torch.multinomial(logs.exp(), 1, generator=generator).squeeze(-1)
            actions[:, grid] = chosen
            selected.append(logs.gather(-1, chosen[:, None]).squeeze(-1))
        return actions, torch.stack(selected).sum(0)


class ResourcePolicy(nn.Module):
    def __init__(self, grid_count, node_size, edge_size, driver_size, order_size, price_count, radius_count,
                 variant="flow", hidden_size=64, heads=4, layers=1, window=4):
        super().__init__()
        self.grids, self.variant = grid_count, variant
        self.micro = MicroStateEncoder(node_size, driver_size, order_size, hidden_size, heads, grid_count)
        args = grid_count, node_size, edge_size, price_count*radius_count, hidden_size, heads, layers, variant, window
        self.actor = TemporalEncoder(*args)
        self.critic = TemporalEncoder(*args, micro=True)
        self.decoder = JointDecoder(grid_count, price_count, radius_count, hidden_size, heads)
        self.value_head = nn.Sequential(nn.Linear(grid_count*hidden_size, 128), nn.Tanh(), nn.Linear(128, 1))
        self.aux_action = nn.Embedding(price_count*radius_count, 16)
        self.aux_head = nn.Sequential(nn.Linear(grid_count*(hidden_size+16), 128), nn.GELU(),
                                      nn.Linear(128, grid_count*2+grid_count**2))

    def actor_parameters(self):
        return list(self.actor.parameters())+list(self.decoder.parameters())

    def critic_parameters(self):
        return list(self.micro.parameters())+list(self.critic.parameters())+list(self.value_head.parameters())+list(self.aux_action.parameters())+list(self.aux_head.parameters())

    def actor_sequence(self, data):
        x = self.actor.sequence(data["nodes"], data["edges"], data["previous_actions"])
        b, t, g, h = x.shape
        logs = self.decoder(x.reshape(b*t, g, h), data["actions"].reshape(b*t, g)).reshape(b, t, g, -1)
        selected = logs.gather(-1, data["actions"][..., None]).squeeze(-1).sum(-1)
        entropy = -(logs.exp()*logs).sum((-1, -2))
        return selected, entropy

    def critic_sequence(self, data):
        b, t, g, _ = data["nodes"].shape
        micro = self.micro(*(data[key].flatten(0, 1) for key in ("nodes", "drivers", "orders", "order_mask")))
        x = self.critic.sequence(data["nodes"], data["edges"], data["previous_actions"], micro.reshape(b, t, g, -1))
        value = self.value_head(x.flatten(2)).squeeze(-1)
        auxiliary = self.aux_head(torch.cat((x, self.aux_action(data["actions"])), -1).flatten(2))
        return value, auxiliary

    @torch.no_grad()
    def step(self, data, actor_state, critic_state, generator, forced=None):
        encoded, actor_state = self.actor.step(data["nodes"], data["edges"], data["previous_actions"], actor_state)
        actions, logp = self.decoder.sample(encoded, generator, forced)
        micro = self.micro(*(data[key] for key in ("nodes", "drivers", "orders", "order_mask")))
        value_encoded, critic_state = self.critic.step(data["nodes"], data["edges"], data["previous_actions"], critic_state, micro)
        value = self.value_head(value_encoded.flatten(1)).squeeze(-1)
        return actions, logp, value, actor_state, critic_state
