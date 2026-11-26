"""完整轨迹的收益／约束 GAE、联合 PPO 与批次边界的拉格朗日更新。"""

import numpy as np
import torch
from torch.nn import functional as F

from online_rl.ppo import advantages, clipped_policy_loss


def constraint_costs(demand, matches, wait_seconds, *, service_floor,
                     wait_limit_seconds_per_demand, demand_scale, wait_scale_seconds=100.):
    """可加成本；整轮求和后对应需求满足下限与单位外生需求等待上限。"""
    settings = np.asarray([service_floor, wait_limit_seconds_per_demand, demand_scale, wait_scale_seconds])
    if (not np.isfinite(settings).all() or not 0 <= service_floor <= 1
            or wait_limit_seconds_per_demand < 0 or demand_scale <= 0 or wait_scale_seconds <= 0):
        raise ValueError
    demand, matches, wait_seconds = np.broadcast_arrays(*[
        np.asarray(value, dtype=float) for value in (demand, matches, wait_seconds)])
    if any(not np.isfinite(value).all() or (value < 0).any() for value in (demand, matches, wait_seconds)):
        raise ValueError
    return np.stack(((service_floor * demand - matches) / demand_scale,
        (wait_seconds - wait_limit_seconds_per_demand * demand) / (demand_scale * wait_scale_seconds)), -1).astype("float32")


def _episode_arrays(episodes):
    if not episodes:
        raise ValueError
    result = {key: np.stack([row[key] for row in episodes]) for key in
              ("states", "costs", "terminals")}
    costs, terminal = result["costs"], result["terminals"]
    if (costs.ndim != 3 or costs.shape[-1] != 2 or terminal.shape != costs.shape[:2]
            or not np.isfinite(costs).all() or not np.isfinite(result["states"]).all()
            or not np.isin(terminal, [0., 1.]).all() or not terminal[:, -1].all()
            or terminal[:, :-1].any()):
        raise ValueError
    return result


def monte_carlo_costs(costs, terminals):
    """有限时域真实约束回报；只用于新 cost heads 的共同预热。"""
    return np.stack([advantages(costs[..., k], np.zeros_like(costs[..., k]), terminals,
                                gamma=1., gae_lambda=1.)[1] for k in range(2)], -1)


def warm_cost_heads(model, episodes, config, rng, device):
    """只拟合新增头；Actor、原收益价值及其编码器逐字节保持不变。"""
    arrays = _episode_arrays(episodes)
    targets = monte_carlo_costs(arrays["costs"], arrays["terminals"]).reshape(-1, 2)
    states = torch.as_tensor(arrays["states"].reshape(-1, model.state_size), device=device)
    target = torch.as_tensor(targets, dtype=torch.float32, device=device)
    batch_size = config["batch_size"]
    epochs = int(config.get("cost_warmup_epochs", 50))
    if epochs < 1 or batch_size < 1:
        raise ValueError
    previous_mode = model.training
    model.eval()
    with torch.no_grad():
        encoded = torch.cat([model.critic_encoder(states[start:start + batch_size]).flatten(1)
                             for start in range(0, len(states), batch_size)])
    scale = target.std(0, unbiased=False).clamp_min(config.get("cost_scale_floor", .01))
    optimizer = torch.optim.Adam(model.cost_head.parameters(), lr=config.get("cost_warmup_learning_rate", .001))
    with torch.no_grad():
        before = float(((model.cost_head(encoded) - target) / scale).square().mean().cpu())
    for _ in range(epochs):
        order = rng.permutation(len(encoded))
        for start in range(0, len(order), batch_size):
            index = torch.as_tensor(order[start:start + batch_size], device=device)
            loss = ((model.cost_head(encoded[index]) - target[index]) / scale).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.cost_head.parameters(), config["max_grad_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError
            optimizer.step()
    with torch.no_grad():
        after = float(((model.cost_head(encoded) - target) / scale).square().mean().cpu())
    model.train(previous_mode)
    return {"warm_cost_loss_before": before, "warm_cost_loss_after": after,
            "warm_cost_epochs": epochs, "warm_cost_samples": len(encoded)}


def update_multipliers(multipliers, episode_costs, learning_rate, constrained=True):
    """违反约束增加非负乘子；满足约束降低乘子，控制组保持为零。"""
    if (not isinstance(multipliers, np.ndarray) or multipliers.shape != (2,)
            or not np.issubdtype(multipliers.dtype, np.floating)
            or not np.isfinite(multipliers).all() or (multipliers < 0).any()
            or not np.isfinite(learning_rate) or learning_rate < 0):
        raise ValueError
    costs = np.asarray(episode_costs, dtype=float)
    if costs.ndim != 2 or costs.shape[1] != 2 or not len(costs) or not np.isfinite(costs).all():
        raise ValueError
    multipliers[:] = np.maximum(0., multipliers + learning_rate * costs.mean(0)) if constrained else 0.


def _diagnostics(model, data, batch_size):
    with torch.no_grad():
        parts = [model.evaluate_all(data["states"][start:start + batch_size], data["actions"][start:start + batch_size])
                 for start in range(0, len(data["states"]), batch_size)]
        logs, entropy, value, costs = [torch.cat([row[k] for row in parts]) for k in range(4)]
        difference = logs - data["log_probs"]
        kl = float((difference.exp() - 1. - difference).mean().cpu())
    return kl, entropy, value, costs


def update(model, optimizer, episodes, config, rng, device, multipliers, constrained=True):
    """只用当前版本完整轨迹；乘子在 PPO 的多个 epoch 内保持固定。"""
    arrays = _episode_arrays(episodes)
    if config["gamma"] != 1. or config["reward_scale"] <= 0:
        raise ValueError
    # 提前验证乘子，不在错误输入后更新网络。
    update_multipliers(multipliers.copy(), arrays["costs"].sum(1), 0., constrained)
    used = multipliers.copy() if constrained else np.zeros(2)
    rewards = np.stack([row["rewards"] for row in episodes]) / config["reward_scale"]
    values = np.stack([row["values"] for row in episodes])
    cost_values = np.stack([row["cost_values"] for row in episodes])
    if (rewards.shape != arrays["terminals"].shape or values.shape != rewards.shape
            or cost_values.shape != arrays["costs"].shape
            or any(not np.isfinite(value).all() for value in (rewards, values, cost_values))):
        raise ValueError
    advantage, returns = advantages(rewards, values, arrays["terminals"], config["gamma"], config["gae_lambda"])
    cost_gae = [advantages(arrays["costs"][..., k], cost_values[..., k], arrays["terminals"],
                           config["gamma"], config["gae_lambda"]) for k in range(2)]
    cost_advantages = np.stack([value[0] for value in cost_gae], -1)
    cost_returns = np.stack([value[1] for value in cost_gae], -1)
    # 保留收益与各约束的相对量纲；先合成拉格朗日优势，再共同标准化。
    mixed = advantage - np.sum(cost_advantages * used, axis=-1)
    mixed = (mixed - mixed.mean()) / max(float(mixed.std()), 1e-8)
    flat = {key: np.concatenate([row[key] for row in episodes]) for key in ("states", "actions", "log_probs")}
    flat.update(advantages=mixed.reshape(-1).astype("float32"), returns=returns.reshape(-1),
                cost_returns=cost_returns.reshape(-1, 2))
    data = {key: torch.as_tensor(value, device=device) for key, value in flat.items()}
    value_scale = max(float(returns.std()), config.get("value_scale_floor", .05))
    cost_scale = torch.as_tensor(np.maximum(cost_returns.reshape(-1, 2).std(0),
        config.get("cost_scale_floor", .01)), device=device)
    count, visits, rows, completed = len(data["states"]), 0, [], 0
    actor_parameters, critic_parameters = model.actor_parameters(), model.critic_parameters()
    model.eval()
    initial_kl, _, _, _ = _diagnostics(model, data, config["batch_size"])
    if not np.isfinite(initial_kl) or initial_kl > 1e-5:
        raise ValueError
    for epoch in range(config["epochs"]):
        model.train()
        order = rng.permutation(count)
        for start in range(0, count, config["batch_size"]):
            index = torch.as_tensor(order[start:start + config["batch_size"]], device=device)
            logp, entropy, value, predicted_cost = model.evaluate_all(data["states"][index], data["actions"][index])
            loss, kl, clipped = clipped_policy_loss(logp, data["log_probs"][index], data["advantages"][index], config["clip"])
            value_loss = F.mse_loss(value / value_scale, data["returns"][index] / value_scale)
            cost_losses = ((predicted_cost - data["cost_returns"][index]) / cost_scale).square().mean(0)
            objective = (loss + config["value_weight"] * value_loss
                         + config.get("cost_value_weight", .5) * cost_losses.mean()
                         - config["entropy_weight"] * entropy.mean())
            if not torch.isfinite(objective):
                raise FloatingPointError
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(actor_parameters, config["max_grad_norm"])
            critic_norm = torch.nn.utils.clip_grad_norm_(critic_parameters, config["max_grad_norm"])
            if not torch.isfinite(actor_norm) or not torch.isfinite(critic_norm):
                raise FloatingPointError
            optimizer.step()
            size = len(index)
            rows.append([size, float(loss.detach().cpu()), float(value_loss.detach().cpu()),
                         *cost_losses.detach().cpu().tolist(), float(entropy.detach().mean().cpu()),
                         float(clipped.detach().cpu()), float(actor_norm.detach().cpu()), float(critic_norm.detach().cpu())])
            visits += size
        completed = epoch + 1
        model.eval()
        final_kl, final_entropy, final_value, final_cost = _diagnostics(model, data, config["batch_size"])
        if final_kl > 1.5 * config["target_kl"]:
            break
    if not rows:
        raise ValueError
    means = np.average(np.asarray(rows)[:, 1:], axis=0, weights=np.asarray(rows)[:, 0])
    observed = arrays["costs"].sum(1).mean(0)
    update_multipliers(multipliers, arrays["costs"].sum(1), config.get("dual_learning_rate", 5.), constrained)
    return {**dict(zip(("policy_loss", "value_loss", "service_value_loss", "waiting_value_loss", "entropy",
                        "clip_fraction", "actor_gradient_norm", "critic_gradient_norm"), map(float, means))),
            "joint_kl": final_kl, "final_entropy": float(final_entropy.mean().cpu()),
            "effective_epochs": visits / count, "completed_epochs": completed,
            "optimizer_steps": len(rows), "sample_visits": visits,
            "value_target_scale": value_scale, "service_target_scale": float(cost_scale[0].cpu()),
            "waiting_target_scale": float(cost_scale[1].cpu()),
            "service_violation": float(observed[0]), "waiting_violation": float(observed[1]),
            "service_lambda_used": float(used[0]), "waiting_lambda_used": float(used[1]),
            "service_lambda_next": float(multipliers[0]), "waiting_lambda_next": float(multipliers[1])}
