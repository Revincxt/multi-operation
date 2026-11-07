"""整条序列PPO：当前参数重算隐状态，策略KL与价值学习分开控制。"""

import numpy as np
import torch
from torch.nn import functional as F

from resource_rl.rollout import collate


def advantages(rewards, values, terminals, gamma=1., gae_lambda=.95):
    if rewards.ndim != 2 or rewards.shape != values.shape or rewards.shape != terminals.shape or not terminals[:, -1].all():
        raise ValueError
    advantage = np.zeros_like(rewards, dtype="float32")
    carry = np.zeros(len(rewards), dtype="float32")
    following = np.zeros(len(rewards), dtype="float32")
    for time in reversed(range(rewards.shape[1])):
        continuation = 1.-terminals[:, time]
        delta = rewards[:, time]+gamma*continuation*following-values[:, time]
        carry = delta+gamma*gae_lambda*continuation*carry
        advantage[:, time] = carry
        following = values[:, time]
    return advantage, advantage+values


def select(data, indices):
    return {name: value[indices] for name, value in data.items()}


@torch.no_grad()
def actor_diagnostics(model, data, warm, batch):
    logp, entropy = [], []
    for start in range(0, len(data["nodes"]), batch):
        part = model.actor_sequence(select(data, slice(start, start+batch)))
        logp.append(part[0][:, warm:]); entropy.append(part[1][:, warm:])
    logp, entropy = torch.cat(logp), torch.cat(entropy)
    difference = logp-data["log_probs"][:, warm:]
    ratio = difference.exp()
    return float((ratio-1.-difference).mean().cpu()), float(entropy.mean().cpu()), logp


def update(model, actor_optimizer, critic_optimizer, episodes, config, rng, device):
    warm, batch = config["warmup_steps"], config["sequence_batch_size"]
    rewards = np.stack([row["rewards"][warm:] for row in episodes])/config["reward_scale"]
    values = np.stack([row["values"][warm:] for row in episodes])
    terminal = np.stack([row["terminals"][warm:] for row in episodes])
    advantage, targets = advantages(rewards, values, terminal, config["gamma"], config["gae_lambda"])
    advantage = (advantage-advantage.mean())/max(float(advantage.std()), 1e-8)
    data = collate(episodes, device)
    target = torch.as_tensor(targets, device=device)
    advantage = torch.as_tensor(advantage, device=device)
    value_scale = max(float(targets.std()), 1.)
    count, steps = rewards.shape
    model.eval()
    _, _, recomputed = actor_diagnostics(model, data, warm, batch)
    torch.testing.assert_close(recomputed, data["log_probs"][:, warm:], rtol=2e-5, atol=5e-5)
    actor_rows, critic_rows = [], []
    actor_epochs = 0
    model.train()
    for epoch in range(config["actor_epochs"]):
        order = rng.permutation(count)
        for start in range(0, count, batch):
            index = torch.as_tensor(order[start:start+batch], device=device)
            current = select(data, index)
            logp, entropy = model.actor_sequence(current)
            difference = logp[:, warm:]-current["log_probs"][:, warm:]
            ratio = difference.exp()
            weighted = ratio*advantage[index]
            clipped = ratio.clamp(1.-config["clip"], 1.+config["clip"])*advantage[index]
            policy_loss = -torch.minimum(weighted, clipped).mean()
            objective = policy_loss-config["entropy_weight"]*entropy[:, warm:].mean()
            if not torch.isfinite(objective):
                raise FloatingPointError
            actor_optimizer.zero_grad(set_to_none=True)
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.actor_parameters(), config["max_grad_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError
            actor_optimizer.step()
            actor_rows.append([len(index)*steps, float(policy_loss.detach().cpu()),
                float(((ratio-1.).abs()>config["clip"]).float().mean().cpu()), float(norm.detach().cpu())])
        actor_epochs = epoch+1
        # 全批完整序列重算联合KL；监测不依赖一个随机小批，亦不停止价值训练。
        model.eval()
        final_kl, _, _ = actor_diagnostics(model, data, warm, batch)
        model.train()
        if final_kl > 1.5*config["target_kl"]:
            break
    for _ in range(config["critic_epochs"]):
        order = rng.permutation(count)
        for start in range(0, count, batch):
            index = torch.as_tensor(order[start:start+batch], device=device)
            current = select(data, index)
            value, prediction = model.critic_sequence(current)
            value_loss = F.mse_loss(value[:, warm:]/value_scale, target[index]/value_scale)
            boundary = model.grids*2
            predicted, actual = prediction[:, warm:], current["aux_targets"][:, warm:]
            auxiliary = .5*(F.mse_loss(predicted[..., :boundary], actual[..., :boundary])
                           +F.mse_loss(predicted[..., boundary:], actual[..., boundary:]))
            objective = config["value_weight"]*value_loss+config["auxiliary_weight"]*auxiliary
            if not torch.isfinite(objective):
                raise FloatingPointError
            critic_optimizer.zero_grad(set_to_none=True)
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.critic_parameters(), config["max_grad_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError
            critic_optimizer.step()
            critic_rows.append([len(index)*steps, float(value_loss.detach().cpu()),
                                float(auxiliary.detach().cpu()), float(norm.detach().cpu())])
    model.eval()
    final_kl, entropy, _ = actor_diagnostics(model, data, warm, batch)
    with torch.no_grad():
        predicted = torch.cat([model.critic_sequence(select(data, slice(k, k+batch)))[0][:, warm:]
                               for k in range(0, count, batch)])
        variance = max(float(target.var(unbiased=False).cpu()), 1e-8)
        explained = 1.-float((target-predicted).var(unbiased=False).cpu())/variance
    a, c = np.asarray(actor_rows), np.asarray(critic_rows)
    actor_means = np.average(a[:, 1:], axis=0, weights=a[:, 0])
    critic_means = np.average(c[:, 1:], axis=0, weights=c[:, 0])
    # 以下MC统计使用采集时的V，未使用本批拟合后的V，且不把时间趋势当作状态辨识能力。
    mc = np.flip(np.cumsum(np.flip(rewards, axis=1), axis=1), axis=1)
    conditioned = 1.-np.var(mc-values, axis=0)/np.maximum(np.var(mc, axis=0), 1e-8)
    return {"policy_loss": float(actor_means[0]), "clip_fraction": float(actor_means[1]),
        "actor_gradient_norm": float(actor_means[2]), "value_loss": float(critic_means[0]),
        "auxiliary_loss": float(critic_means[1]), "critic_gradient_norm": float(critic_means[2]),
        "joint_kl": final_kl, "entropy": entropy, "actor_epochs": actor_epochs,
        "critic_epochs": config["critic_epochs"], "actor_optimizer_steps": len(a), "critic_optimizer_steps": len(c),
        "actor_sample_visits": int(a[:, 0].sum()), "critic_sample_visits": int(c[:, 0].sum()),
        "actor_effective_epochs": float(a[:, 0].sum()/(count*steps)),
        "critic_effective_epochs": float(c[:, 0].sum()/(count*steps)),
        "value_target_scale": value_scale, "lambda_explained_variance": explained,
        "rollout_MC_MAE_usd": float(np.abs(mc-values).mean()*config["reward_scale"]),
        "rollout_initial_value_bias_usd": float((values[:, 0]-mc[:, 0]).mean()*config["reward_scale"]),
        "rollout_time_conditioned_MC_explained_variance": float(conditioned.mean())}
