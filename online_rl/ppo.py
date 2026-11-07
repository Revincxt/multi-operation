"""完整城市级概率比的PPO与不跨轨迹的有限时域GAE。"""

import numpy as np
import torch
from torch.nn import functional as F


def advantages(rewards, values, terminals, gamma=1., gae_lambda=.95):
    """输入为episode×time；10:00是研究目标终点，不作价值bootstrap。"""
    rewards, values, terminals = map(np.asarray, (rewards, values, terminals))
    if (rewards.shape != values.shape or values.shape != terminals.shape or rewards.ndim != 2
            or not terminals[:, -1].all()):
        raise ValueError
    result = np.zeros_like(rewards, dtype="float32")
    carry = np.zeros(len(rewards), dtype="float32")
    following = np.zeros(len(rewards), dtype="float32")
    for step in reversed(range(rewards.shape[1])):
        continuation = 1.-terminals[:, step]
        delta = rewards[:, step]+gamma*continuation*following-values[:, step]
        carry = delta+gamma*gae_lambda*continuation*carry
        result[:, step] = carry
        following = values[:, step]
    return result, result+values


def clipped_policy_loss(new_log, old_log, advantage, clip):
    ratio = (new_log-old_log).exp()
    loss = -torch.minimum(ratio*advantage, ratio.clamp(1.-clip, 1.+clip)*advantage).mean()
    kl = (ratio-1.-(new_log-old_log)).mean()
    return loss, kl, ((ratio-1.).abs() > clip).float().mean()


def update(model, optimizer, episodes, config, rng, device):
    rewards = np.stack([row["rewards"] for row in episodes])/config["reward_scale"]
    values = np.stack([row["values"] for row in episodes])
    terminal = np.stack([row["terminals"] for row in episodes])
    advantage, returns = advantages(rewards, values, terminal, config["gamma"], config["gae_lambda"])
    advantage = (advantage-advantage.mean())/max(float(advantage.std()), 1e-8)
    arrays = {name: np.concatenate([row[name] for row in episodes]) for name in ("states", "actions", "log_probs")}
    arrays.update(advantages=advantage.reshape(-1), returns=returns.reshape(-1))
    data = {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}
    count, minibatches, visits, completed_epochs = len(data["states"]), 0, 0, 0
    return_scale = max(float(returns.std()), 1.)
    diagnostics = []
    model.train()
    for epoch in range(config["epochs"]):
        order = rng.permutation(count)
        interrupted = False
        for start in range(0, count, config["batch_size"]):
            indices = torch.as_tensor(order[start:start+config["batch_size"]], device=device)
            logp, entropy, value = model.evaluate_actions(data["states"][indices], data["actions"][indices])
            loss, kl, clip_fraction = clipped_policy_loss(logp, data["log_probs"][indices], data["advantages"][indices], config["clip"])
            # KL阈值针对全部六区域的联合概率比，不是单个动作头。
            if float(kl.detach().cpu()) > 1.5*config["target_kl"]:
                interrupted = True
                break
            value_loss = F.mse_loss(value/return_scale, data["returns"][indices]/return_scale)
            objective = loss+config["value_weight"]*value_loss-config["entropy_weight"]*entropy.mean()
            if not torch.isfinite(objective):
                raise FloatingPointError
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            size = len(indices)
            diagnostics.append((size, float(loss.detach().cpu()), float(value_loss.detach().cpu()),
                float(entropy.detach().mean().cpu()), float(kl.detach().cpu()),
                float(clip_fraction.detach().cpu()), float(norm.detach().cpu())))
            minibatches += 1; visits += size
        if interrupted:
            break
        completed_epochs = epoch+1
    model.eval()
    with torch.no_grad():
        parts = [model.evaluate_actions(data["states"][start:start+256], data["actions"][start:start+256])
                 for start in range(0, count, 256)]
        final_log, final_value = torch.cat([x[0] for x in parts]), torch.cat([x[2] for x in parts])
        ratio = (final_log-data["log_probs"]).exp()
        final_kl = float((ratio-1.-(final_log-data["log_probs"])).mean().cpu())
        variance = float(data["returns"].var(unbiased=False).cpu())
        explained = 1.-float((data["returns"]-final_value).var(unbiased=False).cpu())/max(variance, 1e-8)
    averaged = np.average(np.asarray(diagnostics)[:, 1:], axis=0, weights=np.asarray(diagnostics)[:, 0]) if diagnostics else np.zeros(6)
    return {**dict(zip(("policy_loss", "value_loss", "entropy", "update_kl", "clip_fraction", "gradient_norm"), map(float, averaged))),
        "joint_kl": final_kl, "explained_variance": explained, "optimizer_steps": minibatches,
        "sample_visits": visits, "effective_epochs": visits/count, "completed_epochs": completed_epochs,
        "value_target_scale": return_scale}
