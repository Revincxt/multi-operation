"""论文 Algorithm 2 的 WESM 多任务损失，支持任意任务数。

来源: Chen et al. (2025), Transportation Research Part E 193, 103822.
DOI: 10.1016/j.tre.2024.103822, Section 4.4 / Algorithm 2 / Table 2.
按文字说明使用时间滞后 gamma**j，而不是任务编号作为衰减指数。
"""

import torch
from torch import nn


class WESMLoss(nn.Module):
    """误差越大的任务权重越高；权重计算不参与梯度传播。"""

    def __init__(self, task_count, gamma=0.1, history_steps=10, update_interval=10):
        super().__init__()
        if (isinstance(task_count, bool) or not isinstance(task_count, int) or task_count <= 0
                or isinstance(history_steps, bool) or not isinstance(history_steps, int)
                or history_steps < 0 or not 0 <= gamma < 1
                or isinstance(update_interval, bool) or not isinstance(update_interval, int)
                or update_interval <= 0):
            raise ValueError
        self.task_count = task_count
        self.gamma = gamma
        self.history_steps = history_steps
        self.update_interval = update_interval
        self.register_buffer("weights", torch.full((task_count,), 1.0 / task_count))
        self.register_buffer("loss_history", torch.zeros(history_steps + 1, task_count))
        self.register_buffer("step_count", torch.zeros((), dtype=torch.int64))

    def forward(self, task_losses):
        """输入当前 batch 的逐任务 MSE，返回加权总损失。"""
        if (task_losses.shape != (self.task_count,)
                or not torch.isfinite(task_losses).all() or (task_losses < 0).any()):
            raise ValueError
        if self.training:
            with torch.no_grad():
                # 第 0 行是当前损失，其后是最多 history_steps 个过去 batch。
                self.loss_history[1:].copy_(self.loss_history[:-1].clone())
                self.loss_history[0].copy_(task_losses.detach())
                self.step_count.add_(1)
                step = int(self.step_count.item())
                if step % self.update_interval == 0:
                    count = min(step, self.history_steps + 1)
                    decay = self.gamma ** torch.arange(
                        count, device=task_losses.device, dtype=task_losses.dtype)
                    smoothed = (self.loss_history[:count] * decay[:, None]).sum(dim=0)
                    smoothed = smoothed / decay.sum()
                    # 六个任务的权重和为 1；全零误差时退回等权，避免 0/0。
                    maximum = smoothed.max()
                    if maximum > 0:
                        relative = smoothed / maximum
                        self.weights.copy_(relative / relative.sum())
                    else:
                        self.weights.fill_(1.0 / self.task_count)
        return torch.sum(self.weights.detach().clone() * task_losses)
