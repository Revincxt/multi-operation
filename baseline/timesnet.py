"""TimesNet 基线：周期识别、二维卷积和网格级六指标预测。

TimesBlock / Inception / embedding 参考 THUML 的 MIT 开源实现：
https://github.com/thuml/Time-Series-Library
许可见 THIRD_PARTY_NOTICES.md。此处适配联合动作条件预测，不是论文完整复现。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from baseline.sequence import CONTEXT_COLUMNS, TARGET_COLUMNS, SequencePredictor


def fft_for_period(x, top_k):
    """逐样本选择非零频率，避免预测结果依赖同 batch 的其他网格。"""
    if (x.ndim != 3 or min(x.shape) <= 0 or x.shape[1] < 2
            or not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
        raise ValueError
    # MPS 的复数 FFT 支持受限，仅将这一计算转到 CPU；保留梯度链。
    fft_input = x.cpu() if x.device.type == "mps" else x
    amplitudes = torch.fft.rfft(fft_input, dim=1).abs().mean(dim=-1).to(x.device)
    # 直接排除直流项，常量/全零序列也不会选中频率 0 而产生除零。
    weights, indices = amplitudes[:, 1:].topk(min(top_k, amplitudes.shape[1] - 1), dim=1)
    periods = torch.div(x.shape[1], indices + 1, rounding_mode="floor")
    return periods, weights


class InceptionBlock(nn.Module):
    """不同尺度的二维卷积共享输入，对输出取均值。"""

    def __init__(self, input_size, output_size, num_kernels):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(input_size, output_size, kernel_size=2 * index + 1, padding=index)
            for index in range(num_kernels)
        ])
        for kernel in self.kernels:
            nn.init.kaiming_normal_(kernel.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(kernel.bias)

    def forward(self, x):
        return torch.stack([kernel(x) for kernel in self.kernels], dim=-1).mean(dim=-1)


class TimesBlock(nn.Module):
    """将一维序列按周期折叠，联合提取周期内与周期间变化。"""

    def __init__(self, hidden_size, d_ff, top_k, num_kernels):
        super().__init__()
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in (hidden_size, d_ff, top_k, num_kernels)):
            raise ValueError
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlock(hidden_size, d_ff, num_kernels),
            nn.GELU(),
            InceptionBlock(d_ff, hidden_size, num_kernels),
        )

    def forward(self, x):
        _, steps, channels = x.shape
        periods, amplitudes = fft_for_period(x, self.top_k)
        weights = amplitudes.softmax(dim=1)
        result = torch.zeros_like(x)
        # 相同周期集中卷积；同一样本的多个频率映射到同一周期时合并权重。
        for period in periods.unique().tolist():
            period_weights = (weights * (periods == period)).sum(dim=1)
            rows = torch.where(period_weights > 0)[0]
            if rows.numel() == 0:
                continue
            values = x[rows]
            padding = (-steps) % period
            values = F.pad(values, (0, 0, 0, padding))
            values = values.reshape(len(rows), -1, period, channels).permute(0, 3, 1, 2)
            values = self.conv(values.contiguous())
            values = values.permute(0, 2, 3, 1).reshape(len(rows), -1, channels)[:, :steps]
            result = result.index_add(0, rows, values * period_weights[rows, None, None])
        return x + result


class GridTimesNet(nn.Module):
    """历史序列经 TimesNet 编码，与当前动作和网格 embedding 合并预测。"""

    def __init__(self, history_size, grid_count, history_steps=4, hidden_size=64,
                 embedding_size=16, num_layers=2, dropout=0.1,
                 top_k=3, d_ff=64, num_kernels=3, context_size=len(CONTEXT_COLUMNS)):
        super().__init__()
        if (any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in (history_size, grid_count, history_steps, hidden_size,
                              embedding_size, num_layers, top_k, d_ff, num_kernels, context_size))
                or history_steps < 2 or hidden_size < 2 or not 0 <= dropout < 1):
            raise ValueError
        self.history_steps = history_steps
        self.history_size = history_size
        self.context_size = context_size
        self.value_embedding = nn.Conv1d(
            history_size, hidden_size, kernel_size=3, padding=1,
            padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.value_embedding.weight, mode="fan_in",
                                nonlinearity="leaky_relu")
        position = torch.arange(history_steps, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32)
                                * (-math.log(10000.0) / hidden_size))
        encoding = torch.zeros(1, history_steps, hidden_size)
        encoding[0, :, 0::2] = torch.sin(position * frequencies)
        encoding[0, :, 1::2] = torch.cos(position * frequencies[:hidden_size // 2])
        self.register_buffer("position_embedding", encoding)
        self.dropout = nn.Dropout(dropout)
        # 只从历史编码外推一个目标位置，不输入第 t 步的真实指标。
        self.temporal_projection = nn.Linear(history_steps, history_steps + 1)
        self.blocks = nn.ModuleList([
            TimesBlock(hidden_size, d_ff, top_k, num_kernels) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.grid_embedding = nn.Embedding(grid_count, embedding_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + embedding_size + context_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(TARGET_COLUMNS)),
        )

    def forward(self, history, context, grid_indices):
        if (history.ndim != 3 or history.shape[1:] != (self.history_steps, self.history_size)
                or context.shape != (len(history), self.context_size)
                or grid_indices.shape != (len(history),)):
            raise ValueError
        values = self.value_embedding(history.transpose(1, 2)).transpose(1, 2)
        values = self.dropout(values + self.position_embedding)
        values = self.temporal_projection(values.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            values = self.layer_norm(block(values))
        features = torch.cat([values[:, -1], self.grid_embedding(grid_indices), context], dim=1)
        # 沿用训练集 Z-score；六个输出量纲不同，不能用输入各通道反归一化。
        return self.head(features)


class TimesNetPredictor(SequencePredictor):
    """加载 TimesNet 最优模型，输出原始量纲的六项预测值。"""

    schema_version = "grid_timesnet_v1"
    model_class = GridTimesNet
