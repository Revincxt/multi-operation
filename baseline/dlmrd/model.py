"""DL-MRD 的时序注意力、末端上下文拼接和独立多任务线性输出。

对应论文第4.3节 Eq. (6)–(15)，适配本项目的历史特征、联合动作和六项指标。
64维隐藏层、128维前馈层，以及逐特征上下文 token 是本项目的具体设置。
"""

import torch
from torch import nn

from baseline.sequence import TARGET_COLUMNS, SequencePredictor


class GridDLMRD(nn.Module):
    """时间维自注意力；当前状态、候选动作和网格身份只在输出前拼接。"""

    def __init__(self, history_size, grid_count, history_steps=4, hidden_size=64,
                 num_heads=8, num_layers=3, d_ff=128, context_size=28):
        super().__init__()
        dimensions = (history_size, grid_count, history_steps, hidden_size,
                      num_heads, num_layers, d_ff, context_size)
        if (any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in dimensions) or hidden_size % num_heads):
            raise ValueError
        self.history_size = history_size
        self.history_steps = history_steps
        self.context_size = context_size
        self.history_embedding = nn.Linear(history_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=d_ff,
            dropout=0., activation="relu", batch_first=True)
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                    enable_nested_tensor=False)
        self.register_buffer("temporal_mask", torch.triu(
            torch.ones(history_steps, history_steps, dtype=torch.bool), diagonal=1))
        # 每个数值特征有自己的仿射映射，避免价格、半径及不同状态失去身份。
        self.context_weight = nn.Parameter(torch.empty(context_size, hidden_size))
        self.context_bias = nn.Parameter(torch.empty(context_size, hidden_size))
        nn.init.uniform_(self.context_weight, -1., 1.)
        nn.init.uniform_(self.context_bias, -1., 1.)
        self.grid_embedding = nn.Embedding(grid_count, hidden_size)
        flattened_size = (history_steps + context_size + 1) * hidden_size
        self.output_heads = nn.ModuleList([
            nn.Linear(flattened_size, 1) for _ in TARGET_COLUMNS
        ])

    def encode_history(self, history):
        """按最早到最近的顺序编码；历史位置不能读取其后的位置。"""
        if (history.ndim != 3 or not len(history)
                or history.shape[1:] != (self.history_steps, self.history_size)):
            raise ValueError
        return self.history_encoder(self.history_embedding(history), mask=self.temporal_mask)

    def forward(self, history, context, grid_indices):
        if (context.shape != (len(history), self.context_size)
                or grid_indices.shape != (len(history),)):
            raise ValueError
        temporal = self.encode_history(history)
        context_tokens = context[..., None] * self.context_weight + self.context_bias
        grid_token = self.grid_embedding(grid_indices)[:, None]
        features = torch.cat((temporal, context_tokens, grid_token), dim=1).flatten(1)
        # Min-Max 在训练范围外可以超出 [0, 1]，回归输出不添加 sigmoid 或截断。
        return torch.cat([head(features) for head in self.output_heads], dim=1)


class DLMRDPredictor(SequencePredictor):
    """加载 DL-MRD 与训练集拟合的 Min-Max 参数，恢复原始指标量纲。"""

    schema_version = "grid_dlmrd_v1"
    model_class = GridDLMRD

    @classmethod
    def load(cls, checkpoint_path, device="cpu"):
        from baseline.dlmrd.data import MinMaxPreprocessor

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["schema_version"] != cls.schema_version:
            raise ValueError
        model = cls.model_class(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        preprocessor = MinMaxPreprocessor.from_state_dict(checkpoint["preprocessor"])
        return cls(model, preprocessor, device)
