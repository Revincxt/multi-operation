"""DL-MRD 时序编码器，以及带联合状态-动作融合的六指标预测器。"""

import torch
from torch import nn

from baseline.sequence import SequencePredictor, SequencePreprocessor, TARGET_COLUMNS


class _TemporalTransformer(nn.Module):
    """对决策前已观测的历史窗口编码。"""

    def __init__(self, history_size, history_steps, hidden_size, num_heads,
                 num_layers, d_ff):
        super().__init__()
        dimensions = (history_size, history_steps, hidden_size, num_heads,
                      num_layers, d_ff)
        if (any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in dimensions) or hidden_size % num_heads):
            raise ValueError
        self.history_size = history_size
        self.history_steps = history_steps
        self.hidden_size = hidden_size
        self.history_embedding = nn.Linear(history_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=d_ff,
            dropout=0., activation="relu", batch_first=True)
        self.history_encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False)
        self.register_buffer("temporal_mask", torch.triu(
            torch.ones(history_steps, history_steps, dtype=torch.bool), diagonal=1))

    def encode_history(self, history):
        """按最早到最近的顺序编码，保持既有因果注意力定义。"""
        if (history.ndim != 3 or not len(history)
                or history.shape[1:] != (self.history_steps, self.history_size)):
            raise ValueError
        return self.history_encoder(self.history_embedding(history), mask=self.temporal_mask)


class GridDLMRDAttentionOnly(_TemporalTransformer):
    """旧版线性末端模型，仅供读取既有权重和复现实验。"""

    def __init__(self, history_size, grid_count, history_steps=4, hidden_size=64,
                 num_heads=8, num_layers=3, d_ff=128, context_size=28):
        super().__init__(history_size, history_steps, hidden_size, num_heads,
                         num_layers, d_ff)
        if (not isinstance(grid_count, int) or isinstance(grid_count, bool) or grid_count <= 0
                or not isinstance(context_size, int) or isinstance(context_size, bool)
                or context_size <= 0):
            raise ValueError
        self.context_size = context_size
        # 旧模型按特征分别做仿射投影，之后直接接线性输出层。
        self.context_weight = nn.Parameter(torch.empty(context_size, hidden_size))
        self.context_bias = nn.Parameter(torch.empty(context_size, hidden_size))
        nn.init.uniform_(self.context_weight, -1., 1.)
        nn.init.uniform_(self.context_bias, -1., 1.)
        self.grid_embedding = nn.Embedding(grid_count, hidden_size)
        flattened_size = (history_steps + context_size + 1) * hidden_size
        self.output_heads = nn.ModuleList([
            nn.Linear(flattened_size, 1) for _ in TARGET_COLUMNS
        ])

    def forward(self, history, context, grid_indices):
        if (context.shape != (len(history), self.context_size)
                or grid_indices.shape != (len(history),)):
            raise ValueError
        temporal = self.encode_history(history)
        context_tokens = context[..., None] * self.context_weight + self.context_bias
        grid_token = self.grid_embedding(grid_indices)[:, None]
        features = torch.cat((temporal, context_tokens, grid_token), dim=1).flatten(1)
        return torch.cat([head(features) for head in self.output_heads], dim=1)


class GridDLMRD(_TemporalTransformer):
    """保留时序 Transformer，并非线性融合状态、全网格动作和网格身份。"""

    def __init__(self, history_size, grid_count, history_steps=4, hidden_size=64,
                 num_heads=8, num_layers=3, d_ff=128, context_size=28,
                 embedding_size=16, fusion_hidden_size=128, dropout=0.1):
        super().__init__(history_size, history_steps, hidden_size, num_heads,
                         num_layers, d_ff)
        if (not isinstance(grid_count, int) or isinstance(grid_count, bool) or grid_count <= 0
                or not isinstance(context_size, int) or isinstance(context_size, bool)
                or context_size <= 0
                or not isinstance(embedding_size, int) or isinstance(embedding_size, bool)
                or embedding_size <= 0
                or not isinstance(fusion_hidden_size, int)
                or isinstance(fusion_hidden_size, bool) or fusion_hidden_size <= 0
                or not 0 <= dropout < 1):
            raise ValueError
        self.context_size = context_size
        self.grid_embedding = nn.Embedding(grid_count, embedding_size)
        fused_size = history_steps * hidden_size + context_size + embedding_size
        self.state_action_fusion = nn.Sequential(
            nn.LayerNorm(fused_size),
            nn.Linear(fused_size, fusion_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 六个任务共享融合表征，各自保留独立输出头。
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in TARGET_COLUMNS
        ])

    def forward(self, history, context, grid_indices):
        if (context.shape != (len(history), self.context_size)
                or grid_indices.shape != (len(history),)):
            raise ValueError
        temporal = self.encode_history(history).flatten(1)
        features = torch.cat((temporal, context, self.grid_embedding(grid_indices)), dim=1)
        fused = self.state_action_fusion(features)
        # 线性末端改为共享的两层非线性融合，允许状态和联合动作产生交互效应。
        return torch.cat([head(fused) for head in self.output_heads], dim=1)


class DLMRDPredictor(SequencePredictor):
    """加载新旧 DL-MRD 权重，并按训练时保存的标准化参数恢复指标量纲。"""

    schema_version = "grid_dlmrd_v2"
    model_class = GridDLMRD

    @classmethod
    def load(cls, checkpoint_path, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        version = checkpoint.get("schema_version")
        model_classes = {
            "grid_dlmrd_v1": GridDLMRDAttentionOnly,
            cls.schema_version: cls.model_class,
        }
        if version not in model_classes:
            raise ValueError
        model = model_classes[version](**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        state = checkpoint["preprocessor"]
        if state.get("normalization") == "train_minmax":
            from baseline.dlmrd.data import MinMaxPreprocessor
            preprocessor = MinMaxPreprocessor.from_state_dict(state)
        else:
            preprocessor = SequencePreprocessor.from_state_dict(state)
        return cls(model, preprocessor, device)
