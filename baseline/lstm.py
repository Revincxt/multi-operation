"""LSTM 基线：历史序列、当前联合动作和网格 embedding 的多输出预测。"""

import torch
from torch import nn

from baseline.sequence import (
    CONTEXT_COLUMNS,
    TARGET_COLUMNS,
    SequencePredictor,
    SequencePreprocessor,
)


class GridLSTM(nn.Module):
    """全部网格共享 LSTM，通过 embedding 区分网格差异。"""

    def __init__(self, history_size, grid_count, hidden_size=64,
                 embedding_size=16, num_layers=1, dropout=0.1, context_size=len(CONTEXT_COLUMNS)):
        super().__init__()
        if isinstance(context_size, bool) or not isinstance(context_size, int) or context_size <= 0:
            raise ValueError
        self.encoder = nn.LSTM(
            input_size=history_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.grid_embedding = nn.Embedding(grid_count, embedding_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + embedding_size + context_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(TARGET_COLUMNS)),
        )

    def forward(self, history, context, grid_indices):
        _, (hidden, _) = self.encoder(history)
        features = torch.cat([
            hidden[-1], self.grid_embedding(grid_indices), context,
        ], dim=1)
        # 输出为标准化后的回归值；request_rate 不是限于 [0, 1] 的概率。
        return self.head(features)


class LSTMPredictor(SequencePredictor):
    """加载已有 LSTM checkpoint，接口和参数格式保持不变。"""

    schema_version = "grid_lstm_v1"
    model_class = GridLSTM
