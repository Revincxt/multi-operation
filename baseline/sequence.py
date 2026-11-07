"""时序基线共用的特征预处理与六指标推理接口。"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.env.observation import STATE_COLUMNS
from src.utils.joint_context import joint_schema


TARGET_COLUMNS = (
    "target_baseline_request_num",
    "target_request_rate",
    "target_driver_accept_rate",
    "target_order_completion_rate",
    "target_mean_wait_to_match_seconds",
    "target_platform_revenue",
)

CONTEXT_COLUMNS = (
    "decision_start_sim_time",
    "interval_minutes",
    "price_multiplier",
    "matching_radius_km",
)


class RunningMoments:
    """合并分块均值和方差，不保留已读样本。"""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, values):
        if len(values) == 0:
            return
        count = len(values)
        mean = values.mean(axis=0, dtype=np.float64)
        m2 = values.var(axis=0, dtype=np.float64) * count
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean, m2
            return
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + delta ** 2 * self.count * count / total
        self.mean += delta * count / total
        self.count = total


def _numeric_values(frame, columns):
    """读取数值列，不静默填补缺失的历史或标签。"""
    values = frame[list(columns)].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError
    return values


class SequencePreprocessor:
    """恢复时间顺序，仅用训练集拟合标准化参数和网格编号。"""

    def __init__(self, history_steps, history_features, context_columns=None):
        if (not isinstance(history_steps, int) or history_steps <= 0
                or not history_features
                or len(set(history_features)) != len(history_features)):
            raise ValueError
        self.history_steps = history_steps
        self.history_features = list(history_features)
        self.context_columns = list(CONTEXT_COLUMNS if context_columns is None else context_columns)
        if self.context_columns[:len(CONTEXT_COLUMNS)] != list(CONTEXT_COLUMNS):
            raise ValueError
        extras = self.context_columns[len(CONTEXT_COLUMNS):]
        state = list(STATE_COLUMNS) if set(extras).intersection(STATE_COLUMNS) else []
        if state and extras[:len(state)] != state:
            raise ValueError
        remaining = extras[len(state):]
        try:
            grid_ids, global_columns, action_columns = joint_schema(remaining)
        except (TypeError, ValueError):
            raise ValueError from None
        expected = list(global_columns) + list(action_columns)
        if remaining != expected or (global_columns and not state):
            raise ValueError
        self.joint_grid_ids = list(grid_ids)
        self.has_joint_actions = bool(action_columns)

    def _raw_inputs(self, frame):
        # CSV 从 lag_1 开始存储，时序模型按最早到最近的顺序读取。
        history = np.stack([
            _numeric_values(frame, [
                f"lag_{lag}_{column}" for column in self.history_features])
            for lag in range(self.history_steps, 0, -1)
        ], axis=1)
        context = _numeric_values(frame, self.context_columns)
        return history, context

    @staticmethod
    def _grid_values(frame):
        values = frame["grid_id"].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or (values < 0).any()
                or not np.equal(values, np.floor(values)).all()):
            raise ValueError
        return values.astype(np.int64)

    def fit(self, frame):
        """只传入 train 行，验证集和测试集不能参与拟合。"""
        return self.fit_chunks((frame,))

    def fit_chunks(self, frames):
        """逐块拟合训练集，计算结果与整表总体均值、方差一致。"""
        moments = {name: RunningMoments() for name in ("history", "context", "targets")}
        grid_ids = set()
        for frame in frames:
            if frame.empty:
                continue
            if "split" in frame and not frame["split"].eq("train").all():
                raise ValueError
            history, context = self._raw_inputs(frame)
            grid_ids.update(self._grid_values(frame).tolist())
            for name, values in (
                ("history", history.reshape(-1, history.shape[-1])),
                ("context", context),
                ("targets", _numeric_values(frame, TARGET_COLUMNS)),
            ):
                moments[name].update(values)
        if not grid_ids:
            raise ValueError
        self.grid_ids = sorted(grid_ids)
        if self.has_joint_actions and self.joint_grid_ids != self.grid_ids:
            raise ValueError
        self.fit_row_count = moments["targets"].count
        self.scalers = {}
        for name, stats in moments.items():
            mean = stats.mean.astype(np.float32)
            scale = np.sqrt(stats.m2 / stats.count).astype(np.float32)
            scale[scale < 1e-6] = 1.0
            self.scalers[name] = {"mean": mean, "scale": scale}
        return self

    def _standardize(self, name, values):
        stats = self.scalers[name]
        return (values - stats["mean"]) / stats["scale"]

    def transform_inputs(self, frame):
        """推理只需要输入特征，不读取当前周期标签。"""
        history, context = self._raw_inputs(frame)
        mapping = {grid_id: index for index, grid_id in enumerate(self.grid_ids)}
        grid_values = self._grid_values(frame)
        if not set(grid_values).issubset(mapping):
            raise ValueError
        grid_indices = np.asarray([mapping[value] for value in grid_values], dtype=np.int64)
        return (
            self._standardize("history", history),
            self._standardize("context", context),
            grid_indices,
        )

    def transform_targets(self, frame):
        return self._standardize("targets", _numeric_values(frame, TARGET_COLUMNS))

    def inverse_targets(self, values):
        stats = self.scalers["targets"]
        return values * stats["scale"] + stats["mean"]

    def state_dict(self):
        """保存纯数值参数，重新加载时不需要训练数据。"""
        return {
            "history_steps": self.history_steps,
            "history_features": self.history_features,
            "context_columns": self.context_columns,
            "target_columns": list(TARGET_COLUMNS),
            "grid_ids": self.grid_ids,
            "fit_row_count": self.fit_row_count,
            "scalers": {
                name: {key: value.tolist() for key, value in stats.items()}
                for name, stats in self.scalers.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state):
        if state["target_columns"] != list(TARGET_COLUMNS):
            raise ValueError
        instance = cls(state["history_steps"], state["history_features"], state["context_columns"])
        instance.grid_ids = state["grid_ids"]
        instance.fit_row_count = state.get("fit_row_count")
        instance.scalers = {
            name: {key: np.asarray(value, dtype=np.float32) for key, value in stats.items()}
            for name, stats in state["scalers"].items()
        }
        return instance


class SequencePredictor:
    """加载最佳模型，对已有网格的新历史及候选动作进行预测。"""

    def __init__(self, model, preprocessor, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.preprocessor = preprocessor

    @classmethod
    def load(cls, checkpoint_path, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["schema_version"] != cls.schema_version:
            raise ValueError
        model = cls.model_class(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        preprocessor = SequencePreprocessor.from_state_dict(checkpoint["preprocessor"])
        return cls(model, preprocessor, device)

    def predict(self, frame, batch_size=512):
        if frame.empty or batch_size <= 0:
            raise ValueError
        tensors = [torch.from_numpy(values)
                   for values in self.preprocessor.transform_inputs(frame)]
        loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False)
        predictions = []
        with torch.inference_mode():
            for batch in loader:
                outputs = self.model(*(values.to(self.device) for values in batch))
                predictions.append(outputs.cpu().numpy())
        values = self.preprocessor.inverse_targets(np.concatenate(predictions))
        return pd.DataFrame(values, columns=TARGET_COLUMNS, index=frame.index)
