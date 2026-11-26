"""DL-MRD 的逐网格候选评分适配；默认选择预测平台收益最大的动作。"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from baseline.sequence import SequencePreprocessor, TARGET_COLUMNS
from src.utils.joint_context import joint_action_columns


def _candidates(values):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or (values <= 0).any() or len(np.unique(values)) != len(values)):
        raise ValueError
    return values


def select_actions(predictor, frame, price_values, radius_values,
                   objective="revenue", metric_weights=None):
    """返回每个网格的价格、半径、六项原量纲预测及有符号 score。

    frame 每个已知网格恰有一行，只含历史及当前可观测输入，不接受 target 列。
    枚举某网格的自身动作时，其联合动作列同步更新，其他网格保持输入上下文。
    revenue 最大化原量纲收益；paper_score 用训练集拟合的标准化参数归一化预测，
    默认六项等权，等待时间取负向。metric_weights 可为六项序列或完整目标列映射，
    非负权重会归一化到和为 1；预测和分数均不裁剪，也不对负分取绝对值。
    这是论文逐网格候选评分的六指标适配，不包含长程规划或全城联合最优保证。
    同分时按传入的价格、半径候选顺序选择。
    """
    if (objective not in ("revenue", "paper_score") or frame.empty
            or any(str(column).startswith("target_") for column in frame.columns)
            or (objective == "revenue" and metric_weights is not None)):
        raise ValueError
    prices, radii = _candidates(price_values), _candidates(radius_values)
    processor = predictor.preprocessor
    grids = SequencePreprocessor._grid_values(frame)
    if len(set(grids)) != len(grids) or set(grids) != set(processor.grid_ids):
        raise ValueError
    if not {"price_multiplier", "matching_radius_km"}.issubset(frame.columns):
        raise ValueError
    current = frame.assign(grid_id=grids).set_index("grid_id")
    joint_columns = joint_action_columns(processor.grid_ids)
    has_joint = any(column in frame for column in joint_columns)
    if has_joint or getattr(processor, "has_joint_actions", False):
        if not set(joint_columns).issubset(frame.columns):
            raise ValueError
        for grid in processor.grid_ids:
            for prefix, action in (("price", "price_multiplier"), ("radius", "matching_radius_km")):
                values = frame[f"joint_{prefix}_grid_{grid}"].to_numpy(dtype=float)
                if not np.isfinite(values).all() or not np.allclose(
                        values, float(current.loc[grid, action]), rtol=1e-7, atol=1e-8):
                    raise ValueError

    if objective == "paper_score":
        if isinstance(metric_weights, Mapping):
            if set(metric_weights) != set(TARGET_COLUMNS):
                raise ValueError
            metric_weights = [metric_weights[column] for column in TARGET_COLUMNS]
        weights = np.ones(len(TARGET_COLUMNS)) if metric_weights is None else np.asarray(metric_weights, dtype=float)
        if (weights.shape != (len(TARGET_COLUMNS),) or not np.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError
        weights = weights / weights.sum()
        weights[list(TARGET_COLUMNS).index("target_mean_wait_to_match_seconds")] *= -1
        stats = processor.scalers["targets"]

    price_grid = np.repeat(prices, len(radii))
    radius_grid = np.tile(radii, len(prices))
    selected = []
    for grid in processor.grid_ids:
        row = frame.loc[grids == grid]
        candidates = row.iloc[np.zeros(len(price_grid), dtype=int)].reset_index(drop=True).copy()
        candidates["price_multiplier"], candidates["matching_radius_km"] = price_grid, radius_grid
        if has_joint:
            candidates[f"joint_price_grid_{grid}"] = price_grid
            candidates[f"joint_radius_grid_{grid}"] = radius_grid
        prediction = predictor.predict(candidates)
        if len(prediction) != len(candidates) or not set(TARGET_COLUMNS).issubset(prediction.columns):
            raise ValueError
        values = prediction[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError
        if objective == "revenue":
            scores = values[:, list(TARGET_COLUMNS).index("target_platform_revenue")]
        else:
            scores = ((values - stats["mean"]) / stats["scale"]) @ weights
        if not np.isfinite(scores).all():
            raise ValueError
        best = int(np.argmax(scores))
        selected.append({"grid_id": int(grid), "price_multiplier": float(price_grid[best]),
                         "matching_radius_km": float(radius_grid[best]),
                         **dict(zip(TARGET_COLUMNS, values[best])), "score": float(scores[best])})
    return pd.DataFrame(selected)
