# LSTM / TimesNet + WESM

两个基线均使用网格 embedding 和当前联合动作，同时预测六个指标，使用 WESM 动态平衡训练损失。
当前 LSTM 使用四步历史、网格 embedding 和 28 维当前上下文：时间与自身动作、本区及邻区状态、四项全城状态，以及六个网格的价格与半径。
训练沿用数据集划分；对于管线生成的 date 模式数据，会额外检查日期隔离和先后顺序，不重新混洗划分。
方法来自 [Chen et al. (2025)](https://doi.org/10.1016/j.tre.2024.103822) 第 4.4 节 Algorithm 2（第 10 页），参数参考 Table 2（第 11 页）。

## 联合动作 BCS

当前模型已输入全网格动作和全城决策前状态，可用于比较一轮坐标贪心与多轮 BCS。独立动作的旧模型已清理。重新构建或训练对照模型时：

```bash
python3 -m data_pipeline.build_supervised_dataset \
  --input trajectories_grid6_low_mixed_20261027 --output datasets_joint_new \
  --history-steps 4 --joint-action-context --max-runs 47000
python3 -m data_pipeline.validate_supervised_dataset --input datasets_joint_new
python3 -m baseline.train_lstm --input datasets_joint_new \
  --output models/lstm_joint_new --epochs 100
python3 -m baseline.compare_bcs --dataset datasets_joint_new \
  --checkpoint models/lstm_joint_new/best_model.pt --runs 10 --max-sweeps 5 \
  --fixed-price 1.6 --fixed-radius 2
```

比较使用同一个新 LSTM 和同一批测试 seed：`greedy` 做一轮有序逐网格更新，`bcs` 最多迭代五轮并在稳定时提前结束；二者都按六个网格预测平台收益之和打分。它是坐标局部搜索，不保证全局最优。结果 CSV 与六指标 PNG/PDF 保存在 `experiments/bcs_comparison_*`。

## 损失计算

对于任务 `i`，先计算当前训练 batch 的标准化 MSE `l[i,t]`：

```text
L[i,t] = (l[i,t] + γ·l[i,t−1] + … + γ^T·l[i,t−T]) / (1 + γ + … + γ^T)
w[i,t] = L[i,t] / Σ(k=1…6) L[k,t]
loss   = Σ(i=1…6) w[i,t] · l[i,t]
```

- 历史误差越大，权重越高；不是倒数加权，也不是只训练最差的一项。
- 默认 `γ=0.1`、保留过去 10 个训练 batch 的损失、每 10 个 batch 更新一次权重。这里的损失历史窗口与 LSTM 的 4 步输入窗口独立。
- 初始化为每项 `1/6`。第 10、20……个 batch 用当前和历史损失更新权重，作用于该 batch 及后续更新间隔；历史不足时按已观察到的项归一化，全零误差退回等权。
- 历史在 epoch 之间保留；验证/测试不更新权重。权重和历史损失均脱离梯度计算，反向传播只作用于当前预测损失。
- 论文伪代码有下标混用：这里依照正文的时间衰减描述使用 `γ^j`，权重分母对六个任务求和，不以任务编号作衰减指数。

本项目移植多任务权重策略，不复现整套 DL-MRD：当前默认使用 AdamW（weight decay 为 0.001）、现有数据划分和训练集 Z-score 标准化；论文使用 Transformer 和 0～1 归一化。这是 WESM 在当前六指标模型上的适配，不是论文实验的完全复现。原 Adam 对照可显式指定 `--optimizer adam --weight-decay 0`。

## 运行

在项目根目录执行：

```bash
python3 -m pip install -r baseline/requirements.txt
python3 -m baseline.train_lstm --loss-weighting wesm --output models/lstm_wesm

# 等权对照，不改变其他参数
python3 -m baseline.train_lstm --loss-weighting equal --output models/lstm_equal

# TimesNet 使用同一份单时段数据，无需重新采集
python3 -m baseline.train_timesnet --input datasets_grid6_low_mixed47k_20261029 \
  --output models/timesnet_new --epochs 1000
python3 -m baseline.plot_results --input models/timesnet_new
```

策略参数为 `--wesm-gamma`、`--wesm-history-steps` 和 `--wesm-update-interval`。
默认完整训练 50 轮，可通过 `--epochs` 指定轮数，不使用早停。
已有非空输出目录不会覆盖，模型加载方式不变。

默认数据集 `datasets_grid6_low_mixed47k_20261029` 包含 47,000 次完整轨迹、15,792,000 条样本。保留的 LSTM 为 `models/lstm_grid6_low_mixed47k_mps100_20261029`，完成 100 轮训练，验证最优为第 84 轮。当前没有 TimesNet 权重；网络和训练代码保留。

## TimesNet

网络参考 [TimesNet 官方实现](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)：

```text
历史序列 → 数值/位置编码 → 时间投影 T→T+1 → TimesBlock × N
         → 目标位置编码 + 网格 embedding + 决策前状态/当前时间/半径/价格 → 六项预测
```

TimesBlock 用 FFT 选择周期，将序列折叠成二维，经过多尺度 Inception 卷积后按频率幅值加权，并保留残差连接。
默认隐藏维度 64、2 层、`top_k=3`、`d_ff=64`、3 个卷积尺度。
对应参数为 `--hidden-size`、`--num-layers`、`--top-k`、`--d-ff`、`--num-kernels`。

针对本项目的适配：

- 周期按每条样本选择，不按整个 batch 求平均，避免同一网格的预测受其他样本或推理批大小影响。
- 保留与 LSTM 一致的训练集 Z-score 和六输出回归头，不使用官方预测分支按输入通道进行的实例归一化/反归一化；输入特征与六个标签并不一一对应。
- 只读取历史、决策前本区/邻区状态、当前联合动作和网格信息，不输入当前周期真实标签；不改变仿真时段、动作或指标口径。
- 历史至少 2 步。现有 4 步历史可直接使用，但投影后的 5 个位置只有 2 个非零 FFT 频率，`top_k` 会自动裁剪到 2，不能据此学习长周期。修改历史长度需要重新构建数据集；比较两模型时应使用同一份数据。

这是一项 TimesNet 条件回归基线，不是所附论文的完整复现，也不保证比 LSTM 更好。参考代码的 MIT 许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

```python
from baseline.timesnet import TimesNetPredictor

model = TimesNetPredictor.load("models/timesnet_new/best_model.pt")  # 完成上述训练后
predictions = model.predict(features)  # 不需要 target_* 列
```

模型保存与加载使用各自的 schema，避免把 TimesNet checkpoint 误当成 LSTM。
训练、验证和推理由 `training.py` / `sequence.py` 共用；原来的 LSTM 命令、加载接口及 checkpoint 继续可用。

## 输出与比较

- `training_history.csv`：等权 train/validation MSE、实际加权训练损失、六项独立 MSE，以及每轮实际使用权重的样本加权均值。
- `best_model.pt`：按等权验证 MSE 选择的模型，同时保存当时的 WESM 配置、历史和权重；推理不需要 WESM。推理只读取保存的模型和预处理配置。
- `metrics.json`：模型名称、最佳 epoch、六项 MAE/RMSE/R² 和上一周期预测对照，`best_task_weights` 是最佳 epoch 结束时的权重，不是整轮均值。

最佳模型选择始终使用固定等权的验证 MSE，避免动态权重变化造成不同 epoch 不可比。WESM 的训练权重不是推理阶段的运营目标权重，也不保证每项指标都优于等权训练；应逐项比较，不把不同量纲的 MAE 直接相加。

训练完成后生成总损失曲线、六指标曲线和测试结果表：

```bash
python3 -m pip install matplotlib
python3 -m baseline.plot_results --input models/lstm_grid6_low_mixed47k_mps100_20261029
```

曲线使用可比较的等权标准化 MSE；实际 WESM 加权训练损失仍保存在 CSV 中。测试结果对应验证集最优模型，而非最后一轮模型。
绘图脚本同时支持两种模型，自动使用对应名称；`test_results.csv` 的模型列分别以 `lstm_` / `timesnet_` 开头。

`baseline.compare_models` 要求显式指定 `--input`、`--lstm-dir` 和 `--timesnet-dir`；不再默认读取已删除的历史目录。模型配置的数据指纹和划分必须一致，不能拿旧模型与新数据集冒充同批测试。

已有 LSTM 的固定/随机动作对比可运行 `python3 -m baseline.compare_lstm_fixed --include-random`；两网络闭环对比 `baseline.compare_policies` 必须显式传入 `--model-results`。

## 强化学习阶段

当前目录保留为监督学习对照。六智能体离线 IQL 和 BC 在 [offline_rl/](../offline_rl/README.md)，复用已有缓存及真实仿真评估。旧模型权重已删除，既有比较图与统计保存在 `experiments/`。
