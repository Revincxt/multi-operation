# DL-MRD 六指标预测基线

依据 Chen 等的 [DL-MRD 论文](https://doi.org/10.1016/j.tre.2024.103822)第4章和 Table 2 独立实现，参考[作者仓库固定版本](https://github.com/HKU-Smart-Mobility-Lab/DL-MRD-Broadcasting/tree/1e627ad895f92a06382d7ec01571aeeb5fa5f48c)，未复制作者代码。此处将其适配为本项目的联合定价与匹配半径预测基线，不是原论文广播模式实验的逐项复现。

## 网络与数据

输入为4个历史决策步，每步24维；当期包含28维数值上下文及网格 embedding，覆盖本网格与邻区状态、全城状态和6个网格的价格/半径。历史先经3层 Transformer encoder，再将时序表示、当前状态、联合动作与网格 embedding 输入共享非线性融合层，最后由6个独立输出头预测六项指标。

本项目原适配版用线性输出头，状态与动作之间的交互表达能力有限。新版保留历史 Transformer，并新增两层共享非线性融合网络，使其他网格动作、当前状态和网格身份可以共同影响预测；这是一项待实验验证的结构改动，不代表已经复现原论文的广播模式。历史注意力仍使用因果掩码，所有输入均来自决策时刻之前。

前馈层采用ReLU，历史使用因果注意力掩码，均属于具体实现选择。序列始终按批次、时间、特征排列，避免不同样本之间混用注意力。

沿用 `datasets_grid6_low_mixed47k_20261029/dataset_manifest.json` 的47,000条完整仿真划分，按完整轨迹隔离训练、验证、测试：

| 集合 | 仿真轨迹 | 监督样本 |
| --- | ---: | ---: |
| 训练 | 33,004 | 11,089,344 |
| 验证 | 6,948 | 2,334,528 |
| 测试 | 7,048 | 2,368,128 |
| 合计 | 47,000 | 15,792,000 |

这是现有数据约70%/15%/15%的划分，未重新采用论文的60%/40%划分。输入和六个标签的标准化参数仅用训练集拟合。网格编号按类别处理。

六指标沿用本项目数据口径：司机接单率不等于论文的司机利用率；匹配等待时间不等于接驾时间；平台收益与论文按已服务订单价格求和的定义也有区别。不同口径的绝对数值不能直接对照论文结果。

## 训练

```bash
python3 -m baseline.dlmrd.launch
```

默认从随机初始化开始训练100轮，AdamW、学习率 `1e-3`、权重衰减 `1e-3`、batch size `2048`。默认复用 LSTM 的训练集 z-score 缓存和同一轨迹划分；平台收益任务倍率为4.5，WESM仍动态平衡六任务。可用 `--normalization minmax` 运行原适配版的归一化口径。每轮记录 `training_history.csv`；`latest.pt` 保存模型、优化器和WESM状态，相同配置重启可续训。验证集等权平均标准化误差最低的检查点保存为 `best_model.pt`，测试集仅在训练结束后评估。

WESM 对每个任务的标准化 MSE 按时间滞后做指数加权：`γ=0.1`，保留10个历史训练步，每10个训练步更新一次任务权重。历史加权误差较大的任务获得较大权重，归一化后六个权重和为1。权重参与当前任务损失加权，历史误差不跨批次反向传播。论文算法中的下标存在排版混用，这里依其文字定义按时间滞后衰减、按任务维归一化。

WESM 是预测任务的损失平衡方法，不是运营目标权重。`decision.select_actions` 提供逐网格候选枚举：默认最大化预测收益，也支持按训练集标准化参数计算的六指标综合分数，等待时间取负向。此接口不包含长程规划，也不保证全城联合最优。

```python
from baseline.dlmrd.model import DLMRDPredictor
from baseline.dlmrd.decision import select_actions

model = DLMRDPredictor.load("checkpoints/dlmrd/best_model.pt")
actions = select_actions(model, features, price_values, radius_values)
```

`features` 必须为每个网格各一行历史及当前可观测特征，不含当前标签；联合动作列须与各网格动作一致。候选评分时其他网格动作固定为该输入上下文，返回的各行预测对应这一评分条件；闭环策略评估应另行固定场景与随机流。

每轮写入 `latest.pt` 和 CSV，暂停信号在当前完整轮次（含验证）结束后生效。再次执行相同启动命令可续训，完整恢复模型、Adam、WESM和随机状态；输出目录的训练配置与源码指纹需一致。

## 曲线与测试

```bash
python3 -m baseline.dlmrd.report models/dlmrd_joint_fusion_grid6_low47k_cpu100
```

`paper_figures/` 输出600 dpi PNG和矢量PDF：完整训练曲线及右侧前100轮、六指标误差曲线、WESM权重曲线。纵轴为标准化 MSE，保留原始观测，不做平滑；总体曲线是六任务等权平均，不是动态加权训练损失。

终测后生成 `test_results.csv`（MAE、RMSE、R²）和 `test_metrics` 图，与上一步观测值保持不变的 persistence 基线对照。图中比率误差用百分点表示，CSV保留原量纲。预测误差比较不等同于闭环运营收益比较。
