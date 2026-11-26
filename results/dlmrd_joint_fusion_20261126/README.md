# DL-MRD Joint Fusion：训练与闭环评估

本目录保存模型训练记录、六项预测误差、论文图件和闭环策略比较摘要。模型为 DL-MRD 的六指标适配版：三层时序 Transformer 编码历史；共享非线性层融合当前状态、六网格联合动作和网格身份。

## 训练

使用低密度数据的47,000条完整轨迹，按轨迹划分训练、验证和测试集，共15,792,000条监督样本。模型从随机权重训练100轮，不早停；使用 AdamW、WESM 动态损失平衡，并将平台收益任务倍率设为4.5。验证误差最低的检查点为第33轮，权重见 checkpoints/dlmrd/best_model.pt。

## 闭环评估

使用100个配对随机种子，在相同仿真条件和随机流下比较固定策略F3、LSTM-G、MA-IQL+及DL-MRD-BCS（最多5轮坐标搜索）。汇总表、配对差值、动作选择诊断及科研绘图位于 operational/。评估未提交订单数据和逐种子原始仿真目录。

## 文件

- training_history.csv、metrics.json、test_results.csv：训练过程与测试集预测指标。
- figures/：训练收敛、六任务误差、任务权重和测试误差图。
- operational/：100种子闭环结果及配对统计。
- evaluate_full.py：闭环评估脚本；运行仍需本地数据和基准协议文件。
- training_config.json：训练配置及数据指纹；数据和缓存未随仓库发布。
