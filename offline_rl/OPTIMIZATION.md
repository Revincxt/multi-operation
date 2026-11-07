# 协同优化试验

使用原 47,000 次轨迹和原数据划分，保持六个独立智能体、每个两个动作头。奖励仍为全城平台佣金，推理允许相同区域动作重复。

| 版本 | 改变 |
| --- | --- |
| IQL-refit | 旧的 106 维局部观测，平行双头，作为重新提取策略的参照 |
| IQL-shared | 追加其他五区域的当前供需计数和上一周期动作，136 维观测 |
| IQL-cond | 在共享信息基础上，半径头以自己的价格选择为条件 |
| BC-cond | 与 IQL-cond 相同初始化、结构和批次，去掉优势权重 |
| OMIGA-adapt | 相同共享观测和条件双头，使用局部 Q/V、全城混合价值和局部优势权重 |

前三个 IQL 策略从随机参数开始，共用旧版第 20,000 次更新的固定 Q/V，进行 20,000 次策略提取更新。这样能够区分观测、动作头结构和策略重提取的影响；不能称为三个完整 IQL 从头训练。BC-cond 为同结构对照。

OMIGA-adapt 的价值与策略从头训练 20,000 次。价值目标与软目标网络参考 [论文](https://arxiv.org/abs/2307.11620)和[官方实现](https://github.com/ZhengYinan-AIR/OMIGA/blob/master/algos/OMIGA.py)。局部价值网络共享参数并带区域 ID，六个策略网络不共享参数；动作分布改为离散条件双头。奖励除以 100，alpha=0.3、discount=0.99、lr=3e-4、tau=0.005、梯度范数上限 10。它是项目适配，不是原 MuJoCo/SMAC 实验复现，也没有额外的无条件最优保证。

广播只读取当前动作前各区域的计数，以及历史中上一周期实际动作。半径训练使用记录价格计算正确的条件似然；执行时使用自身刚选出的价格。未输入其他区域真实当期动作，未重标原始动作。

## 选择与采用

1. 在 1,000、2,500、5,000、10,000、15,000、20,000 次更新，用固定五个验证种子比较；完整动作与九组合推理限制均保留。
2. 选出两个不同策略结构的候选，在另外二十个验证种子上确认并选定一个。BC 只作对照，不混称为强化学习策略。
3. 冻结选择后，与原 IQL-9、原 BC-9、新 BC、F3、LSTM-BCS 在二十个新的测试种子上比较；排除旧 IQL 和 LSTM 闭环测试种子。
4. 仅当确认验证相对原 IQL 至少提高 0.5%，且新测试的配对收益差为正、95% bootstrap 区间下界大于零时，才采用该候选。测试失败后不再改选其他模型。

采用只针对默认离线 RL 策略，不将“超过旧 IQL”解释为“超过 F3”。结果报告始终单列 F3 和 BC。当前试验使用一个策略初始化 seed；固定 critic 的三项 IQL 消融也共用同一个训练 critic，不能将其解释为多个独立的完整 IQL 训练 seed。

```bash
python3 -m offline_rl.optimize --device auto --workers 4

# 从最近一次完整的 checkpoint 恢复；已完成时不重复运行
python3 -m offline_rl.optimize --device auto --workers 4 --resume

# 用当前采用的策略与 F3 对比
python3 -m offline_rl.evaluate --output experiments/active_policy_comparison --runs 20 --workers 4
```

结果位于 `models/iql_coordination_20261102/`，包括训练曲线、验证轨迹、`selection.json`、新测试结果和 `adoption.json`。采用记录为 `offline_rl/active_policy.json`；`offline_rl.evaluate.Policy()` 默认读取该记录并检查 checkpoint 指纹。原模型和旧实验保留。

## 本次结果（2026-11-02）

五组均已完成 20,000 次更新。五种子筛选选出 IQL-cond-9（第 5,000 次更新）和 IQL-refit-9（第 20,000 次更新）；二十个额外验证种子选定前者。其确认验证收益为 7056.88 USD，旧 IQL-9 为 6999.56 USD，提升 0.82%，配对 95% 区间为 [−4.07, 119.91] USD。OMIGA-adapt 未进入前二。

冻结选择后，二十个新测试种子的 280 分钟平台佣金均值：

| 策略 | 收益（USD） | 匹配排队等待（秒） |
| --- | ---: | ---: |
| F3（价格 1.6、半径 2 km） | 7074.47 | 84.42 |
| 原 BC-9 | 7063.26 | 84.41 |
| IQL-cond-9 候选 | 7017.03 | 93.98 |
| 原 IQL-9 | 7004.18 | 103.91 |
| BC-cond-9 | 6994.33 | 99.43 |
| LSTM-BCS | 6788.80 | 71.63 |

候选较旧 IQL-9 仅增加 12.85 USD（0.18%），配对种子 bootstrap 95% 区间 [−41.99, 68.24] USD，9/20 次胜出。较 F3 减少 57.44 USD（−0.81%），区间 [−107.15, −7.36] USD；较原 BC-9 的差异也未排除零。候选虽改善了旧 IQL 的等待时间和原口径完成率，尚未证明目标佣金稳定改善。

**不采用该候选，默认保留 `models/iql_grid6_low_20261102/best_iql9.pt`。** 试验代码、候选权重与完整轨迹保留，测试后不改选其他模型。结论限于这一组配置、一个训练初始化种子和一个订单日期。

图和核对记录：

- [训练及验证曲线](../models/iql_coordination_20261102/training_curves.png)
- [新测试六指标对比](../models/iql_coordination_20261102/test/policy_comparison.png)
- [逐种子收益差与配对区间](../models/iql_coordination_20261102/test/paired_revenue_gain.png)
- [完整实验说明](../models/iql_coordination_20261102/experiment_report.txt)
- [结果核对记录](../experiments/iql_coordination_optimization_20261102/verification.json)

每幅图同时保留 PDF、SVG；PNG 为 600 dpi。完成率分母包含重复候选池，等待只含成功匹配乘客的排队时间；不能解释为独立订单送达率或所有乘客的总等待。
