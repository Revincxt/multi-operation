# 六网格离线策略学习

当前默认模型为[保守actor-critic，初始化74](../models/conservative_policy_joint100k_20261103/train_seed74/policy.pt)，由`active_policy.json`指定。在6网格、250司机的低密度场景中，100个新测试种子相对F3增收0.29%，配对95%区间[0.12,40.29]美元；另50个独立确认种子增收0.63%，区间[17.07,72.44]美元。相对EQ-Joint也通过两阶段门槛，旧权重保留。另两组初始化仍弱于对照，当前只采用验证选中的模型，不代表训练对初始化已稳定。[报告](../models/conservative_policy_joint100k_20261103/experiment_report.txt)、[六指标](../models/conservative_policy_joint100k_20261103/test/paper_comparison.png)、[训练曲线](../models/conservative_policy_joint100k_20261103/training_curves.png)。

根据同状态回放诊断，新增保守actor-critic：Q使用实际目标策略的联合MAP动作构造TD目标，在训练中加入逐网格反事实保守项；Q与actor同时学习队列、司机状态、接驾取消及收益辅助任务。六个区域以全城Q作策略改进，保留价格条件半径双头、全部110个动作及原四步历史窗口。这是CFCQL式项目适配，尚未实现持续递归记忆或完整InSPO，不能引用其理论保证。

默认复用完整100k轨迹的原划分，Q从已训练的三个STQ成员初始化，actor重新训练；每组1000次行为初始化、2000次初始策略评估、6000次联合学习及1000次固定最终策略评估。每500次保存完整断点，相同参数加`--resume`可继续；恢复时源码需与本轮`code_snapshot`一致。导出策略与TD目标使用相同的确定性执行方式，执行时不附加Q搜索。结果位于`models/conservative_policy_full100k_20261103/`，独立验证和测试完成前不替换默认模型。

首轮仅使用反事实项，验证选中的初始化在100个新测试种子上相对F3为−2.94%，未采用。开发验证显示局部反事实项不足以约束实际联合动作的估值差，因此下一轮加入当前/目标策略联合动作的CQL式正则，非负拉格朗日系数自动学习，平均估值差目标为5美元；该目标不构成实际收益保证。保留相同初始化、批次和预算，复用开发验证池，重新保留最终测试种子。

```bash
python3 -m offline_rl.train_conservative --device mps \
  --output models/conservative_policy_joint100k_20261103 \
  --policy-gap-limit 0.05 \
  --validation-protocol models/conservative_policy_full100k_20261103/protocol.json
```

旧结果及其训练源码快照保留。新版已完成100种子测试和50种子独立确认，采用记录见`models/conservative_policy_joint100k_20261103/adoption.json`。

时空联合价值试验已完成：Q/V从随机初始化训练三个成员，各25,000次更新，有限终点gamma=1，加入时序编码、网格注意力、数值动作基及非可加交互。发现单区域搜索的协调局部最优后，增加全部110种全城统一动作中的最佳起点，最终仍自由逐网格调整。同权重新100种子对照中，收益由6880.38恢复至7027.03 USD（+2.13%），但F3为7042.49、EQ-Joint为7059.66，未替换默认策略。[六指标](../models/structured_joint_search_20261103/test/paper_comparison/policy_comparison.png)、[收益差](../models/structured_joint_search_20261103/test/paired_revenue_gain.png)、[报告](../models/structured_joint_search_20261103/experiment_report.txt)。

新一轮训练使用`python3 -m offline_rl.train_structured --output models/新目录`；恢复断点时，源码需与该轮`code_snapshot`一致，再使用相同参数加`--resume`。原始replay只读，四期历史以无损float32滑窗驻留内存。搜索修正位于`offline_rl/joint_starts.py`，本轮固定权重对照的步骤保存在`experiments/structured_joint_search_20261103/run_experiment.py`及结果目录的protocol中。

完整100k数据的价值引导策略提取（EQ-Joint）已完成：100个新配对测试种子上，相对完整动作IQL增收3.74%，相对旧IQL-9增收1.30%；相对F3仅增加0.05%，配对95%区间包含零，未替换默认模型。全部110个单区域动作均合法，无F3覆盖或手动禁选。[六指标图](../models/value_guided_full100k_20261103/test/paper_comparison/policy_comparison.png)、[配对收益图](../models/value_guided_full100k_20261103/test/paired_revenue_gain.png)、[报告](../models/value_guided_full100k_20261103/experiment_report.txt)。

轻量协同策略试验由`python3 -m offline_rl.train_cooperative`执行：冻结上述三个IQL目标双Q，重新训练四期LSTM、六网格注意力和价格条件半径双头，共61,530参数；500次行为初始化加12,000次策略更新，MPS实测约8.4分钟。验证仅选择纯策略或附加联合价值搜索，最终使用100个新种子评估。完整断点为`models/cooperative_full100k_20261103/latest.pt`，相同参数加`--resume`可恢复。它是IQL价值引导的策略适配，不是标准InSPO复现；训练交叉熵不能直接解释为运营收益改善。

该试验已完成：新版CQPI-refined为7061.41 USD、EQ-Joint为7056.72 USD、F3为7064.10 USD。新版相对F3为−2.69 USD，配对95%区间[−23.69,18.12]；相对EQ-Joint为+4.69 USD，区间[−13.75,22.29]，均未证明优势，默认策略保持原状。[改进总览](../models/cooperative_full100k_20261103/method_progress.png)、[六指标](../models/cooperative_full100k_20261103/test/paper_comparison/policy_comparison.png)、[训练曲线](../models/cooperative_full100k_20261103/training_curves.png)、[报告](../models/cooperative_full100k_20261103/experiment_report.txt)。两次试验采用不同的100种子测试池，原始收益均值不能跨池直接作差。

状态共享、条件双头和局部信用分配的新对照见[协同优化试验](OPTIMIZATION.md)。默认评估可通过 `python3 -m offline_rl.evaluate --output experiments/active_policy_comparison` 使用已验证采用的策略。

五个训练初始化、100个新测试种子的[全动作联合策略对照](JOINT_ABLATION.md)已完成：自回归结构未改善收益，诊断发现固定历史下缺少动作切换样本。本轮未替换默认模型。

完整100k数据试验也已完成（2026-11-03）：70k训练、15k验证、15k测试，保持原划分、训练集标准化及每初始化10k更新预算。五个初始化在100个新配对测试种子上，100k与47k完整动作IQL均为6812.97 USD，F3为7062.27 USD；相对F3为−3.53%，95%层级区间[−281.29, −216.52] USD。未替换默认模型。[完整报告](../models/iql_full100k_20261103/experiment_report.txt)、[数据量效果图](../models/iql_full100k_20261103/test/data_volume_effect.png)。

100k转移目录为`datasets_grid6_low_mixed100k_20261103/offline_rl/`，可用`python3 -m offline_rl.prepare_collection --resume`检查已完成构建。复现本轮训练/评估用`python3 experiments/iql_full100k_20261103/run_experiment.py --resume`；已完成时保留结果，训练中每1,000更新保存完整断点。实际批次序列已核对，五个初始化均访问全部70,000条训练轨迹。

该轮协同优化的候选在二十个新测试种子上较原 IQL-9 增收0.18%，配对95%区间覆盖零，未通过采用门槛，当时保留原IQL-9。

六个区域各有一个独立策略网络，每个网络有两个并行分类头：11 个价格、10 个匹配半径。总共 126 个 logits，不枚举全城联合动作。各智能体参数独立，执行时只读本区四步历史、当前时间、本区及一阶邻区的订单/司机状态，共 106 维。集中式双 Q 网络和 V 网络读取 660 维全城历史与决策前状态；Q 还读取六区域完整动作，以全城平台累计佣金为目标。

这是 [IQL](https://arxiv.org/abs/2110.06169) 的多智能体、离散双头策略适配。保留[官方实现](https://github.com/ikostrikov/implicit_q_learning)的 expectile 价值回归、双 Q Bellman 回归、优势加权行为克隆和软目标更新；不是官方多智能体实现，也不是 CQL。采用自己编写的 PyTorch 实现，依赖与 `baseline/requirements.txt` 相同。

## 数据

复用前 47,000 次完整轨迹的现有磁盘缓存，不重新解析 18 GB 监督 CSV，不改变已有数据及划分。

| 划分 | 轨迹 | 城市级转移 |
| --- | ---: | ---: |
| train | 33,004 | 1,848,224 |
| validation | 6,948 | 389,088 |
| test | 7,048 | 394,688 |

- 一条转移包含六网格共同的状态、六区域真实动作、六网格当期佣金之和、下一决策状态。
- 状态排除当前价格、当前半径、当前联合动作及当前结果。历史动作可观测。
- 从第 4 个决策周期开始，与原基线的 20 分钟预热保持一致。
- 10:00 是本次有限时域目标的终止点，保留最后一个周期收益，终止样本不自举、不跨轨迹连接。
- 特征沿用仅在训练集拟合的标准化器。奖励除以固定 100，保持美元收益排序；不减均值。
- 历史与汇总状态不能还原全部排队订单和司机剩余行程，是仿真器的部分观测。

## 方法

```text
V 损失 = expectile_0.7(min(Q1_target, Q2_target) - V)
Q 目标 = reward / 100 + 0.99 × nonterminal × V(next_state)
策略权重 = min(exp(3 × (min(Q1_target, Q2_target) - V)), 100)
策略损失 = -mean(策略权重 × 联合动作 log_probability)
```

联合对数概率是六智能体价格和半径共 12 项对数概率之和。六区域共用全城优势；Q 是完整联合动作函数，不按网格相加。BC 使用相同网络结构、相同初始化与训练批次，去掉优势权重。两个输出头在给定本区观测时独立，这是当前策略分布的限制。

原始IQL默认 Adam 3e-4，隐藏层 256，折扣 0.99/五分钟，expectile 0.7，优势系数 3，目标软更新系数 0.005，batch 512，20,000 次梯度更新。仅策略学习率余弦衰减。模型与 BC 都从随机权重开始，不使用 LSTM 权重。

## 运行

```bash
python3 -m pip install -r baseline/requirements.txt

# 一次性构建；不覆盖已有目录
python3 -m offline_rl.data

# 离线训练、验证选模型及最终测试
python3 -m offline_rl.train --device auto --steps 20000 --workers 4

# 从最近一次完整 checkpoint 继续相同配置的训练
python3 -m offline_rl.train --device auto --steps 20000 --workers 4 --resume
```

以上原始IQL命令使用转移目录`datasets_grid6_low_mixed47k_20261029/offline_rl/`，训练输出目录为`models/iql_grid6_low_20261102/`。`latest.pt`每1000次更新保存IQL、BC、目标网络、优化器和采样RNG状态，中断后恢复到最近一次保存。

## 评估

每 5,000 次更新，使用固定抽取的五条 validation 轨迹种子进行真实闭环仿真，分别选择各策略版本验证收益最高的 checkpoint。完成后在另外十个 test 种子上与 F3、随机策略、LSTM 贪心及 LSTM-BCS 比较，测试结果不用于训练或选 checkpoint。验证中使用仿真器选择 checkpoint，因此不是完全不接触环境的离线模型选择。

评估所有方法共同预热 20 分钟（价格 1.5、半径 3 km），比较 05:20–10:00 的真实运营结果。F3 为价格 1.6、半径 2 km。按种子共享外生订单、司机初态及预热，记录哈希；收益差给出配对种子 bootstrap 95% 区间。

额外比较 `IQL-9` 和 `BC-9`：在推理时将价格限制为 `{1.4, 1.6, 1.8}`、半径限制为 `{1.5, 2.0, 2.5}`，每网格九种组合。训练保留所有真实动作；不修改标签，不把旧动作映射为新动作。它是同一类策略的推理约束对照，不是独立的九动作数据集训练。

输出：

```text
training_history.csv / training_curves.{png,pdf,svg}
validation_history.csv / validation/step*/
training_config.json / training_status.json
latest.pt / best_{iql,iql9,bc,bc9}.pt
test/run_summary.csv / policy_summary.csv
test/paired_revenue_comparisons.csv
test/policy_comparison.{png,pdf,svg}
test/runs/seed*/{policy}/
```

六项指标沿用仿真器原口径；“完成率”分母累计重复候选池，“平均等待”仅含成功匹配乘客的排队等待。首次试验只有一个训练随机种子和一个原始订单日期，不能据此声称跨日期泛化或 IQL 优于其他离线 RL 算法。

## 首版结果（2026-11-02）

已用 MPS 完成 20,000 次更新，含验证约 12.3 分钟，含最终测试约 15.9 分钟。十个测试种子的 280 分钟平台佣金均值：

| 策略 | 收益（USD） |
| --- | ---: |
| F3 | 7119.26 |
| BC-9 | 6990.41 |
| IQL-9 | 6973.50 |
| LSTM 贪心 | 6792.21 |
| LSTM-BCS | 6776.07 |
| IQL / BC（完整动作） | 6771.85 |
| 随机 | 5463.12 |

IQL-9 相对 LSTM 贪心提高 2.67%，相对 F3 下降 2.05%。IQL-9 与 BC-9 的配对收益差为 −16.90 USD，95% seed bootstrap 区间 [−91.62, 52.66]，未证明强化学习优于行为克隆。

完整动作的 IQL 与 BC 在 3,360 次网格决策中均延续预热动作（价格 1.5、半径 3 km），各自独立仿真得到相同结果。两者网络参数不同，不能把结果一致解释为未训练或复用仿真。IQL-9 的服务指标亦有取舍：排队等待 105.98 秒，F3 为 84.83 秒。

完整结果为 `models/iql_grid6_low_20261102/experiment_report.txt`，核对记录为 `experiments/offline_rl_iql_20261102/verification.json`。
