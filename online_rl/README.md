# 在线联合决策实验

研究问题：在相同仿真条件与交互预算下，显式条件化其他区域的当期动作，能否提高全城累计平台收益？

方法名称暂记为 **AR-PPO**。采用MAT启发的因果动作解码器、四周期LSTM窗口和集中式PPO，属于CTCE。这是针对当前问题的适配，不是MAT论文的逐行复现，不沿用其单调改进结论。区域顺序只用于策略内部解码，六区域动作同时提交仿真器。

## 研究设计

1. **问题与目标**：六区域部分可观测、有限时域控制。每5分钟决策一次，共20分钟共同预热和56次计分决策。奖励为该周期六区域平台佣金之和，目标是05:20—10:00累计美元收益，γ=1；10:00为任务终点，不作bootstrap，不延长到清空订单。收益以100美元为单位缩放，仅用于训练数值稳定。
2. **观测与信息边界**：复用24维历史指标×4周期及每区域14维动作前上下文，合计660维。包括过去动作，排除当前候选动作、当期结果、未来订单。历史窗口使用现有训练集的冻结特征标准化；网络权重随机初始化。它是有限历史窗口策略，不是跨整个episode保存隐藏状态的循环策略。
3. **策略**：共享LSTM编码各区域历史，区域Embedding表达异质性，全局注意力表达区域联系。因果解码器右移输入，区域i只看已生成的前缀动作。区域内先定义价格分布，再定义给定价格的半径分布，保留11×10个合法组合。固定区域顺序0—5，不加动作屏蔽、F3回退或禁止重复动作。
4. **价值函数**：独立时序/空间编码器输出全城V(H)，不使用候选动作Q搜索。actor与critic编码参数独立，防止价值回归直接改变策略表示。
5. **训练**：一次收集固定策略版本下的完整episode，再用GAE(λ=0.95)估计优势。PPO采用完整六区域联合概率比 `exp(logπ_new(a|H)-logπ_old(a|H))`，clip=0.1。训练时用已采样前缀并行重算概率，未来动作被因果mask屏蔽。该目标不同于原MAT的逐区域clipping目标。
6. **数值约定**：优势按当前rollout标准化；value loss用该rollout回报标准差缩放，V的目标单位仍为收益/100。熵项为已采样前缀的条件熵之和，不声称精确枚举新策略全联合熵。每批最多5轮更新，joint KL目标0.02；KL过大只停止本批剩余更新，不停止总交互预算。

## 对照与统计

- 运营对照：F3（价格1.6，半径2 km）、LSTM-G、原始MA-IQL、冻结MA-IQL+。四者权重和仿真源文件留存指纹。
- 结构对照：`--factorized`仅关闭动作前缀输入，保留相同的时序/空间表示、网络结构、全城奖励、PPO设置及预算。它仍能通过全局状态协同；该对照具体检验显式当期动作条件化的增益。
- 主实验预先固定5个训练初始化81—85，每个4096条episode，即229376条计分城市级转移。六区域只算一次联合交互，不把一条转移计成六条。
- 固定30个验证种子监测训练，每16次策略迭代记录；固定100个新测试种子用于最终比较。网络与配置修改后须生成新协议。最终checkpoint取预定预算终点，不选择测试表现最佳的模型。
- 每个训练初始化使用同一批外生训练场景；训练、验证、测试使用互不相交的种子空间，并排除已记录的旧评估种子。主测试必须提交全部5个完整初始化。
- 配对条件是相同原始订单抽样、司机初态和预热。动作改变后仿真器消耗随机数的数量可能不同，因此不称为动作无关的完全配对未来随机流。策略随机数与仿真器随机数独立。
- 主评估使用随机祖先采样，与训练的期望收益目标一致；不把逐区域条件众数称为全局联合MAP。每个场景固定一个独立的策略随机种子。确定性解码若另做实验，应单独标记。
- 报告美元收益差、相对收益差、95%区间及各初始化结果。单模型对环境seed做配对bootstrap；多初始化使用训练初始化×环境seed的交叉重采样，不能把500次模拟当作500个独立训练结果。
- 其余五指标作为次要终点，不因某一指标变好而宣称总体最优。匹配成功率分母包含跨周期重复候选，等待只统计已匹配乘客；单日新种子检验不代表跨日期泛化。

## 分步实施及通过条件

| 阶段 | 模块 | 通过条件 |
|---|---|---|
| 1 | `protocol.py` | 数据/代码/权重指纹固定，种子隔离，指标与预算预先确定 |
| 2 | `environment.py` | F3复现已有评估；56个reward之和等于全窗口佣金；观察时序一致 |
| 3 | `policy.py` | 前缀因果性、联合概率归一化、采样与训练概率一致 |
| 4 | `ppo.py`、`rollout.py` | GAE终点正确、概率比正确、同批数据版本一致、有限梯度、断点可恢复 |
| 5 | `train.py` | 小试跑通采集—更新—保存—恢复，记录实际样本复用、KL及训练曲线 |
| 6 | `evaluate.py` | 冻结模型、新种子配对比较、多初始化统计；通过前五阶段后再开展主实验 |

小试只验证研究实现和计算成本，不据其少量收益比较判定方法有效。小试与主实验的输出分开保留。验证模拟单独计数，不混入训练交互预算。

## 运行

```bash
# 正确性验证，无需pytest。
python3 -m unittest discover -s tests -p test_online_rl.py -v

# 小试：16条episode、2次PPO迭代，可先提交1次迭代再恢复。
python3 -m online_rl.protocol --pilot --output experiments/online_ppo_pilot_20261104
python3 -m online_rl.train --protocol experiments/online_ppo_pilot_20261104/protocol.json \
  --output models/online_ppo_pilot_20261104 --training-seed 81 --workers 4 --max-updates 1
python3 -m online_rl.train --protocol experiments/online_ppo_pilot_20261104/protocol.json \
  --output models/online_ppo_pilot_20261104 --training-seed 81 --workers 4 --resume
python3 -m online_rl.evaluate --protocol experiments/online_ppo_pilot_20261104/protocol.json \
  --checkpoints models/online_ppo_pilot_20261104/policy.pt --split validation \
  --references fixed_f3 --workers 4 --output experiments/online_ppo_pilot_20261104/validation

# 主协议只确定研究预算，不自动启动大规模训练。
python3 -m online_rl.protocol --output experiments/online_ppo_main_20261104
python3 -m online_rl.train --protocol experiments/online_ppo_main_20261104/protocol.json \
  --output models/online_ppo_main_20261104/seed81 --training-seed 81 --workers 8
# 其余初始化82—85采用相同设置。结构对照添加--factorized并使用独立输出目录。
```

训练自动选择可用的MPS或CPU，仿真worker使用CPU。每次PPO迭代保存行为策略、联合转移和最新完整断点；在完整迭代边界恢复。中断期间未提交的迭代会重算，故恢复不意味着逐条仿真交互从中断点无重复继续。主实验资源报告还应计入这种重算开销。

`training_history.csv`记录城市级交互数、收益、损失、联合KL、熵、value解释方差、实际样本访问次数和有效更新轮数；`training_curves.*`为训练图。评估输出保留六指标、逐seed结果、配对收益差和区间。

参考：[MAT](https://arxiv.org/abs/2205.14953)、[PPO](https://arxiv.org/abs/1707.06347)、[GAE](https://arxiv.org/abs/1506.02438)。

## 当前完成情况（2026-11-04）

上述六个阶段的代码已实现，10项正确性检查通过，包括真实仿真中F3的完整复现。两种结构各完成16条轨迹、896条城市级转移的小试；每条转移实际参与5轮更新，AR-PPO已验证在第一轮断点后继续第二轮。保存的行为策略重算日志概率，最大误差约3.82×10⁻⁶。

另用4个预留验证种子完成六策略、24条episode的配对评估，输入、司机初态及预热指纹一致。该小试的AR-PPO尚未超过F3；AR-PPO与结构对照的收益差区间跨零。这里只能确认实现正确并可运行，不能据两次PPO迭代作收敛或方法有效性判断。

本轮记录位于`experiments/online_ppo_pilot_20261104/`，包含`verification_report.json`、`correctness_tests.log`和`validation/`；权重及训练图位于`models/online_ppo_pilot_20261104/`。

正式协议位于`experiments/online_ppo_main_20261104/protocol.json`。2026-11-04已启动AR-PPO正式训练：从随机权重开始，初始化81—85依次运行，每个4096条轨迹，使用MPS及8个CPU仿真worker。训练队列为`experiments/online_ppo_main_20261104/run_training.py`，总体进度见`models/online_ppo_main_20261104/progress.json`，逐初始化记录及完整断点位于其`seed81`—`seed85`子目录。正式测试尚未运行。

2026-11-04按用户要求延长训练预算，每个初始化由4096条增加至8192条完整episode，即512次更新、458752条计分城市级转移。新协议为`experiments/online_ppo_extended_20261104/protocol.json`，原协议和结果保留；算法、网络、超参数及验证/测试种子不变，在原训练场景序列后追加4096个独立种子。该预算调整发生于观察到验证收益仍在上升之后，不视为原始预注册预算。

扩展训练保留初始化81第256次、初始化82第72次更新的模型、优化器、随机数状态及全部历史，从完整迭代边界接续；其余初始化仍从原随机种子开始。迁移检查见`experiments/online_ppo_extended_20261104/extension_verification.json`，逐初始化来源见`models/online_ppo_extended_20261104/seed*/continuation.json`。原队列已停止，当前队列为`experiments/online_ppo_extended_20261104/run_training.py`，总体进度见`models/online_ppo_extended_20261104/progress.json`；早期rollout保留在原目录，新增rollout保存到新目录。最终仍取固定预算终点，不依验证曲线提前停止或挑选checkpoint。

随后按用户明确要求，累计预算调整为每个初始化10000条完整episode，即625次更新、560000条城市级转移。当前协议与队列位于`experiments/online_ppo_10000_20261104/`，输出位于`models/online_ppo_10000_20261104/`，之前两个队列均已停止。初始化81从第271次更新接续，初始化82保留第72次更新断点；网络、优化器、随机数状态及历史均完整保留，其余三个初始化按原种子从头训练。

当前目录下的`seed*/training_history.csv`与`validation_history.csv`覆盖自第一次更新起的全部记录，并持续追加；日志也继承前两个阶段。`training_curves.png/pdf/svg`启动时生成，之后每16次更新刷新，训练结束时再次保存。新的10000条预算保留原8192个训练场景并追加1808个独立场景，验证/测试场景不变。迁移与连续性检查见本轮`extension_verification.json`，总体进度为本轮`progress.json`。

最新预算按用户要求提高到每个初始化100000条完整episode，即6250次更新、5600000条城市级转移；五个初始化依次训练，总目标500000条episode。当前协议、迁移检查和队列位于`experiments/online_ppo_100000_20261104/`，输出及实时进度位于`models/online_ppo_100000_20261104/`。10000条预算的队列已停止，初始化81从第625次更新、初始化82从第74次更新接续，完整保留参数、优化器、随机数状态及自第一次更新起的全部记录。算法与超参数不变，原10000个训练场景后追加90000个独立场景，验证/测试种子不变，不早停。训练记录逐次追加，曲线每16次更新刷新；之前各阶段的文件保留以便追溯。
