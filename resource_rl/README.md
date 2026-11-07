# 资源流感知的循环图在线强化学习

独立于 `online_rl` 的新研究框架，采用随机权重初始化。训练目标仍为六区域05:20—10:00的累计平台收益，六组价格与半径每5分钟同时提交；20分钟共同预热不计入奖励目标。

## 方法

- **状态**：86维区域特征、9维有向边特征。除既有供需计数及上一周期24项指标外，加入订单年龄、锁定报价、接驾距离分布、预计5／10／20分钟释放的司机及目的地。仅访问当前任务、当前队列、静态地理与历史指标。
- **策略**：资源流图注意力、跨整轮保留的LSTM记忆、六区域因果动作解码。区域内价格和半径使用条件分布，保留全部11×10个候选组合。
- **价值**：独立编码器，额外读取当前司机／订单集合。每名司机24维、每条等待订单23维，不读取未来订单或未来随机抽样。集合注意力和历史记忆是函数近似，不声称构造出精确Markov状态。
- **辅助学习**：当前动作只进入资源预测头，预测下一周期空闲司机、等待订单以及实际完成服务的司机OD流；价值头不读取当前动作。辅助目标不改变经济奖励。
- **训练**：完整4步预热＋56步计分序列；每次更新按当前参数重算隐状态，只打乱episode。actor和critic使用独立优化器，全批联合KL监测可停止actor本批后续epoch，critic仍完成预定回归。

## 对照与边界

`flow` 为新结构；`window` 是同信息窗口式控制结构（Window-AC）：actor观测、critic个体集合、动作解码器、辅助目标及训练设置相同，以四周期窗口和空间注意力替代持续的资源流图递归。它不是已有AR-PPO权重的迁移版本，也不声称是原算法的完全复现。

两种结构各使用3个独立随机初始化，相同环境交互预算。F3（1.6、2 km）与原暂停AR-PPO快照作为额外运营参考。所有输入均为当期可获得的信息；训练侧个体集合不进入actor。平台采用CTCE。

当前协议为每次更新64条完整轨迹，每个初始化100000条，共600000条。最后一批为32条。每个初始化最多1563次更新。固定预算终点用于最终比较，不依验证收益挑checkpoint；100个新测试种子在所有运行完成前封存。配对保证订单样本、司机初态和预热一致，策略不同后仿真随机数消耗可能不同。

## 运行

在项目根目录执行：

```bash
python3 -m unittest tests.test_resource_rl -v
python3 -m resource_rl.check --output experiments/resource_rl_preflight_20261105 --device mps --workers 8

# 完整小试与恢复，权重不用于正式训练
python3 -m resource_rl.protocol --pilot --output experiments/resource_flow_pilot_20261105
python3 -m resource_rl.train --protocol experiments/resource_flow_pilot_20261105/protocol.json \
  --output models/resource_flow_pilot_20261105 --device mps --workers 8 --max-updates 1
python3 -m resource_rl.train --protocol experiments/resource_flow_pilot_20261105/protocol.json \
  --output models/resource_flow_pilot_20261105 --device mps --workers 8 --evaluate-at-end

# 正式研究：新随机初始化，不读取小试或原模型权重
python3 -m resource_rl.protocol --episodes 100000 --output experiments/resource_flow_main_20261105
python3 -m resource_rl.launch --protocol experiments/resource_flow_main_20261105/protocol.json \
  --output models/resource_flow_100k_20261105 --device mps --workers 8
```

同一命令可恢复本框架自己的完整断点。暂停时向 `progress.json` 中的训练PID发送SIGTERM，当前完整批次提交后退出。异常中断的未提交批次会重算。输出目录使用文件锁防止重复训练。

`progress.json` 显示当前采集／更新／验证阶段；`run.log` 和逐运行CSV每批更新。每次保存原始联合转移、行为策略以及含两个优化器和随机数状态的断点。等待订单集合以flat数组和长度编码保存，无object数组。训练图为600 DPI PNG和矢量PDF／SVG，定期更新。

`training_curve.*`为完整收益曲线，`training_diagnostics.*`包含价值／资源预测损失、KL、熵和实际样本复用次数。MC诊断使用采集时的V并按决策时刻计算解释方差；其误差含未来随机性。最终 `test/` 输出六指标、动作轨迹和跨训练初始化／环境种子的配对bootstrap区间。

关键实现依次为 `observation.py`、`environment.py`、`policy.py`、`ppo.py`、`train.py` 和 `evaluate.py`。

## 当前实施记录（2026-11-05）

25项新旧检查通过，包括因果性、跨窗口记忆、信息边界、F3分钟指标和周期指标的精确复现、两个优化器及随机数的恢复、封存测试的预算检查。完整小试在第一轮断点后恢复第二轮并完成闭环评估，结果仅用于正确性验证。

MPS检查中逐步采样与完整序列重算的最大对数概率误差约5.72×10⁻⁶，GPU分配约1.3 GB。区域图方法400086个参数，窗口控制428494个参数。驱动任务ETA按固定车速模型及当前接驾／送客阶段计算，不读取未来订单。

正式训练已按每个初始化100000条、两结构各3个种子的预算启动。协议和代码快照位于 `experiments/resource_flow_main_20261105/`，实时进度、完整断点及曲线位于 `models/resource_flow_100k_20261105/`。正式初始权重与随机初始化校验一致，未加载小试权重。验证记录位于 `experiments/resource_rl_checks_20261105/verification.json`，原在线训练及权重保持暂停且未改动。

参考：[递归Actor–Critic](https://arxiv.org/abs/2110.05038)、[历史—状态Critic](https://arxiv.org/abs/2105.11674)、[MAPPO训练研究](https://arxiv.org/abs/2103.01955)、[辅助学习](https://arxiv.org/abs/1611.05397)。这些文献支持设计动机，不代表本组合已被验证有效或具有单调改进保证。
