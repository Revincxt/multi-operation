# 数据准备与指标口径

## 仓库内容与输入数据

源码仓库包含运行入口、仿真器、价格响应模块、数据管线、诊断脚本、测试、依赖说明与展示资源。

`my_data/` 中的历史输入、`trajectories*/` 中的原始轨迹、`datasets*/` 中的训练集和 `experiments/` 中的诊断结果保留在本地。它们默认被 `.gitignore` 排除，运行管线会重新生成后三类输出。

**输入数据本身不能从代码重新生成。** 在另一台电脑执行真实采集前，必须通过独立的数据传输方式取得对应文件。仅上传源码不等于已备份原始数据。若需要把输入也保存在 GitHub，可单独配置 Git LFS 或数据附件；当前仓库尚未配置这项流程。

若维护者提供 `simulation-inputs.zip`，在项目根目录解压：

```bash
python3 -m zipfile -e /path/to/simulation-inputs.zip .
```

这个数据包保留 `my_data/` 路径，包含当前日期运行所需的订单、司机、订单网格映射、邻接矩阵、网格坐标和 `new_grids_263.csv`。它不包含已生成的训练集或其他日期的数据。

## 六网格输入

`run_data_pipeline.py` 默认使用 `my_data/grid6/`：在原 263 个六边形覆盖区域内重新铺设较大的六边形，再按原区域边界裁切，得到 6 个网格（ID 为 0–5），不改成矩形分区。完整六边形的中心到顶点约 4 km，裁切后的区域面积不同。划分预览保存在 `my_data/grid6/grid_preview.png`。

首次准备时，在已有原始输入的基础上执行；已生成该目录则无需重复运行：

```bash
python3 -m pip install -r requirements-grid.txt
python3 -m data_pipeline.regrid_hex --grid-count 6 --output my_data/grid6
```

重划分仅修改网格归属。2015-05-18 的全天 333,332 条订单和 1,000 名司机全部保留，经纬度、时间和行程数据不变；旧文件不覆盖。原输入中少量位于覆盖边界外的坐标仍按最近六边形中心归属，并在 `grid_manifest.json` 中计数。

新目录包含 `orders_grid6_2015-05-18.pkl`、`drivers.pickle`、`grid_info.csv`、`grid_adjacency.csv`、网格 GeoJSON 和网格清单。采集器直接使用已重映射的起终点，不再读取旧的 263 网格映射。几何依赖只用于重划分和绘图，日常采集不需要。

```bash
python3 run_data_pipeline.py

# 沿用原 263 网格时，输入和输出目录一起指定。
python3 run_data_pipeline.py --grid-num 263 --data-dir my_data \
  --raw-output trajectories_263 --dataset-output datasets_263
```

`run_experiment.py`、单轮和普通批量采集模块仍保留原 263 网格默认设置；`run_data_pipeline.py` 与 `data_pipeline.collect_density` 使用 6 网格。不要混合两种空间划分的轨迹。当前已完成的 LSTM 对应六网格；当前没有有效 TimesNet 权重。

### 当前保留的数据与模型（2026-11-02）

| 内容 | 路径 | 规模 |
| --- | --- | --- |
| 低密度原始轨迹 | `trajectories_grid6_low_mixed_20261027` | 混合采样，允许不同网格采用相同动作 |
| 最新监督数据 | `datasets_grid6_low_mixed47k_20261029` | 前 47,000 次完整轨迹，15,792,000 条样本 |
| 已完成 LSTM | `models/lstm_grid6_low_mixed47k_mps100_20261029` | 训练 100 轮，验证最优第 84 轮 |
| 中、高密度轨迹 | `trajectories_grid6_medium`、`trajectories_grid6_high` | 独立场景，保留后续使用 |

当前数据沿用采集时固定的 run 级划分：训练 11,089,344 条、验证 2,334,528 条、测试 2,368,128 条。模型使用四个历史周期和 28 维当前上下文，包括全网格动作与全城状态。旧版低密度轨迹、旧监督数据集及被替代的模型已经清理；旧实验图表和汇总统计保留，删除明细见 `experiments/project_cleanup_20261102/manifest.json`。旧实验记录中的输入路径用于说明历史来源，部分已不能重新运行。

强化学习复用 `src/env` 的仿真器和决策前状态。当前 `offline_rl/` 实现六智能体双头 IQL 与 BC，直接将已有缓存构成城市级转移，不重读监督 CSV；在线验证和测试使用公共策略回调。方法、数据边界和命令见 [离线 RL 说明](../offline_rl/README.md)。

## 原始输入清单

以现有 263 网格、2015-05-18 配置为例，文件放在项目根目录的 `my_data/` 中：

| 文件 | 用途 | 哪些任务需要 |
| --- | --- | --- |
| `orders_grid35_2015-05-18.pkl` | 按时间索引的历史订单 | 所有真实数据仿真 |
| `drivers_grid35_1000.pickle` | 司机初始位置与排班 | 所有真实数据仿真 |
| `orders_grid35_2015-05-18-map263.csv` | 订单起终点的 263 网格映射 | 原 263 网格采集 |
| `hexo_updated_r300_adj.csv` | 网格邻接矩阵 | 邻区状态采集、Treatment / Control 空间实验与空车调度 |
| `hex_updated_r300_info.csv` | 网格中心坐标与几何信息 | 空车调度 |

`new_grids_263.csv`、其他网格划分和节点映射属于区域重构或数据预处理资料。可与上述文件一并保留，但当前默认采集入口不直接读取所有这些文件。

历史文件名中的 `grid35` 是既有数据命名。263 网格订单通过单独的映射表转换；不能根据文件名推断实际输出网格数量。

原 263 网格添加日期时，需要同时准备 `orders_grid35_{date}.pkl` 与 `orders_grid35_{date}-map263.csv`，日期必须与 `--dates` 一致。六网格输入则使用重划分脚本的 `--dates` 指定全部日期，并输出到新目录；日期之间共用同一套空间划分。数据加载器使用约定命名，不支持随意替换文件名。

可用 `--data-dir` 指定目录。批量采集会先检查所有日期的订单、司机和必要映射文件；不会复制某一天的订单冒充新日期。`--start-times 05:00,12:00,18:00` 表示每天分别从这些时刻独立开始，时长由 `--simulation-steps` 指定。订单时间索引必须覆盖完整窗口，且窗口不能跨午夜。各次仿真复用司机初始位置，并不还原该时刻真实的司机分布。

## 输出目录

```text
trajectories_grid6/
├── batch_manifest.json
└── {date}_simseed{seed}_actionseed{action_seed}_grid6_5min_utility_choice_start0500_steps300/
    ├── manifest.json
    ├── action_plan.csv
    └── grid_minute_metrics.csv

datasets_grid6/
├── dataset_manifest.json
├── supervised_dataset.csv
└── validation_report.json
```

- `action_plan.csv`：每个决策周期、每个网格的价格、半径、联合动作 ID，以及动作执行前的本区/邻区状态快照。
- `grid_minute_metrics.csv`：每分钟的行为计数、网格状态、实际动作及计划动作。
- `manifest.json`：单次仿真的参数、起止时间、输入目录、动作候选集、记录数与概要指标。
- `batch_manifest.json`：整批任务全部完成后写出的运行清单。
- `dataset_manifest.json`：特征列、标签列、历史窗口、划分依据，以及各集合的日期与 run 数。
- `supervised_dataset.csv`：一行对应一个 run 中某网格某决策周期的训练样本。
- `validation_report.json`：结构、范围、时间连续性和动作覆盖检查结果。

默认 300 分钟 / 5 分钟得到 60 个决策周期，前 4 个周期仅作历史。六网格每个 run 得到 `(60 − 4) × 6 = 336` 条监督样本，10 个种子合计 3,360 条；原 263 网格为每个 run 14,728 条。历史窗口不跨网格、run 或时段，不随机混合相邻行。旧 run 名称仍可读取。

动作候选为 11 个价格 × 10 个半径，共 110 种，具体取值统一在 `src/env/actions.py` 中定义。2026-10-27 起，新采集采用 `joint_action_mixture_v1`，允许不同网格采用同一组合，不改变候选值或联合动作 ID。每个 run 独立抽取一种模式：

- 约 60% 为优势分段：价格 1.4–1.8、半径 1.5–2.5 km，在可用候选组合内按网格有放回抽样；每三个决策周期重抽，即默认保持 15 分钟。
- 约 20% 为全域随机：所有候选组合按网格有放回抽样，每个决策周期重抽。
- 约 20% 为持续参考：整次仿真所有网格采用同一个参考组合；默认一半参考 run 使用 F3（价格 1.6、半径 2 km），其余覆盖其他既有参考。

这些比例是 run 级期望比例，不是每个周期的强制配额。自定义候选集合缺少优势或参考动作时，相应比例回退到全域随机。优势区间来自低密度实验，不代表其他密度下都最优。

`manifest.json` 记录 `action_sampling`、`sampling_mode` 和参考动作；批次配置记录采样版本，拒绝在旧批次中混入新方式。新版校验器检查候选支持集、网格覆盖、主键、指标和时序，重复联合动作合法；动作覆盖不足仅提示，不再强制每周期均衡。旧数据没有采样元数据时仍按原平衡采样规则校验。旧数据与模型保留，新采集写入独立目录。

`--split-by auto` 对多日期数据按日期先后划分，较早日期训练，随后验证，最后测试；同一天的所有种子和时段保持在同一集合。单日期自动使用原有 run 随机划分并提示不能用于跨日期泛化。可以用 `--split-by date` 强制日期隔离；默认验证/测试比例下至少需要 3 个日期，不足时直接报错。校验器和 LSTM 训练入口都会检查 date 模式是否存在日期交叉。

## 聚合规则

新采集额外记录八个当前状态特征，采样时点为决策边界、执行 `sim.step()` 之前（在取消、匹配、新订单生成之前）：

| 本区字段 | 含义 |
| --- | --- |
| `start_waiting_request_num` | 队列中尚未匹配的订单数，按订单起点网格归属，包含本轮尚未执行超时取消的订单 |
| `start_idle_driver_num` | 当前网格空闲司机数，status=0 |
| `start_busy_driver_num` | 当前网格接驾或送客司机数，status=1/2 |
| `start_repositioning_driver_num` | 当前网格调度中司机数，status=4；关闭调度时为 0 |

另外四列为上述字段加 `neighbor_` 前缀，按固定六边形邻接矩阵对一阶邻区求和，不包含本区、二阶邻区或邻区动作。矩阵及其摘要保存在 manifest，6 网格使用 `grid_adjacency.csv`。

这八列直接作为当前输入，不用周期平均值替代，也不作为预测标签。历史司机数仍按原有周期聚合保留。LSTM/TimesNet 的当前输入由 4 维扩展为 12 维，六个预测目标不变。旧数据缺少决策前快照，无法仅靠旧分钟表精确补齐，不能填零伪装为新数据；新旧格式分开构建，旧数据和 checkpoint 仍按原格式读取。

当前构建器为 [build_supervised_dataset.py](../data_pipeline/build_supervised_dataset.py)。

| 字段类型 | 周期聚合方法 |
| --- | --- |
| 基准请求、实际提交、匹配、司机报价/接受、取消数量 | 分钟计数求和 |
| GBV、平台收益 | 分钟金额求和 |
| 候选请求池、空车、忙碌车辆、调度车辆、车辆利用率 | 分钟值取均值 |
| 等待时间、接驾时间、接驾距离 | 按各分钟成功匹配数加权平均 |
| 呼出率、司机接单率等比例 | 先聚合分子分母，再重新计算 |

在监督表中，分母为 0 的比例和没有成功匹配时的等待时间统一记为 0。训练模型时应结合请求数、报价数和匹配数区分“没有观测”与“实际观测为零”。

原始分钟记录中的比例可包含空值。例如没有基准请求时呼出率不可定义；部分 pandas 对齐产生的空值也会在计数聚合后重新计算。直接训练原始 CSV 时不能简单把所有空值解释为数据损坏，也不能不加处理地输入模型。

## 六个研究目标

当前候选标签：

```python
target_columns = [
    "target_candidate_request_num",
    "target_request_rate",
    "target_driver_accept_rate",
    "target_order_completion_rate",
    "target_mean_wait_to_match_seconds",
    "target_platform_revenue",
]
```

需要区分的含义：

1. `candidate_request_num` 为每分钟超时取消后参与匹配的候选池数量，包含积压；监督标签保存周期均值。若“冒泡”指发单前新增需求，应使用周期求和的 `baseline_request_num`。
2. `request_rate = submitted_request_num / baseline_request_num` 是基准需求数量比，不是单个乘客的概率；需求扩增可使其大于 1。
3. `driver_accept_rate = driver_accept_num / driver_offer_num`，不包含乘客后续是否接受。
4. `order_completion_rate = matched_request_num / (sum(minute_candidate_request_num) + unmatched_cancel_num)`。分子是双方接受后的匹配数，不是车辆到达终点的订单数；分母包含跨分钟重复等待的订单。因此不能把它当作唯一订单的最终送达率。
5. `mean_wait_to_match_seconds` 只统计成功匹配订单等到匹配的时间，不含接驾时间，也不代表取消乘客的等待体验。
6. `platform_revenue` 是成功匹配订单的车费乘抽成率，默认抽成 `0.25`；`gbv` 是车费总额，两者不能混用。收入在匹配接受时记账。

## 时间与动作

`lag_1_*` 表示上一决策周期，`lag_4_*` 表示前四个周期；序列模型通常需要按 `lag_4 → lag_1` 还原时间顺序。

价格乘子作用于新请求报价，积压订单保留原报价。半径应用于本轮匹配，并按订单起点网格读取当前动作。当前周期标签包含历史订单与历史动作的影响，不能解释为一次独立的、全量订单即时重定价实验。

当前 `grid_id` 是原始标识；数据管线没有训练 embedding。训练时可使用网格类别编码或 embedding，不应把 ID 的数值距离直接当作空间距离。

## 数据使用检查

- 将不同行为模型、参数版本的运行结果存放到独立目录；当前构建器并不验证全部仿真配置是否相同。
- schema 名称、文件格式兼容不代表业务假设相同；保留每个 run 的 manifest。
- `train / validation / test` 根据 `split_by` 按日期或完整 run 划分。同一日期不同 seed 或不同时段仍不是独立日期；单日期报告会给出警告。
- 全局动作覆盖充分不代表每个“网格 × 动作 × 时段”都有充足样本。
- 结构校验通过不代表行为参数已完成现实校准或预测模型已经有效。
