<div align="center">

<img src="docs/assets/banner.svg" alt="Grid Mobility Lab" width="100%">

# Grid Mobility Lab

网格级网约车仿真 · 价格与匹配半径决策 · 策略评估

</div>

每五分钟为六个网格分别选择价格乘子和匹配半径，仿真乘客发单、匹配拒绝、等待取消及司机状态变化，记录六项运营指标。

现有方法包括 LSTM / TimesNet 监督预测、贪心与分块坐标搜索（BCS），以及六智能体离线 IQL。IQL 的各区域策略分别输出价格和匹配半径，使用集中式价值网络训练。方法与评估流程见 [离线强化学习说明](offline_rl/README.md)。

## 环境与动作

| 设置 | 当前配置 |
| --- | --- |
| 研究区域 | 原 263 网格覆盖区域，重划分为 6 个六边形网格 |
| 日期 / 时段 | 2015-05-18，05:00–10:00 |
| 仿真步 / 决策周期 | 1 分钟 / 5 分钟 |
| 低 / 中 / 高密度 | 约 5,000 / 10,000 / 25,000 个任务，250 / 500 / 1,000 名司机 |
| 价格乘子 | 0.4, 1.0, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 4.0 |
| 匹配半径（km） | 0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0 |
| 每网格候选组合 | 110 种；不同网格允许采用同一动作 |
| 价格响应 / 空车调度 | Utility Choice / 默认关闭 |

价格乘子作用于新订单报价，积压订单保留发单价格；匹配半径按当前周期和订单起点网格使用。匹配采用现有 LD 迭代算法。

## 安装与检查

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

模型与绘图测试还需安装 `baseline/requirements.txt` 和 `requirements-grid.txt`。真实仿真需要本地 `my_data/grid6/` 输入；文件约定见 [数据说明](docs/data_guide.md)，行为模型见 [价格响应说明](docs/price_response_models.md)。

## 运行

```bash
# 短流程：采集、构建监督样本、校验
python3 run_data_pipeline.py --seeds 42,43,44 --simulation-steps 30 \
  --raw-output trajectories_quickstart --dataset-output datasets_quickstart

# 训练新的监督学习对照模型，使用当前数据集
python3 -m pip install -r baseline/requirements.txt
python3 -m baseline.train_lstm --epochs 100 --output models/lstm_new

# 最新 LSTM 与固定策略、随机策略进行闭环比较
python3 -m baseline.compare_lstm_fixed --include-random --runs 10 --workers 4

# 同一 LSTM 的单轮贪心与 BCS 闭环比较；固定参考为 F3
python3 -m baseline.compare_bcs --dataset datasets_grid6_low_mixed47k_20261029 \
  --checkpoint models/lstm_grid6_low_mixed47k_mps100_20261029/best_model.pt \
  --fixed-price 1.6 --fixed-radius 2 --runs 10

# 离线 RL：首次构建转移，再训练、验证选模型和测试
python3 -m offline_rl.data
python3 -m offline_rl.train --device auto --steps 20000 --workers 4
```

完整参数见各入口的 `--help`。训练和评估会写入新目录；已有结果不会自动覆盖。监督学习方法、WESM 与 TimesNet 适配见 [baseline/README.md](baseline/README.md)。

## 当前保留的数据与模型

| 内容 | 路径 |
| --- | --- |
| 最新低密度原始轨迹 | `trajectories_grid6_low_mixed_20261027/` |
| 前 47,000 次完整轨迹构建的数据集 | `datasets_grid6_low_mixed47k_20261029/` |
| 最新 LSTM 权重 | `models/lstm_grid6_low_mixed47k_mps100_20261029/best_model.pt` |
| 六智能体 IQL 权重 | `models/iql_grid6_low_20261102/best_iql9.pt` |
| 离线 RL 测试与曲线 | `models/iql_grid6_low_20261102/test/`、`training_curves.png` |
| 中、高密度原始轨迹 | `trajectories_grid6_medium/`、`trajectories_grid6_high/` |
| 最新闭环对比 | `experiments/lstm_mixed47k_bcs_retest_20261029/` |
| 完整测试集上的搜索对比 | `experiments/lstm_mixed47k_testset_search_20261029/` |

当前监督数据共 15,792,000 条，按完整轨迹划分：训练 11,089,344 条、验证 2,334,528 条、测试 2,368,128 条。LSTM 已训练 100 轮，验证最优为第 84 轮。六网格共享模型，使用四步历史、网格 embedding 和 28 维当前上下文。

旧版低密度数据集、中断训练及被替代的权重已清理。历史图表、汇总统计和必要训练记录保留；清理明细见 `experiments/project_cleanup_20261102/manifest.json`。历史结果中的旧输入路径仅表示当时来源，部分实验已无法完整重跑。

## 指标

| 指标 | 口径 |
| --- | --- |
| 冒泡请求数量 | 价格响应前的基准新增需求 |
| 订单呼出率 | 提交订单数 / 基准新增需求 |
| 司机接单率 | 司机接受数 / 司机报价数 |
| 订单完成率 | 最终接受匹配数 / 取消前候选池累计数量 |
| 乘客平均等待时间 | 成功匹配乘客的排队等待时间，不含接驾 |
| 平台收益 | 成功匹配订单的平台佣金收入 |

积压订单可能多次进入完成率分母，因此该指标不是按唯一订单计算的送达比例。完整测试集搜索对比报告的是模型预测收益；真实运营收益以闭环仿真结果为准。现有数据只有一个日期。

## 项目结构

```text
src/env/             仿真器、网格动作、决策前状态、价格响应
src/utils/           匹配算法、距离与联合动作上下文
run_data_pipeline.py 监督数据管线入口
data_pipeline/       采集、构建、校验与六边形重划分
baseline/            LSTM、TimesNet、训练与策略评估
offline_rl/          六智能体双头 IQL、BC 与真实闭环评估
diagnostics/         行为诊断、动作扫描与论文图表
run_experiment.py    可选 Treatment / Control 诊断实验
tests/               单元及集成测试
docs/                当前数据与模型说明
my_data/             原始仿真输入，本地保留
```

数据、模型和实验输出默认由 `.gitignore` 排除。
