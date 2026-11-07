# 价格响应模型

本页是当前实现的统一说明。模型只计算行为概率，不负责选择价格和半径；运营动作由随机采集器或预测策略提供。参数是研究假设，未经现实市场数据标定。

## 当前采集使用的模型

低密度六网格轨迹使用 `utility_choice`。每个网格每个决策周期对应一个 `GridAction(price_multiplier, matching_radius_km)`。

| 阶段 | 当前调用 | 作用 |
| --- | --- | --- |
| 乘客发单 | `passenger_accept_probability` | 根据报价、预计等待和行程时间计算发单概率 |
| 未匹配取消 | `passenger_cancel_unmatched_probability` | Utility 默认在等待达到 600 秒时取消 |
| 司机接单 | `driver_accept_probability` | 根据净收入相对参考净收入的变化和接驾距离计算概率 |
| 接驾阶段取消 | `passenger_cancel_pickup_probability` | 司机接受后，根据预计接驾时间和乘客距离容忍度计算取消概率 |

司机拒绝的订单返回等待队列；乘客取消的订单移除。被拒绝或取消的匹配不会让司机进入忙碌状态。

价格应用于新订单，积压订单保留提交时价格；匹配半径采用当前周期订单起点网格的动作。匹配处理已有队列，新请求随后进入队列，仿真时序不因本次接口整理而改变。

### Utility 公式

乘客效用：

```text
price_premium = quoted_fare / base_fare - 1
utility = intercept - price_coefficient * price_premium
          - wait_coefficient * wait_minutes - trip_time_coefficient * trip_minutes
P(request) = sigmoid(utility)
```

司机效用：

```text
reference_net = reference_payment - operating_cost_per_km * trip_distance
net_payment = driver_payment - operating_cost_per_km * (pickup_distance + trip_distance)
              - opportunity_cost
income_premium = net_payment / max(reference_net, 1e-9) - 1
utility = intercept + income_coefficient * income_premium
          - pickup_coefficient * pickup_distance
P(accept) = sigmoid(utility)
```

这是相对净收入模型，不是只使用价格倍率，也不是直接将绝对净收入代入 sigmoid。乘客价格敏感度、接驾距离容忍度和司机机会成本具有个体异质性。

## 需求与供给口径

- 默认 `demand_input_mode="observed_requests"`：输入是已经观察到的历史订单，按当前价格与参考价格的发单概率之比重采样。参考倍率为 1；低价可扩增需求，所以呼出率可能大于 1。
- `demand_input_mode="potential"` 才把输入解释为未经筛选的潜在需求。
- 默认基础车费为 `3 + 0.7 × trip_distance_km`，司机所得为报价的 75%，平台收入为 25%。
- 每轮抽取的司机全程在线。独立模型中的 `driver_online_probability` 不控制仿真司机上下线。
- 空车调度由 `enable_reposition` 控制，采集默认关闭；开启时会调用司机模型的 `driver_reposition_probabilities`。

## 支持的参数与示例

```python
from src.env import uniform_grid_actions
from src.env.simulator_env import Simulator

simulator = Simulator(
    grid_num=6,
    driver_num=250,
    decision_freq=5,
    price_response_model="utility_choice",
    grid_action=uniform_grid_actions(6, 1.5, 3.0),
    unmatched_cancellation=True,
    pickup_cancellation=True,
    enable_reposition=False,
)
```

以上构造环境；实际运行还需加载与六网格一致的订单和司机输入，完整采集入口见[项目说明](../README.md)。

`grid_action` 也支持 `{grid_id: GridAction}` 或返回完整映射的 `(simulation_time, grid_state)` 回调。所有网格必须有动作。候选值见 [actions.py](../src/env/actions.py)；自定义候选需同时传入 `allowed_price_multipliers` 和 `allowed_matching_radii_km`。

配置可使用类型化 `price_response_config` 或字典 `price_response_parameters`，不能同时提供。乘客和司机也可分别指定模型与配置。

以下旧开关已移除，传入会抛出 `TypeError`：

| 旧参数 | 当前做法 |
| --- | --- |
| `post_match_cancellation` | 使用 `pickup_cancellation` |
| `driver_supply_response` | 当前为整轮固定在线司机，没有替代的价格敏感上线开关 |
| `repo_mode` | 使用 `enable_reposition` 及对应调度配置 |

## 另外两种模型的范围

`aggregate_elasticity`（别名 `aggregate/huang`）与 `bounded_rational_agent`（别名 `bounded_rational/agent/xie`）保留独立概率计算接口，分别采用聚合弹性、保留价格与目标收入机制。

**它们尚未实现当前仿真器调用的两阶段取消接口，不能在保留默认取消开关时直接替换 Utility。**其旧 `passenger_cancel_probability` 不等于当前的两阶段接口。如关闭两项取消后单独测试，需明确该实验不包含相同的乘客取消机制，不能视为等条件对比。

独立使用从 `src.env.price_response` 导入 `PassengerOffer`、`DriverOffer`、配置类和 `create_price_response_model`。该包是公共接口，原 `price_response_models.py` 兼容文件已移除。
