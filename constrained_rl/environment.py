"""独立记录需求覆盖和全队列等待，不改变原版仿真转移。"""

import numpy as np
import pandas as pd

from online_rl.environment import DecisionEnvironment


SIGNALS = ("cohort_demand", "unique_matched", "physical_matched_orders", "queue_wait_seconds")
PARENT_COLUMN = "_constraint_baseline_order_id"


def order_ids(frame):
    if not len(frame):
        return []
    values = frame["order_id"].tolist()
    if any(pd.isna(value) for value in values) or len(set(values)) != len(values):
        raise ValueError
    return values


class CohortLedger:
    """预热积压按实际ID计入；新增复制单共享一个基准需求的服务信用。"""

    def __init__(self, backlog):
        carry = order_ids(backlog)
        self.cohort = set(carry)
        self.parent = {value: value for value in carry}
        self.covered = set()
        self.matched_orders = set()
        self.wait_seconds = 0.
        self.carry_in_demand = len(carry)
        self.baseline_arrivals = 0
        self.end_queue_orders = len(carry)

    def register_lineage(self, frame):
        for order, parent in zip(order_ids(frame), frame[PARENT_COLUMN].tolist()):
            if pd.isna(parent) or order in self.parent:
                raise ValueError
            self.parent[order] = parent

    def observe_minute(self, baseline, matched, queue, seconds):
        """按本分钟新生成事件计分；取消前已累计的等待不会被撤销。"""
        arrivals = order_ids(baseline)
        if self.cohort.intersection(arrivals):
            raise ValueError
        self.cohort.update(arrivals)
        self.baseline_arrivals += len(arrivals)
        for value in arrivals:
            self.parent.setdefault(value, value)
        for value in order_ids(matched):
            parent = self.parent.get(value)
            if parent not in self.cohort or value in self.matched_orders:
                raise ValueError
            self.matched_orders.add(value)
            self.covered.add(parent)
        queued = order_ids(queue)
        if (not np.isfinite(seconds) or seconds <= 0 or self.matched_orders.intersection(queued)
                or any(self.parent.get(value) not in self.cohort for value in queued)):
            raise ValueError
        # 原仿真在派单/取消/新增之后给剩余队列统一增加 delta_t 等待。
        # 此处精确累计这段新增时间，包含末分钟订单与最终未服务订单。
        self.wait_seconds += len(queued) * float(seconds)
        self.end_queue_orders = len(queued)

    def totals(self):
        return {"cohort_demand": len(self.cohort), "unique_matched": len(self.covered),
                "physical_matched_orders": len(self.matched_orders), "queue_wait_seconds": self.wait_seconds}

    def summary(self):
        count = len(self.cohort)
        coverage = len(self.covered) / count if count else 0.
        return {**self.totals(), "demand_satisfaction": coverage,
                "baseline_demand_coverage": coverage,
                "wait_per_demand_seconds": self.wait_seconds / count if count else 0.,
                "carry_in_demand": self.carry_in_demand, "scored_baseline_arrivals": self.baseline_arrivals,
                "end_unserved": count - len(self.covered), "end_queue_orders": self.end_queue_orders}


class ConstrainedEnvironment(DecisionEnvironment):
    """保留原三元组step接口，通过last_signals暴露本周期新增统计。"""

    def reset(self, seed):
        self.ledger = None
        self.last_signals = {"revenue": 0., **dict.fromkeys(SIGNALS, 0.)}
        self.signal_history = []
        observation = super().reset(seed)
        self.ledger = CohortLedger(self.sim.wait_requests)
        self._unreported_carry = self.ledger.carry_in_demand
        self._attach_tracking()
        return observation

    def _attach_tracking(self):
        simulator, ledger = self.sim, self.ledger
        resample = simulator._resample_orders_by_demand_factor
        original_step = simulator.step

        def tracked_resample(orders, factor):
            if PARENT_COLUMN in orders:
                raise ValueError
            # 临时列只穿过既有重采样函数；移除后表结构和RNG与原版完全相同。
            annotated = orders.assign(**{PARENT_COLUMN: orders["order_id"].to_numpy()})
            adjusted = resample(annotated, factor)
            ledger.register_lineage(adjusted)
            return adjusted.drop(columns=[PARENT_COLUMN])

        def tracked_step():
            previous_matches = len(simulator.matched_requests)
            original_step()
            # _step_events属于上一分钟新生订单；pending才是此刻实际生成的订单。
            # 读取pending也使最后一分钟生成、来不及派单的需求进入分母。
            pending = simulator._pending_request_events
            baseline = pending[pending["event"] == "request_baseline"]
            matched = simulator.matched_requests.iloc[previous_matches:]
            ledger.observe_minute(baseline, matched, simulator.wait_requests, simulator.delta_t)

        simulator._resample_orders_by_demand_factor = tracked_resample
        simulator.step = tracked_step

    def _advance(self, actions):
        if self.ledger is None:
            return super()._advance(actions)
        before = self.ledger.totals()
        reward = super()._advance(actions)
        after = self.ledger.totals()
        self.last_signals = {"revenue": reward, **{name: after[name] - before[name] for name in SIGNALS}}
        # 积压只加入第一个计分决策，故逐步分母之和等于整轮cohort大小。
        self.last_signals["cohort_demand"] += self._unreported_carry
        self._unreported_carry = 0
        self.signal_history.append(dict(self.last_signals))
        return reward

    def summary(self):
        result = super().summary()
        totals = self.ledger.summary()
        for name in SIGNALS:
            np.testing.assert_allclose(sum(row[name] for row in self.signal_history), totals[name], rtol=0, atol=1e-8)
        np.testing.assert_allclose(totals["physical_matched_orders"], result["matched_request_num"], rtol=0, atol=1e-8)
        return {**result, **totals}
