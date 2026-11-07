import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.env.actions import GridAction
from src.env.simulator_env import Simulator


class OrderLifecycleTests(unittest.TestCase):
    def test_expired_orders_are_removed_before_dispatch(self):
        sim = Simulator(
            grid_num=1,
            driver_num=1,
            grid_action={0: GridAction(1.0, 1.0)},
            unmatched_cancellation=False,
        )
        driver_info = pd.DataFrame({
            "driver_id": [1],
            "start_time": [18000],
            "end_time": [36000],
            "lng": [-73.99],
            "lat": [40.75],
            "grid_id": [0],
        })
        sim.reset(
            seed=42,
            given_data=True,
            request_databases={},
            driver_info=driver_info,
        )
        sim.finish_run_step = 1
        sim.wait_requests = pd.DataFrame({
            "order_id": [10, 20],
            "origin_grid_id": [0, 0],
            "wait_time": [601.0, 600.0],
            "maximum_wait_time": [600.0, 600.0],
        })

        with patch(
            "src.env.simulator_env.order_dispatch",
            return_value=([], np.array([])),
        ) as dispatch:
            sim.step()

        dispatched_orders = dispatch.call_args.args[0]
        self.assertEqual(dispatched_orders["order_id"].tolist(), [20])
        cancelled = sim._step_events[
            sim._step_events["event"] == "unmatched_cancel"]
        self.assertEqual(cancelled["order_id"].tolist(), [10])
        self.assertEqual(sim.passenger_cancelled_requests_num, 1)


if __name__ == "__main__":
    unittest.main()
