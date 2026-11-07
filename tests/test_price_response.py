import unittest

import numpy as np

from src.env.price_response import DriverOffer, PassengerOffer, UtilityChoiceModel


class UtilityChoiceTests(unittest.TestCase):
    def test_passenger_utility_uses_price_wait_and_trip_time(self):
        model = UtilityChoiceModel(passenger_price_coefficient_std=0.0)

        low_price = model.passenger_accept_probability(offer=PassengerOffer(
            quoted_fare=10.0,
            base_fare=10.0,
            expected_wait_time=0.0,
            trip_time=600.0,
        ))
        high_price = model.passenger_accept_probability(offer=PassengerOffer(
            quoted_fare=13.0,
            base_fare=10.0,
            expected_wait_time=0.0,
            trip_time=600.0,
        ))
        long_wait = model.passenger_accept_probability(offer=PassengerOffer(
            quoted_fare=10.0,
            base_fare=10.0,
            expected_wait_time=600.0,
            trip_time=600.0,
        ))
        long_trip = model.passenger_accept_probability(offer=PassengerOffer(
            quoted_fare=10.0,
            base_fare=10.0,
            expected_wait_time=0.0,
            trip_time=1800.0,
        ))

        self.assertGreater(low_price, high_price)
        self.assertGreater(low_price, long_wait)
        self.assertGreater(low_price, long_trip)

    def test_driver_utility_uses_absolute_payment_costs_and_pickup(self):
        model = UtilityChoiceModel(driver_order_opportunity_cost_std=0.0)
        profile = {"driver_order_opportunity_cost": np.array([0.0, 0.0])}

        probability = model.driver_accept_probability(
            offer=DriverOffer(
                driver_payment=np.array([3.9, 13.0]),
                reference_payment=np.array([3.0, 10.0]),
                pickup_distance=np.array([1.0, 1.0]),
                trip_distance=np.array([1.0, 1.0]),
            ),
            profile=profile,
        )
        farther_pickup = model.driver_accept_probability(
            offer=DriverOffer(
                driver_payment=13.0,
                reference_payment=10.0,
                pickup_distance=2.0,
                trip_distance=1.0,
            ),
            profile={"driver_order_opportunity_cost": 0.0},
        )

        self.assertGreater(probability[1], probability[0])
        self.assertGreater(probability[1], farther_pickup)


if __name__ == "__main__":
    unittest.main()
