"""Passenger utility and driver opportunity-cost response model."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from .base import (
    ArrayLike,
    DriverMarket,
    DriverOffer,
    PassengerMatch,
    PassengerOffer,
    PriceResponseModel,
    Profile,
    as_float_array,
    coerce_driver_market,
    coerce_driver_offer,
    coerce_passenger_match,
    coerce_passenger_offer,
    combine_profile,
    return_scalar_if_scalar,
    safe_positive,
    sigmoid,
)


@dataclass(frozen=True)
class UtilityChoiceConfig:
    # Stage 1: passenger utility from price, expected wait and trip duration.
    passenger_intercept: float = 2.0
    passenger_price_coefficient: float = 2.0
    passenger_price_coefficient_std: float = 0.15
    passenger_wait_coefficient: float = 0.12
    passenger_trip_time_coefficient: float = 0.015

    # Stage 2: driver utility from net income premium and pickup distance.
    driver_intercept: float = 2.2
    driver_income_coefficient: float = 2.0
    driver_pickup_coefficient: float = 0.7
    driver_order_opportunity_cost_mean: float = 0.0
    driver_order_opportunity_cost_std: float = 0.15
    operating_cost_per_km: float = 0.5

    # Stage 3: submitted but unmatched passenger cancellation. The optional hard
    # threshold makes a passenger cancel deterministically after ten minutes.
    unmatched_cancel_intercept: float = -5.0
    unmatched_cancel_wait_coefficient: float = 0.8
    unmatched_cancel_threshold_seconds: Optional[float] = 600.0

    # Stage 4: matched passenger cancellation while awaiting pickup. Tolerance is
    # heterogeneous (normal, truncated at zero) and expressed in km.
    pickup_cancel_intercept: float = -5.0
    pickup_cancel_time_coefficient: float = 0.8
    pickup_tolerance_km_mean: float = 2.0
    pickup_tolerance_km_std: float = 0.5

    def __post_init__(self) -> None:
        non_negative = (
            "passenger_price_coefficient_std",
            "passenger_wait_coefficient",
            "passenger_trip_time_coefficient",
            "driver_pickup_coefficient",
            "driver_order_opportunity_cost_std",
            "operating_cost_per_km",
            "pickup_tolerance_km_std",
        )
        for field_name in non_negative:
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.passenger_price_coefficient <= 0:
            raise ValueError("passenger_price_coefficient must be positive")


class UtilityChoiceModel(PriceResponseModel):
    """Interpretable utility-choice model based on Mo et al. (2022)."""

    name = "utility_choice"
    passenger_profile_fields = frozenset(
        {"passenger_price_coefficient", "passenger_pickup_tolerance_km"}
    )
    driver_profile_fields = frozenset({"driver_order_opportunity_cost"})

    def __init__(
        self,
        config: Optional[Union[UtilityChoiceConfig, Mapping[str, Any]]] = None,
        **parameters: Any,
    ) -> None:
        if config is not None and parameters:
            raise TypeError("Use either config or keyword parameters, not both")
        if config is None:
            config = UtilityChoiceConfig(**parameters)
        elif isinstance(config, Mapping):
            config = UtilityChoiceConfig(**dict(config))
        elif not isinstance(config, UtilityChoiceConfig):
            raise TypeError("config must be UtilityChoiceConfig or a mapping")
        self.config = config

    def create_passenger_profiles(
        self, count: int, base_fares: ArrayLike, rng: Any
    ) -> Dict[str, np.ndarray]:
        del base_fares
        if self.config.passenger_price_coefficient_std == 0:
            values = np.full(count, self.config.passenger_price_coefficient)
        else:
            values = rng.lognormal(
                mean=np.log(self.config.passenger_price_coefficient),
                sigma=self.config.passenger_price_coefficient_std,
                size=count,
            )
        pickup_tolerance = np.maximum(
            rng.normal(
                self.config.pickup_tolerance_km_mean,
                self.config.pickup_tolerance_km_std,
                count,
            ),
            0.0,
        )
        return {
            "passenger_price_coefficient": values,
            "passenger_pickup_tolerance_km": pickup_tolerance,
        }

    def create_driver_profiles(self, count: int, rng: Any) -> Dict[str, np.ndarray]:
        """为每个司机生成一次固定的订单机会成本。"""
        if self.config.driver_order_opportunity_cost_std == 0:
            opportunity_cost = np.full(
                count, self.config.driver_order_opportunity_cost_mean)
        else:
            opportunity_cost = np.maximum(
                rng.normal(
                    self.config.driver_order_opportunity_cost_mean,
                    self.config.driver_order_opportunity_cost_std,
                    count,
                ),
                0.0,
            )
        return {"driver_order_opportunity_cost": opportunity_cost}

    def passenger_accept_probability(
        self,
        quoted_fare: Optional[ArrayLike] = None,
        base_fare: Optional[ArrayLike] = None,
        expected_wait_time: ArrayLike = 0.0,
        trip_time: ArrayLike = 0.0,
        *,
        offer: Optional[PassengerOffer] = None,
        profile: Optional[Profile] = None,
        maximum_wait_time: ArrayLike = 300.0,
        **legacy_profile: ArrayLike,
    ) -> ArrayLike:
        profile = combine_profile(
            profile,
            legacy_profile,
            self.passenger_profile_fields,
            self.name,
            "passenger_accept_probability",
        )
        offer = coerce_passenger_offer(
            quoted_fare, base_fare, expected_wait_time, trip_time, maximum_wait_time, offer
        )
        price_coefficient = as_float_array(
            profile.get(
                "passenger_price_coefficient",
                self.config.passenger_price_coefficient,
            )
        )
        multiplier_premium = as_float_array(offer.quoted_fare) / safe_positive(
            offer.base_fare
        ) - 1.0
        wait_minutes = as_float_array(offer.expected_wait_time) / 60.0
        trip_minutes = as_float_array(offer.trip_time) / 60.0
        utility = (
            self.config.passenger_intercept
            - price_coefficient * multiplier_premium
            - self.config.passenger_wait_coefficient * wait_minutes
            - self.config.passenger_trip_time_coefficient * trip_minutes
        )
        return return_scalar_if_scalar(sigmoid(utility))

    def passenger_cancel_unmatched_probability(
        self,
        wait_time: ArrayLike = 0.0,
        *,
        profile: Optional[Profile] = None,
        **legacy_profile: ArrayLike,
    ) -> ArrayLike:
        """Stage 3: cancel an order that is still waiting for a driver match."""
        combine_profile(profile, legacy_profile, self.passenger_profile_fields,
                        self.name, "passenger_cancel_unmatched_probability")
        wait_time = as_float_array(wait_time)
        threshold = self.config.unmatched_cancel_threshold_seconds
        if threshold is not None:
            probability = np.where(wait_time >= threshold, 1.0, 0.0)
        else:
            wait_minutes = wait_time / 60.0
            probability = sigmoid(
                self.config.unmatched_cancel_intercept
                + self.config.unmatched_cancel_wait_coefficient * wait_minutes
            )
        return return_scalar_if_scalar(probability)

    def passenger_cancel_pickup_probability(
        self,
        pickup_time: ArrayLike = 0.0,
        pickup_distance: ArrayLike = 0.0,
        *,
        profile: Optional[Profile] = None,
        **legacy_profile: ArrayLike,
    ) -> ArrayLike:
        """Stage 4: cancel after a match, based only on pickup delay/tolerance."""
        profile = combine_profile(profile, legacy_profile, self.passenger_profile_fields,
                                  self.name, "passenger_cancel_pickup_probability")
        pickup_time = as_float_array(pickup_time)
        pickup_distance = as_float_array(pickup_distance)
        tolerance = as_float_array(profile.get(
            "passenger_pickup_tolerance_km", self.config.pickup_tolerance_km_mean))
        # The sampled tolerance is a hard behavioural limit; below it, the smooth
        # time-based term keeps a small but increasing cancellation probability.
        wait_minutes = pickup_time / 60.0
        probability = sigmoid(
            self.config.pickup_cancel_intercept
            + self.config.pickup_cancel_time_coefficient * wait_minutes
        )
        probability = np.where(pickup_distance > tolerance, 1.0, probability)
        return return_scalar_if_scalar(probability)

    def driver_accept_probability(
        self,
        driver_payment: Optional[ArrayLike] = None,
        reference_payment: Optional[ArrayLike] = None,
        pickup_distance: ArrayLike = 0.0,
        trip_distance: ArrayLike = 0.0,
        *,
        offer: Optional[DriverOffer] = None,
        profile: Optional[Profile] = None,
        **legacy_profile: ArrayLike,
    ) -> ArrayLike:
        profile = combine_profile(
            profile,
            legacy_profile,
            self.driver_profile_fields,
            self.name,
            "driver_accept_probability",
        )
        offer = coerce_driver_offer(
            driver_payment, reference_payment, pickup_distance, trip_distance, offer
        )
        pickup_distance = as_float_array(offer.pickup_distance)
        trip_distance = as_float_array(offer.trip_distance)
        opportunity_cost = as_float_array(profile.get(
            "driver_order_opportunity_cost",
            self.config.driver_order_opportunity_cost_mean,
        ))
        reference_net_payment = (
            as_float_array(offer.reference_payment)
            - self.config.operating_cost_per_km * trip_distance
        )
        net_payment = (
            as_float_array(offer.driver_payment)
            - self.config.operating_cost_per_km
            * (pickup_distance + trip_distance)
            - opportunity_cost
        )
        income_premium = (
            net_payment
            / safe_positive(reference_net_payment)
            - 1.0
        )

        # 暂时保留此前的简化公式，便于后续对照实验：
        # income_premium = driver_payment / reference_payment - 1.0
        utility = (
            self.config.driver_intercept
            + self.config.driver_income_coefficient * income_premium
            - self.config.driver_pickup_coefficient * pickup_distance
        )
        return return_scalar_if_scalar(sigmoid(utility))

    def driver_online_probability(
        self,
        expected_hourly_income: Optional[ArrayLike] = None,
        reference_hourly_income: Optional[ArrayLike] = None,
        *,
        market: Optional[DriverMarket] = None,
        profile: Optional[Profile] = None,
        **legacy_profile: ArrayLike,
    ) -> ArrayLike:
        """Compatibility method; the simulator keeps all input drivers online."""
        combine_profile(profile, legacy_profile, self.driver_profile_fields,
                        self.name, "driver_online_probability")
        market = coerce_driver_market(expected_hourly_income, reference_hourly_income, market)
        return return_scalar_if_scalar(np.ones_like(as_float_array(market.expected_hourly_income)))


__all__ = ["UtilityChoiceConfig", "UtilityChoiceModel"]
