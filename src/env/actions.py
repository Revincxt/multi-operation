"""Action types used by the grid-level simulator."""

from dataclasses import dataclass
import math
from numbers import Integral


DEFAULT_PRICE_MULTIPLIERS = (
    0.4, 1.0, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 4.0,
)
DEFAULT_MATCHING_RADII_KM = (
    0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0,
)


@dataclass(frozen=True)
class GridAction:
    """One platform decision for one grid at one decision step.

    ``price_multiplier`` affects newly generated requests, while
    ``matching_radius_km`` affects the orders considered for dispatch in the
    current step.  The simulator validates both values against its configured
    discrete action sets when the action is resolved.
    """

    price_multiplier: float = 1.0
    matching_radius_km: float = 1.0

    def __post_init__(self):
        price = float(self.price_multiplier)
        radius = float(self.matching_radius_km)
        if not math.isfinite(price) or price <= 0:
            raise ValueError("price_multiplier must be finite and positive")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("matching_radius_km must be finite and positive")
        object.__setattr__(self, 'price_multiplier', price)
        object.__setattr__(self, 'matching_radius_km', radius)


def uniform_grid_actions(
    grid_num: int,
    price_multiplier: float = 1.0,
    matching_radius_km: float = 1.0,
):
    """Build a complete baseline or diagnostic action mapping."""
    if isinstance(grid_num, bool) or not isinstance(grid_num, Integral) or grid_num <= 0:
        raise ValueError("grid_num must be a positive integer")
    action = GridAction(price_multiplier, matching_radius_km)
    return {grid_id: action for grid_id in range(int(grid_num))}
