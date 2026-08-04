"""Solver utilities for the day-night model."""

from .common_utils import (
    DEFAULT_INITIAL_CENTER,
    DEFAULT_INITIAL_WIDTH,
    DEFAULT_SMELL_RADIUS,
    DEFAULT_SIGHT_RADIUS,
    compute_spread_indicator,
    gaussian_initial_condition,
)
from .solver import DayNightModel1D

__all__ = [
    "DEFAULT_INITIAL_CENTER",
    "DEFAULT_INITIAL_WIDTH",
    "DEFAULT_SMELL_RADIUS",
    "DEFAULT_SIGHT_RADIUS",
    "DayNightModel1D",
    "compute_spread_indicator",
    "gaussian_initial_condition",
]