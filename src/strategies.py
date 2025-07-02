from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

import sympy as sp
from lambda_explorer.tools import formula_registry


class PressureStrategy(ABC):
    """Abstract base class for pressure calculation strategies."""

    @abstractmethod
    def calculate(self, data: Dict[str, Any]) -> float:
        """Calculate pressure based on the provided data."""
        raise NotImplementedError


class FixedPressureStrategy(PressureStrategy):
    """Return a constant pressure value."""

    def __init__(self, pressure: float) -> None:
        self.pressure = pressure

    def calculate(self, data: Dict[str, Any]) -> float:
        return float(self.pressure)


class AltitudePressureStrategy(PressureStrategy):
    """Calculate pressure from altitude using the barometric formula."""

    def __init__(self) -> None:
        self.registry = formula_registry.FormulaRegistry()
        # Use an exponential barometric equation as a default.
        P, P0, M, g, h, R, T0 = sp.symbols("P P0 M g h R T0")
        eq = sp.Eq(P, P0 * sp.exp(-M * g * h / (R * T0)))
        self.formula_cls = self.registry.create_formula(
            "BarometricPressure", ["P", "P0", "M", "g", "h", "R", "T0"], eq
        )

    def calculate(self, data: Dict[str, Any]) -> float:
        altitude = data.get("altitude", 0.0)
        params = {
            "P0": data.get("P0", 101325.0),
            "M": data.get("M", 0.0289644),
            "g": data.get("g", 9.80665),
            "h": altitude,
            "R": data.get("R", 8.31447),
            "T0": data.get("T0", 288.15),
        }
        return self.formula_cls().solve(**params)
