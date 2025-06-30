from dataclasses import dataclass

from lambda_explorer.tools.aero_formulas import ReynoldsNumber
from lambda_explorer.tools.solver import FormulaSolver


@dataclass
class FlowState:
    """Airflow state used for Reynolds number calculation."""
    rho: float
    velocity: float
    chord: float
    mu: float


def calculate_reynolds(flow: FlowState) -> float:
    """Calculate the Reynolds number for the given flow state."""
    solver = FormulaSolver(ReynoldsNumber())
    return solver.solve({
        'rho': flow.rho,
        'velocity': flow.velocity,
        'characteristic_length': flow.chord,
        'mu': flow.mu,
    })
