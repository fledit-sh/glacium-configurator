import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from reynolds import FlowState, calculate_reynolds

def test_calculate_reynolds_simple():
    flow = FlowState(rho=1.0, velocity=2.0, chord=3.0, mu=4.0)
    expected = flow.rho * flow.velocity * flow.chord / flow.mu
    assert math.isclose(calculate_reynolds(flow), expected)
