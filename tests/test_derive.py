import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configurator import Case, derive
from reynolds import FlowState, calculate_reynolds

def test_derive_reynolds_mapping():
    flow = FlowState(rho=1.0, velocity=2.0, chord=3.0, mu=4.0)
    case = Case(mach=0.5, flow=flow, alpha_start=0, alpha_end=1, lwc=0.0)
    cfg = derive(case)
    expected = calculate_reynolds(flow)
    keys = [
        "PWS_POL_REYNOLDS",
        "PWS_PSI_REYNOLDS",
        "ICE_REYNOLDS_NUMBER",
        "FSP_REYNOLDS_NUMBER",
    ]
    for k in keys:
        assert math.isclose(cfg[k], expected)
