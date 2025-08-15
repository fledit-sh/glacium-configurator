import numpy as np
from types import SimpleNamespace
from pathlib import Path
import sys

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import merge_zones

def _make_nodes(xs, cps):
    nodes = []
    for x, cp in zip(xs, cps):
        p = 0.5 * cp  # rho=1, u=1, p_inf=0 => cp = 2*p
        nodes.append([x, 0.0, 0.0, p, 1.0, 1.0, 0.0, 0.0])
    return np.array(nodes)

def test_merge_zones_opposite_orientation_monotonic_cp():
    elem = np.array([[0, 1], [1, 2]], dtype=int)

    # Zone 1: already oriented with decreasing x
    z1 = SimpleNamespace(
        nodes=_make_nodes([2.0, 1.0, 0.0], [5.0, 4.0, 3.0]),
        elem=elem.copy(),
    )

    # Zone 2: opposite orientation (increasing x)
    z2 = SimpleNamespace(
        nodes=_make_nodes([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
        elem=elem.copy(),
    )

    inlet = SimpleNamespace(
        nodes=np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    )

    var_map = {
        "x": 0,
        "y": 1,
        "z": 2,
        "p": 3,
        "rho": 4,
        "u": 5,
        "v": 6,
        "w": 7,
    }

    x, y, cp = merge_zones([z1, z2], [inlet], var_map)
    cp_curve = cp[:-1]  # last point repeats the first
    assert np.all(np.diff(cp_curve) <= 0)
