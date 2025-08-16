import numpy as np
from types import SimpleNamespace
from pathlib import Path
import sys

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import merge_zones


def _node(x: float) -> list[float]:
    # [x, y, z, p, rho, u, v, w]
    return [x, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_return_full_provides_nodes_and_conn():
    z1 = SimpleNamespace(
        nodes=np.array([
            _node(3.0),
            _node(2.0),
        ]),
        elem=np.array([[0, 1]], dtype=int),
    )
    z2 = SimpleNamespace(
        nodes=np.array([
            _node(1.0),
            _node(0.0),
        ]),
        elem=np.array([[0, 1]], dtype=int),
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

    nodes_cp, conn = merge_zones([z1, z2], [], var_map, return_full=True)

    assert nodes_cp.shape[1] == 9
    assert np.allclose(nodes_cp[:, -1], 0.0)

    n = nodes_cp.shape[0]
    expected = set(map(tuple, np.column_stack([np.arange(n), np.roll(np.arange(n), -1)])))
    assert set(map(tuple, conn)) == expected
