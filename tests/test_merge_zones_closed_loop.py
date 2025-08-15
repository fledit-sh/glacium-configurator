import numpy as np
from types import SimpleNamespace
from pathlib import Path
import sys

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import merge_zones


def _node(x):
    # [x, y, z, p, rho, u, v, w]
    return [x, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_merge_zones_single_closed_loop():
    xs = [1.0, 0.0, -1.0, 0.0]
    nodes = np.array([_node(x) for x in xs])
    elem = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    zone = SimpleNamespace(nodes=nodes, elem=elem)

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

    x_closed, y_closed, cp_closed = merge_zones([zone], [], var_map)
    assert x_closed.shape[0] == len(xs) + 1
    assert x_closed[0] == max(xs)
    assert np.allclose(cp_closed, 0.0)


def test_merge_zones_multiple_closed_loops():
    xs1 = [1.0, 0.0, -1.0, 0.0]
    xs2 = [4.0, 3.0, 2.0, 3.0]
    zone1 = SimpleNamespace(
        nodes=np.array([_node(x) for x in xs1]),
        elem=np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int),
    )
    zone2 = SimpleNamespace(
        nodes=np.array([_node(x) for x in xs2]),
        elem=np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int),
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

    x_closed, y_closed, cp_closed = merge_zones([zone1, zone2], [], var_map)
    assert x_closed.shape[0] == len(xs1) + len(xs2) + 1
    assert np.allclose(cp_closed, 0.0)
