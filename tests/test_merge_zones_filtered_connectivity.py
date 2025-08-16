import numpy as np
from types import SimpleNamespace
from pathlib import Path
import sys

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import merge_wall_nodes


def _node(x, z):
    # [x, y, z, p, rho, u, v, w]
    return [x, 0.0, z, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_connectivity_preserved_after_initial_z_filter():
    # Zone 1 has a node slightly above z=0 that should be retained
    z1 = SimpleNamespace(
        nodes=np.array([
            _node(0.0, 0.0),
            _node(1.0, 0.05),  # within tolerance
        ]),
        elem=np.array([[0, 1]], dtype=int),
    )

    # Zone 2 consists entirely of z<=0 nodes
    z2 = SimpleNamespace(
        nodes=np.array([
            _node(2.0, 0.0),
            _node(3.0, 0.0),
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

    nodes, _ = merge_wall_nodes([z1, z2], var_map)
    x = nodes[:, var_map["x"]]
    x_closed = np.append(x, x[0])

    # All four nodes should be present in the closed loop
    assert x_closed.shape[0] == 5  # four unique points + closure
    assert 1.0 in x_closed[:-1]
