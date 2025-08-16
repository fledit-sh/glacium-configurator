import numpy as np
from pathlib import Path
import sys

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import compute_dynamic_pressure, compute_speed


def test_compute_dynamic_pressure_and_speed():
    # Node layout: [x, y, z, p, rho, u, v, w]
    nodes = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 4.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 5.0],
        ]
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
    speed = compute_speed(nodes, var_map)
    q = compute_dynamic_pressure(nodes, var_map)

    expected_speed = np.sqrt(nodes[:, 5] ** 2 + nodes[:, 6] ** 2 + nodes[:, 7] ** 2)
    expected_q = 0.5 * nodes[:, 4] * expected_speed ** 2

    assert np.allclose(speed, expected_speed)
    assert np.allclose(q, expected_q)
