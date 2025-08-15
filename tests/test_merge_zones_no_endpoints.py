import numpy as np
from types import SimpleNamespace
from pathlib import Path
import sys

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

import pytest

from merge_wall_zones import merge_zones


def _node(x):
    # [x, y, z, p, rho, u, v, w]
    return [x, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_merge_zones_no_endpoints_message():
    nodes = np.array([_node(0.0), _node(1.0)])
    zone = SimpleNamespace(nodes=nodes, elem=None, title="TestZone")

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

    with pytest.raises(
        ValueError, match=r"Zone 1 \(TestZone\) has no endpoints"
    ):
        merge_zones([zone], [], var_map)

