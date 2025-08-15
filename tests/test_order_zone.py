import numpy as np
from types import SimpleNamespace
from pathlib import Path
import sys

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import order_zone
from node_order import nearest_neighbor_order


def test_order_zone_no_elements_uses_nearest_neighbor():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    zone = SimpleNamespace(nodes=nodes, elem=None)
    result = order_zone(zone, 0, 1)
    expected = np.array(nearest_neighbor_order(nodes), dtype=int)
    assert np.array_equal(result, expected)
