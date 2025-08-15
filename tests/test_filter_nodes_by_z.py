import sys
from pathlib import Path
import numpy as np

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import filter_nodes_by_z


def test_filter_nodes_by_z_removes_positive_and_remaps():
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.1],
        [2.0, 0.0, -0.2],
        [3.0, 0.0, 0.0],
    ])
    elem = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    filtered_nodes, filtered_elem = filter_nodes_by_z(nodes, elem, 2)
    assert filtered_nodes.shape[0] == 3
    assert np.all(filtered_nodes[:, 2] <= 0)
    expected_elem = np.array([[1, 2], [2, 0]], dtype=int)
    assert np.array_equal(filtered_elem, expected_elem)
