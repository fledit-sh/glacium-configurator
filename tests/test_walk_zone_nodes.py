import numpy as np
from types import SimpleNamespace
from pathlib import Path
import sys

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import walk_zone_nodes, filter_nodes_by_z


def test_walk_zone_nodes_no_elements_returns_in_order():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    zone = SimpleNamespace(nodes=nodes, elem=None)
    result, n_endpoints = walk_zone_nodes(zone)
    expected = np.arange(nodes.shape[0])
    assert np.array_equal(result, expected)
    assert n_endpoints == 0


def test_walk_zone_nodes_empty_connectivity_returns_in_order():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    zone = SimpleNamespace(nodes=nodes, elem=np.empty((0, 2), dtype=int))
    result, n_endpoints = walk_zone_nodes(zone)
    expected = np.arange(nodes.shape[0])
    assert np.array_equal(result, expected)
    assert n_endpoints == 0


def test_walk_zone_nodes_wrong_dimensionality_returns_in_order():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    zone = SimpleNamespace(nodes=nodes, elem=np.array([0, 1, 2]))
    result, n_endpoints = walk_zone_nodes(zone)
    expected = np.arange(nodes.shape[0])
    assert np.array_equal(result, expected)
    assert n_endpoints == 0


def test_walk_zone_nodes_split_components_count_endpoints():
    # Original zone: square (0-1-2-3) with a tail (1-4-5-6). Node 4 has z=1 and
    # is filtered out, leaving a closed square and a disconnected segment (5-6).
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [1.0, 0.0, 1.0],  # 4 (to be filtered)
            [2.0, 0.0, 0.0],  # 5
            [3.0, 0.0, 0.0],  # 6
        ]
    )
    elem = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [1, 4],
            [4, 5],
            [5, 6],
        ],
        dtype=int,
    )

    filtered_nodes, filtered_elem = filter_nodes_by_z(nodes, elem, 2)
    zone = SimpleNamespace(nodes=filtered_nodes, elem=filtered_elem)
    order, n_endpoints = walk_zone_nodes(zone)
    expected_order = np.array([0, 1, 2, 3, 4, 5])
    assert np.array_equal(order, expected_order)
    assert n_endpoints == 2
