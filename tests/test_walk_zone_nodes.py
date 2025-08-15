import numpy as np
from types import SimpleNamespace
from pathlib import Path
import sys

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import walk_zone_nodes


def test_walk_zone_nodes_no_elements_returns_in_order():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    zone = SimpleNamespace(nodes=nodes, elem=None)
    result = walk_zone_nodes(zone)
    expected = np.arange(nodes.shape[0])
    assert np.array_equal(result, expected)


def test_walk_zone_nodes_empty_connectivity_returns_in_order():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    zone = SimpleNamespace(nodes=nodes, elem=np.empty((0, 2), dtype=int))
    result = walk_zone_nodes(zone)
    expected = np.arange(nodes.shape[0])
    assert np.array_equal(result, expected)


def test_walk_zone_nodes_wrong_dimensionality_returns_in_order():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    zone = SimpleNamespace(nodes=nodes, elem=np.array([0, 1, 2]))
    result = walk_zone_nodes(zone)
    expected = np.arange(nodes.shape[0])
    assert np.array_equal(result, expected)
