import sys
from pathlib import Path
import numpy as np

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import read_solution, merge_wall_nodes


def test_surface_zone_sliced_has_endpoints():
    data_path = Path(__file__).with_name("sample_surface_crossing_plane.dat")
    wall_zones, inlet_zones, total_nodes, wall_nodes, var_map, *_ = read_solution(data_path)
    assert inlet_zones == []
    assert len(wall_zones) == 1
    zone = wall_zones[0]
    # Slicing should produce a single edge with two nodes at z=0
    assert zone.elem.shape == (1, 2)
    assert np.allclose(zone.nodes[:, 2], 0.0)
    # merge_wall_nodes should process without raising the "no endpoints" error
    merge_wall_nodes([zone], var_map)
