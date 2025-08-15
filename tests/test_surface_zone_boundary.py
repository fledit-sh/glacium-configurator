import sys
from pathlib import Path
import numpy as np

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import read_solution, walk_zone_nodes


def test_quadrilateral_wall_zone_boundary_has_two_endpoints():
    data_path = Path(__file__).with_name("sample_quadrilateral_wall_zone.dat")
    wall_zones, inlet_zones, *_ = read_solution(data_path)
    assert inlet_zones == []
    assert len(wall_zones) == 1
    zone = wall_zones[0]
    # Boundary edges should reduce the single quadrilateral to three segments
    assert zone.elem.shape == (3, 2)
    order, n_endpoints, is_closed = walk_zone_nodes(zone)
    assert np.array_equal(order, np.array([0, 1, 2, 3]))
    assert n_endpoints == 2
    assert not is_closed
