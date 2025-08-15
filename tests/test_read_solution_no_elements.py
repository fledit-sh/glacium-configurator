import sys
from pathlib import Path

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import read_solution, order_zone


def test_read_solution_and_order_zone_no_elements():
    data_path = Path(__file__).with_name("sample_wall_zone_no_elements.dat")
    wall_zones, _, _, var_map, _ = read_solution(data_path)
    assert len(wall_zones) == 1
    zone = wall_zones[0]
    ordering = order_zone(zone, var_map["x"], var_map["y"])
    assert len(ordering) == zone.nodes.shape[0]
