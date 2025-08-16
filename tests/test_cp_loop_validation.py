import sys
from pathlib import Path

import pytest

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import merge_wall_nodes, compute_cp, read_solution


def _load():
    data_path = Path(__file__).with_name("sample_cp_loop.dat")
    wall_zones, inlet_zones, _, _, var_map, _, _, _ = read_solution(data_path)
    return wall_zones, inlet_zones, var_map


def test_cp_loop_validation_passes():
    wall_zones, inlet_zones, var_map = _load()
    nodes, _ = merge_wall_nodes(wall_zones, var_map)
    # Should not raise for the unmodified dataset
    compute_cp(nodes, var_map, inlet_zones)


def test_cp_loop_validation_fails_on_jump():
    wall_zones, inlet_zones, var_map = _load()
    p_idx = var_map["p"]
    # Introduce a large pressure spike to trigger the Cp jump check
    wall_zones[-1].nodes[-1, p_idx] = 60.0
    nodes, _ = merge_wall_nodes(wall_zones, var_map)
    with pytest.raises(ValueError):
        compute_cp(nodes, var_map, inlet_zones)
