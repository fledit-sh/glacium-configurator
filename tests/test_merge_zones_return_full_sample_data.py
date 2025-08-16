import numpy as np
from pathlib import Path
import sys

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import merge_zones, read_solution


def test_return_full_on_sample_data_preserves_columns_and_connectivity():
    data_path = Path(__file__).with_name("sample_cp_loop.dat")
    wall_zones, inlet_zones, _, _, var_map, _, _, _ = read_solution(data_path)

    nodes_cp, conn = merge_zones(wall_zones, inlet_zones, var_map, return_full=True)

    # Node table should retain original variables and append Cp
    n_vars = wall_zones[0].nodes.shape[1]
    assert nodes_cp.shape[1] == n_vars + 1

    x_idx = var_map["x"]
    y_idx = var_map["y"]
    p_idx = var_map["p"]
    rho_idx = var_map["rho"]
    u_idx = var_map["u"]
    v_idx = var_map["v"]
    w_idx = var_map["w"]

    # Reconstruct expected node ordering following orientation logic
    expected_nodes = []
    prev_first = None
    for z in wall_zones:
        nodes = z.nodes
        if prev_first is not None:
            d_start = np.linalg.norm(nodes[0, [x_idx, y_idx]] - prev_first)
            d_end = np.linalg.norm(nodes[-1, [x_idx, y_idx]] - prev_first)
            if d_end < d_start:
                nodes = nodes[::-1]
        expected_nodes.append(nodes)
        prev_first = nodes[0, [x_idx, y_idx]]
    expected_nodes = np.vstack(expected_nodes)

    assert np.allclose(nodes_cp[:, :-1], expected_nodes)

    # Cp column should match expected values from the original data
    if inlet_zones:
        inlet_nodes = np.concatenate([z.nodes for z in inlet_zones])
        rho_inf = float(np.median(inlet_nodes[:, rho_idx]))
        p_inf = float(np.median(inlet_nodes[:, p_idx]))
        vel_mag = np.sqrt(
            inlet_nodes[:, u_idx] ** 2
            + inlet_nodes[:, v_idx] ** 2
            + inlet_nodes[:, w_idx] ** 2
        )
        u_inf = float(np.median(vel_mag))
    else:
        first = wall_zones[0].nodes[0]
        rho_inf = first[rho_idx]
        p_inf = first[p_idx]
        u_inf = float(
            np.sqrt(first[u_idx] ** 2 + first[v_idx] ** 2 + first[w_idx] ** 2)
        )

    expected_cp = (expected_nodes[:, p_idx] - p_inf) / (0.5 * rho_inf * u_inf ** 2)
    assert np.allclose(nodes_cp[:, -1], expected_cp)

    # Connectivity should link consecutive points and close the loop
    n = nodes_cp.shape[0]
    expected_conn = set(
        map(tuple, np.column_stack([np.arange(n), np.roll(np.arange(n), -1)]))
    )
    assert set(map(tuple, conn)) == expected_conn
