import numpy as np
from pathlib import Path
import sys

# Ensure modules in the analysis directory are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))

from merge_wall_zones import write_tecplot


def test_write_tecplot_writes_expected_format(tmp_path):
    nodes = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 4.0],
            [2.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 0.0, 6.0],
        ]
    )
    conn = np.array([[0, 1], [1, 2], [2, 0]])
    var_names = ["X", "Y", "Z", "P", "Rho", "U", "V", "W"]

    out_file = tmp_path / "out.dat"
    write_tecplot(out_file, nodes, conn, var_names, "Cp")

    text = out_file.read_text().splitlines()

    zone_line = next(line for line in text if line.startswith("ZONE"))
    assert "N=3" in zone_line
    assert "E=3" in zone_line
    assert "ZONETYPE=FELINESEG" in zone_line

    var_line = next(line for line in text if line.startswith("VARIABLES"))
    for v in var_names + ["Cp"]:
        assert f'"{v}"' in var_line

    conn_lines = text[-conn.shape[0]:]
    assert conn_lines == ["1 2", "2 3", "3 1"]
