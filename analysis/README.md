# Analysis Tools

## `merge_wall_zones.py`

This script merges wall zones from a Tecplot solution and produces basic
validation plots.

### Expected Outputs

Running the script with

```bash
python merge_wall_zones.py /path/to/solution.dat --out /path/to/merged_wall.dat
```

creates the following files next to the output location (or the input
solution when `--out` is omitted):

- `*_airfoil_geometry.png` – scatter plot of the wall nodes in their
  merged order. The points should trace a closed airfoil shape.
- `*_surface_cp.png` – surface pressure coefficient plotted against the
  ordered `x` coordinate. The curve should follow the expected Cp
  distribution for the case and can be compared against reference data
  for validation.

The optional Tecplot file produced by `--out` contains the same ordered
`x`, `y` and `Cp` data.
