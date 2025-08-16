# Analysis Tools

## `merge_wall_zones.py`

This script merges wall zones from a Tecplot solution and produces basic
validation plots.  The calculation to perform on the merged wall nodes is
selected via the ``--calc`` option (default: ``cp``).

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
  ordered `x` coordinate when ``--calc cp`` is used. The curve should follow
  the expected Cp distribution for the case and can be compared against
  reference data for validation.

The optional Tecplot file produced by ``--out`` contains the ordered ``x``,
``y`` and the selected derived quantity.

### Adding custom calculations

Derived quantities are dispatched through the ``CALC_HANDLERS`` mapping in
``merge_wall_zones.py``.  To extend the script:

1. Implement a function ``compute_*`` accepting ``nodes``, ``var_map`` and an
   optional ``inlet_zones`` list and returning a one-dimensional NumPy array
   with one value per node.
2. Add the function to ``CALC_HANDLERS`` and provide a label in
   ``CALC_LABELS``.
3. Invoke the script with ``--calc <name>`` to run the custom calculation.

### Boundary edges and automatic slicing

`merge_wall_zones.py` expects every surface zone to provide boundary edges so
that the node order can be walked from one endpoint to the other.  Triangular
and quadrilateral Tecplot zones often describe only faces; without boundary
edges the merge routine cannot identify endpoints and fails with a `no
endpoints` error.

When boundary edges are missing the script now slices the surface elements by a
plane at `z=z_threshold` and constructs edges along the intersection.  The
generated edges allow the zone to be merged like any other wall segment.

**Limitations and assumptions**

- The slice is planar and aligned with the `z` axis at the specified
  `z_threshold` (with optional `--tolerance`). Surfaces that do not intersect
  this plane will still produce the `no endpoints` error.
- Multiple disconnected intersections are not connected into a single loop.
- The operation assumes the surface geometry is reasonably planar near the
  cutting plane.

**Example commands**

```bash
# Slice and merge directly from a Tecplot solution
python merge_wall_zones.py analysis/soln.fensap.000001.zip --z-threshold 0.0

# Increase the tolerance when nodes should lie on the plane but fail to slice
python merge_wall_zones.py path/to/solution.dat --z-threshold 0.0 --tolerance 1e-6
```

**Troubleshooting**

- If slicing still fails, confirm that the surface crosses the `z=z_threshold`
  plane and adjust the threshold or tolerance accordingly.
- Ensure the zone contains surface elements (`FEQUADRILATERAL` or
  `FETRIANGLE`). Volume zones cannot be sliced into boundary edges.
- For complex geometry or multiple intersections, create boundary edges
  manually in Tecplot before running the script.
