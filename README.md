# glacium-configurator

This repository contains configuration files used by the Glacium project.

## Dependencies

The examples require the following Python packages:

- [pyvista](https://docs.pyvista.org/)
- [matplotlib](https://matplotlib.org/)

Install them with:

```bash
pip install pyvista matplotlib
```

## Usage

### Generate `global_default.yaml`

The `src/generate_default.py` script rebuilds `conf/global_default.yaml` by
merging all YAML files from the `conf` directory. Later files override earlier
ones during the merge.

```bash
python -m src.generate_default
```

Optional arguments allow changing the configuration directory or output file:

```bash
python -m src.generate_default --conf-dir conf --output conf/global_default.yaml
```

### Merge wall zones

`src/merge_wall_zones.py` combines multiple mesh zone IDs into a single wall
zone and optionally plots the zone counts.

```bash
python -m src.merge_wall_zones --input examples/sample_mesh.vtp \
    --output examples/merged_mesh.vtp --wall-zones 1 2 \
    --plot examples/zone_counts.png
```

The same options can be provided in a YAML file:

```bash
python -m src.merge_wall_zones --config examples/merge_config.yaml
```

The script writes the merged mesh to `--output`. If `--plot` is supplied a bar
chart of zone counts is saved. Sample data lives in `examples/`.
