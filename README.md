# glacium-configurator

This repository contains configuration files used by the Glacium project.

The `src/generate_default.py` script can rebuild `conf/global_default.yaml` by
merging all YAML files from the `conf` directory. Later files override earlier
ones during the merge.

## Usage

```bash
python -m src.generate_default
```

Optional arguments allow changing the configuration directory or output file:

```bash
python -m src.generate_default --conf-dir conf --output conf/global_default.yaml
```
