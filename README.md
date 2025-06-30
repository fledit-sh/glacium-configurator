# Glacium Configurator

This repository contains a small tool that translates a concise YAML case
description into a FENSAP input file. The script calculates parameters such as
the Reynolds number from the given flow state and renders a final
`icing.def` configuration using Jinja templates.

## Example case file

```yaml
flow:
  rho: 1.225
  velocity: 100
  chord: 0.431
  mu: 1.7e-05
mach: 0.15
alpha_start: -2
alpha_end: 10
lwc: 0.001
```

The `flow:` section defines the air properties used for the Reynolds number
calculation. Additional values like `mach` and angle of attack complete the
case description.

## Usage

To generate an `icing.def` file run:

```bash
python configurator.py case.yaml
```

The command reads the case file, derives all necessary FENSAP variables and
writes the output configuration into `icing.def`.
