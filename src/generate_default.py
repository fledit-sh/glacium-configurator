import argparse
from pathlib import Path
import yaml

from .strategies import (
    PressureStrategy,
    FixedPressureStrategy,
    AltitudePressureStrategy,
)


def merge_dicts(a, b):
    """Recursively merge dict b into dict a and return the result."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return b
    result = dict(a)
    for key, value in b.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _apply_pressure_strategy(data: dict, strategy: PressureStrategy) -> None:
    for key in ("FSP_FREESTREAM_PRESSURE", "ICE_REF_AIR_PRESSURE"):
        if key in data:
            data[key] = strategy.calculate(data)
    for value in list(data.values()):
        if isinstance(value, dict):
            _apply_pressure_strategy(value, strategy)


def build_default(
    conf_dir: Path, output_path: Path, strategy: PressureStrategy, extra: dict
) -> None:
    files = sorted(conf_dir.glob("*.yaml"))
    merged: dict = {}
    for file in files:
        if file.resolve() == output_path.resolve():
            continue
        with open(file, "r") as f:
            data = yaml.safe_load(f) or {}
        data.update(extra)
        _apply_pressure_strategy(data, strategy)
        merged = merge_dicts(merged, data)
    output_path.write_text(yaml.dump(merged, sort_keys=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate global_default.yaml by merging YAML files.")
    parser.add_argument('--conf-dir', default='conf', type=Path, help='Directory containing YAML files.')
    parser.add_argument('--output', default=None, type=Path, help='Output YAML file path.')
    parser.add_argument(
        '--strategy',
        choices=['fixed', 'altitude'],
        default='fixed',
        help='Pressure calculation strategy.'
    )
    parser.add_argument(
        '--pressure',
        type=float,
        default=101325.0,
        help='Fixed pressure value (Pa) if using the fixed strategy.'
    )
    parser.add_argument(
        '--altitude',
        type=float,
        default=0.0,
        help='Altitude in meters for the altitude strategy.'
    )
    args = parser.parse_args(argv)

    output = args.output or args.conf_dir / 'global_default.yaml'
    if args.strategy == 'altitude':
        strategy = AltitudePressureStrategy()
        extra = {'altitude': args.altitude}
    else:
        strategy = FixedPressureStrategy(args.pressure)
        extra = {}

    build_default(args.conf_dir, output, strategy, extra)


if __name__ == '__main__':
    main()
