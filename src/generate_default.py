import argparse
from pathlib import Path
import yaml


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


def build_default(conf_dir: Path, output_path: Path):
    files = sorted(conf_dir.glob('*.yaml'))
    merged = {}
    for file in files:
        if file.resolve() == output_path.resolve():
            continue
        with open(file, 'r') as f:
            data = yaml.safe_load(f) or {}
        merged = merge_dicts(merged, data)
    output_path.write_text(yaml.dump(merged, sort_keys=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate global_default.yaml by merging YAML files.")
    parser.add_argument('--conf-dir', default='conf', type=Path, help='Directory containing YAML files.')
    parser.add_argument('--output', default=None, type=Path, help='Output YAML file path.')
    args = parser.parse_args(argv)

    output = args.output or args.conf_dir / 'global_default.yaml'
    build_default(args.conf_dir, output)


if __name__ == '__main__':
    main()
