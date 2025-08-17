#!/usr/bin/env python3
"""Batch process multiple shot datasets with analysis scripts.

The script scans an input directory for six-digit shot IDs and for each ID
expects three ``.dat`` (or ``.zip``) files.  For every shot the files are merged
and analysed by ``00_merge.py``, ``01_auto_cp_normals.py``, ``01_plot.py`` and
``02_correlate.py`` with their results stored under ``<output>/<shot_id>/``.
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import sys

ID_RE = re.compile(r"(\d{6})")
SCRIPT_DIR = Path(__file__).parent


def extract_dat(path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """Return a path to a .dat file, extracting from zip if needed."""
    if path.suffix.lower() != ".zip":
        return path, None
    tmp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(tmp.name)
    dats = sorted(Path(tmp.name).rglob("*.dat"))
    if not dats:
        tmp.cleanup()
        raise FileNotFoundError(f"No .dat inside {path}")
    return dats[0], tmp


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def find_shots(input_dir: Path) -> Dict[str, List[Path]]:
    shots: Dict[str, List[Path]] = {}
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        if not (p.suffix.lower() in {".dat", ".zip"} or p.name.endswith(".dat.zip")):
            continue
        m = ID_RE.search(p.name)
        if not m:
            continue
        sid = m.group(1)
        shots.setdefault(sid, []).append(p)
    return shots


def process_shot(shot_id: str, files: List[Path], out_root: Path) -> None:
    logging.info("Processing shot %s", shot_id)
    shot_dir = out_root / shot_id
    shot_dir.mkdir(parents=True, exist_ok=True)

    # Identify base and augment files
    base_candidates = [p for p in files if re.search(r"soln|fensap", p.name, re.I)]
    base = base_candidates[0] if base_candidates else sorted(files)[0]
    augments = [p for p in files if p != base]

    tmp_objs: List[tempfile.TemporaryDirectory] = []
    try:
        base_dat, tmp_base = extract_dat(base)
        if tmp_base:
            tmp_objs.append(tmp_base)
        aug_dats: List[Path] = []
        for a in augments:
            dat, tmp = extract_dat(a)
            aug_dats.append(dat)
            if tmp:
                tmp_objs.append(tmp)

        with tempfile.TemporaryDirectory() as workdir:
            wd = Path(workdir)
            merged = wd / "merged.dat"
            merge_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "00_merge.py"),
                str(base_dat),
                "--out",
                str(merged),
            ]
            for a in aug_dats:
                merge_cmd += ["--augment", str(a)]
            run_cmd(merge_cmd, cwd=wd)

            normals_png = wd / "merged_normals.png"
            run_cmd(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "01_auto_cp_normals.py"),
                    "--solution",
                    str(base_dat),
                    "--merged-in",
                    str(merged),
                    "--png-out",
                    str(normals_png),
                ],
                cwd=wd,
            )

            plots_dir = wd / "plots"
            run_cmd(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "01_plot.py"),
                    str(merged),
                    "--outdir",
                    str(plots_dir),
                ],
                cwd=wd,
            )

            corr_dir = wd / "correlation_analysis"
            run_cmd(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "02_correlate.py"),
                    str(merged),
                    "--outdir",
                    str(corr_dir),
                ],
                cwd=wd,
            )

            for item in wd.iterdir():
                dest = shot_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
    finally:
        for tmp in tmp_objs:
            tmp.cleanup()


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-run analysis scripts over multiple shots")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--start-shot", type=int, default=None)
    ap.add_argument("--end-shot", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    shots = find_shots(args.input_dir)
    for sid in sorted(shots):
        val = int(sid)
        if args.start_shot is not None and val < args.start_shot:
            continue
        if args.end_shot is not None and val > args.end_shot:
            continue
        files = shots[sid]
        if len(files) != 3:
            logging.warning("Shot %s has %d files (expected 3); skipping", sid, len(files))
            continue
        try:
            process_shot(sid, files, args.output_dir)
        except Exception as exc:
            logging.error("Shot %s failed: %s", sid, exc)


if __name__ == "__main__":
    main()
