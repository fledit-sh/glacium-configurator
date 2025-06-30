#!/usr/bin/env python3
from dataclasses import dataclass, asdict
import yaml, jinja2, jsonschema, pathlib

# ---------- 1. High-Level Spec ----------------------------------------------
@dataclass
class Case:
    mach: float
    reynolds: float
    alpha_start: int
    alpha_end: int
    lwc: float                # liquid water content [kg/m³]
    drop_mvd: float = 20e-6   # default 20 µm
    grid_level: str = "coarse"  # coarse | medium | fine

# ---------- 2. Regeln / Ableitungen -----------------------------------------
def derive(case: Case) -> dict:
    """Mappt High-Level-Speck auf FENSAP-Variablendschungel."""
    cfg = {}

    # Beispiel: Profil-Polaren
    cfg["PWS_POL_REYNOLDS"] = case.reynolds
    cfg["PWS_POL_MACH"]     = case.mach
    cfg["PWS_POL_ALPHA_START"] = case.alpha_start
    cfg["PWS_POL_ALPHA_END"]   = case.alpha_end

    # Gitter-Auflösung
    levels = {"coarse": 8, "medium": 16, "fine": 32}
    cfg["PWS_REFINEMENT"] = levels[case.grid_level]

    # Icing-Zeug
    cfg["ICE_DROP_DIAM"]     = case.drop_mvd*1e6   # µm
    cfg["ICE_LIQ_H2O_CONTENT"] = case.lwc

    # …weiteres Mapping hier…
    return cfg

# ---------- 3. Rendern via Jinja-Template ------------------------------------
def render_fensap(cfg: dict, template_path="template.j2") -> str:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("."),
                             trim_blocks=True, lstrip_blocks=True)
    template = env.get_template(template_path)
    return template.render(**cfg)

# ---------- 4. Vollpipeline ---------------------------------------------------
def build(casefile: str, outfile="icing.def"):
    case = Case(**yaml.safe_load(open(casefile)))
    raw  = derive(case)
    open(outfile, "w").write(render_fensap(raw))
    print(f"✅  {outfile} geschrieben.")

if __name__ == "__main__":
    build("case.yaml")
