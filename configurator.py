#!/usr/bin/env python3
from dataclasses import dataclass
import yaml, jinja2, jsonschema, pathlib
from reynolds import FlowState, calculate_reynolds


class ConfigBuilder:
    """Collects configuration data for FENSAP."""

    def __init__(self, calculator=calculate_reynolds):
        self.calculator = calculator
        self.cfg: dict[str, float | int | str] = {}

    def derive(self, case: "Case") -> dict:
        """Map high-level case data to configuration entries."""
        re = self.calculator(case.flow)

        self.cfg["PWS_POL_REYNOLDS"] = re
        self.cfg["PWS_PSI_REYNOLDS"] = re
        self.cfg["ICE_REYNOLDS_NUMBER"] = re
        self.cfg["FSP_REYNOLDS_NUMBER"] = re

        self.cfg["PWS_POL_MACH"] = case.mach
        self.cfg["PWS_POL_ALPHA_START"] = case.alpha_start
        self.cfg["PWS_POL_ALPHA_END"] = case.alpha_end

        levels = {"coarse": 8, "medium": 16, "fine": 32}
        self.cfg["PWS_REFINEMENT"] = levels[case.grid_level]

        self.cfg["ICE_DROP_DIAM"] = case.drop_mvd * 1e6  # µm
        self.cfg["ICE_LIQ_H2O_CONTENT"] = case.lwc

        # …weiteres Mapping hier…
        return self.cfg

# ---------- 1. High-Level Spec ----------------------------------------------
@dataclass
class Case:
    mach: float
    flow: FlowState
    alpha_start: int
    alpha_end: int
    lwc: float                # liquid water content [kg/m³]
    drop_mvd: float = 20e-6   # default 20 µm
    grid_level: str = "coarse"  # coarse | medium | fine

# ---------- 2. Regeln / Ableitungen -----------------------------------------
def derive(case: Case, calculator=calculate_reynolds) -> dict:
    """Mappt High-Level-Spec auf FENSAP-Variablendschungel."""
    builder = ConfigBuilder(calculator)
    return builder.derive(case)

# ---------- 3. Rendern via Jinja-Template ------------------------------------
def render_fensap(cfg: dict, template_path="template.j2") -> str:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("."),
                             trim_blocks=True, lstrip_blocks=True)
    template = env.get_template(template_path)
    return template.render(**cfg)

# ---------- 4. Vollpipeline ---------------------------------------------------
def build(casefile: str, outfile="icing.def"):
    """Read a YAML case description and render the FENSAP configuration."""
    with open(casefile, "r", encoding="utf-8") as fh:
        case_dict = yaml.safe_load(fh)

    flow = FlowState(**case_dict.pop("flow"))
    case = Case(flow=flow, **case_dict)

    raw = derive(case)
    pathlib.Path(outfile).write_text(render_fensap(raw))
    print(f"✅  {outfile} geschrieben.")

if __name__ == "__main__":
    build("case.yaml")
