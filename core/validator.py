"""
Input Validation
=================
Semantic checks on all user inputs before computation.

Design: validate EVERYTHING before running ANYTHING.
Return a list of errors and warnings — never crash.

Error levels:
  - error:   Cannot proceed. Input is invalid.
  - warning: Can proceed but results may be unreliable.
  - info:    FYI — no action needed.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os


@dataclass
class ValidationMessage:
    """A single validation result."""
    level: str       # 'error', 'warning', 'info'
    field: str       # which input field
    message: str     # human-readable explanation

    def __repr__(self) -> str:
        icons = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}
        icon = icons.get(self.level, "?")
        return f"{icon} [{self.field}] {self.message}"


@dataclass
class ValidationResult:
    """Complete validation output."""
    messages: List[ValidationMessage] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(m.level == "error" for m in self.messages)

    @property
    def errors(self) -> List[ValidationMessage]:
        return [m for m in self.messages if m.level == "error"]

    @property
    def warnings(self) -> List[ValidationMessage]:
        return [m for m in self.messages if m.level == "warning"]

    def summary(self) -> str:
        if not self.messages:
            return "✅ All inputs valid."
        lines = ["Input Validation:"]
        for m in self.messages:
            lines.append(f"  {m}")
        n_err = len(self.errors)
        n_warn = len(self.warnings)
        if n_err > 0:
            lines.append(f"\n  ❌ {n_err} error(s) — cannot proceed.")
        elif n_warn > 0:
            lines.append(f"\n  ⚠️ {n_warn} warning(s) — proceed with caution.")
        else:
            lines.append(f"\n  ✅ All checks passed.")
        return "\n".join(lines)


def validate_run_config(config: Dict[str, Any]) -> ValidationResult:
    """
    Validate a JSON run configuration.

    Parameters
    ----------
    config : dict
        Run configuration with fields matching the JSON schema.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    # --- Required: TGA data path ---
    tga_csv = config.get("tga_csv") or config.get("input")
    if not tga_csv:
        result.messages.append(ValidationMessage(
            "error", "tga_csv",
            "TGA data file path is required."
        ))
    elif not os.path.exists(tga_csv):
        result.messages.append(ValidationMessage(
            "error", "tga_csv",
            f"File not found: {tga_csv}"
        ))

    # --- Temperature windows ---
    w_start = config.get("window_start")
    w_end = config.get("window_end")
    if w_start is not None and w_end is not None:
        if w_start >= w_end:
            result.messages.append(ValidationMessage(
                "error", "window_start/window_end",
                f"Window start ({w_start}°C) must be below "
                f"window end ({w_end}°C)."
            ))

    decomp_start = config.get("decomp_start")
    if decomp_start is not None and w_end is not None:
        if w_end > decomp_start:
            result.messages.append(ValidationMessage(
                "warning", "window_end",
                f"Guest-loss window ({w_start}–{w_end}°C) extends "
                f"beyond decomposition onset ({decomp_start}°C). "
                f"Mass loss in this interval includes framework "
                f"decomposition and should NOT be interpreted as "
                f"clean guest loss."
            ))

    # --- Formula ---
    formula = config.get("formula")
    if formula:
        try:
            from core.formula_parser import parse_formula
            parsed = parse_formula(formula)
            if not parsed.element_counts:
                result.messages.append(ValidationMessage(
                    "error", "formula",
                    f"Formula '{formula}' parsed but contains no elements."
                ))
        except ValueError as e:
            result.messages.append(ValidationMessage(
                "error", "formula",
                f"Cannot parse formula '{formula}': {e}"
            ))
    else:
        result.messages.append(ValidationMessage(
            "info", "formula",
            "No formula provided. Modules 3 and 4 (composition, "
            "defects) will be skipped. Modules 1, 2, 5 can still run."
        ))

    # --- Guests ---
    guests_str = config.get("guests")
    if guests_str:
        guest_list = [g.strip() for g in guests_str.split(",")]
        for g in guest_list:
            try:
                from core.formula_parser import parse_formula
                parse_formula(g)
            except ValueError:
                result.messages.append(ValidationMessage(
                    "error", "guests",
                    f"Unknown guest '{g}'. Provide as a chemical formula "
                    f"(e.g., 'C3H7NO' for DMF) or add to FORMULA_ALIASES."
                ))

    # --- Atmosphere ---
    atmosphere = config.get("atmosphere", "")
    if not atmosphere or atmosphere.lower() in ("unknown", ""):
        result.messages.append(ValidationMessage(
            "warning", "atmosphere",
            "Atmosphere not specified. Residue interpretation "
            "assumes oxidative conditions (air). If TGA was run "
            "under N₂/Ar, oxide residue predictions may be invalid."
        ))

    # --- Observed final mass ---
    obs_mass = config.get("observed_final_mass")
    if obs_mass is not None:
        if obs_mass < -5:
            result.messages.append(ValidationMessage(
                "error", "observed_final_mass",
                f"Observed final mass ({obs_mass}%) is strongly negative. "
                f"Check data or units."
            ))
        elif obs_mass < 0:
            result.messages.append(ValidationMessage(
                "warning", "observed_final_mass",
                f"Observed final mass ({obs_mass}%) is slightly negative. "
                f"Possible baseline drift in TGA."
            ))
    else:
        result.messages.append(ValidationMessage(
            "info", "observed_final_mass",
            "No observed final mass provided. Residue comparison "
            "will use the last data point from the TGA curve."
        ))

    # --- Heating rate ---
    hr = config.get("heating_rate")
    if hr is not None:
        if hr <= 0:
            result.messages.append(ValidationMessage(
                "error", "heating_rate",
                f"Heating rate ({hr}°C/min) must be positive."
            ))
        elif hr > 50:
            result.messages.append(ValidationMessage(
                "warning", "heating_rate",
                f"Heating rate ({hr}°C/min) is unusually high. "
                f"Thermal lag may significantly shift T_onset."
            ))
    else:
        result.messages.append(ValidationMessage(
            "info", "heating_rate",
            "Heating rate not specified. T_onset values cannot be "
            "compared across different heating rates."
        ))

    # --- Numeric range checks ---
    for field_name in ("mass_loss", "tolerance", "count_step", "max_count"):
        val = config.get(field_name)
        if val is not None and val < 0:
            result.messages.append(ValidationMessage(
                "error", field_name,
                f"{field_name} ({val}) must be non-negative."
            ))

    return result