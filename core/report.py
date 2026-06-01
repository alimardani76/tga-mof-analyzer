"""
Report Generator
==================
Produces structured JSON output and suggested publication wording
for every analysis run.

Two outputs:
  1. JSON report — machine-readable, complete audit trail
  2. Suggested wording — copy-paste text for methods/results sections
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List


# ------------------------------------------------------------------
# JSON report
# ------------------------------------------------------------------

def save_report_json(
    result: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Save a complete analysis result as JSON.

    Parameters
    ----------
    result : dict
        The result dictionary from run_from_json or run_full_analysis.
    output_path : str
        Where to save.

    Returns
    -------
    str
        The path that was written.
    """
    # Add metadata
    result["_metadata"] = {
        "toolkit": "TGA-MOF-Analyzer",
        "version": "0.3",
        "generated": datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=_json_serializer)

    return output_path


def _json_serializer(obj):
    """Handle numpy types and other non-serializable objects."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return str(obj)
    return str(obj)


# ------------------------------------------------------------------
# Suggested wording generator
# ------------------------------------------------------------------

def generate_methods_wording(
    atmosphere: str = "air",
    heating_rate: Optional[float] = None,
    sample_mass_mg: Optional[float] = None,
    gas_flow: Optional[float] = None,
    temp_range: Optional[tuple] = None,
) -> str:
    """
    Generate suggested methods section text.

    Parameters
    ----------
    atmosphere : str
    heating_rate : float or None
    sample_mass_mg : float or None
    gas_flow : float or None
    temp_range : tuple or None

    Returns
    -------
    str
        Suggested methods paragraph.
    """
    parts = ["Thermogravimetric analysis (TGA) was performed"]

    if atmosphere:
        parts.append(f"under {atmosphere} atmosphere")

    if heating_rate:
        parts.append(f"at a heating rate of {heating_rate:.0f} °C/min")

    if gas_flow:
        parts.append(f"with a gas flow of {gas_flow:.0f} mL/min")

    if temp_range:
        parts.append(
            f"from {temp_range[0]:.0f} to {temp_range[1]:.0f} °C"
        )

    if sample_mass_mg:
        parts.append(f"using approximately {sample_mass_mg:.1f} mg of sample")

    text = " ".join(parts) + "."

    text += (
        " Data were analyzed using TGA-MOF-Analyzer v0.3, an open-source"
        " toolkit for automated TGA interpretation of metal–organic"
        " frameworks. Mass-loss events were detected using valley-to-valley"
        " segmentation of the derivative thermogravimetric (DTG) curve,"
        " ensuring 100% mass balance across all detected events."
    )

    return text


def generate_results_wording(
    result: Dict[str, Any],
) -> str:
    """
    Generate suggested results section text from analysis output.

    Parameters
    ----------
    result : dict
        Analysis result dictionary.

    Returns
    -------
    str
        Suggested results paragraph.
    """
    parts = []
    case_id = result.get("case_id", "the sample")

    # Module 1
    m1 = result.get("module_1", {})
    t_onset = m1.get("t_onset_tangent")
    if t_onset and t_onset == t_onset:  # not NaN
        parts.append(
            f"The decomposition onset temperature was determined to be"
            f" {t_onset:.1f} °C by the tangent intersection method."
        )

    # Residue analysis
    res = result.get("residue_analysis", {})
    obs = res.get("observed_residue_pct")
    pred = res.get("predicted_residue_pct")
    formula = res.get("formula")
    fw_mass = res.get("framework_mass")
    oxide = res.get("oxide_formula")
    delta = res.get("delta_pct")

    if obs is not None and pred is not None:
        parts.append(
            f"The observed TGA residue ({obs:.2f} wt%) was compared to the"
            f" predicted {oxide} residue ({pred:.2f} wt%) calculated from"
            f" the ideal guest-free framework formula"
            f" (MW = {fw_mass:.2f} g/mol)."
        )
        if delta is not None:
            if delta < 0:
                parts.append(
                    f"The deficit of {abs(delta):.1f} percentage points"
                    f" indicates excess organic content relative to the"
                    f" ideal stoichiometry, consistent with residual guests,"
                    f" modulator, or missing-linker defects."
                )
            elif delta > 2:
                parts.append(
                    f"The excess of {delta:.1f} percentage points above"
                    f" the predicted residue suggests inorganic impurity"
                    f" or incomplete organic combustion."
                )
            else:
                parts.append(
                    f"The close agreement (Δ = {delta:+.1f} pp) is consistent"
                    f" with near-ideal framework stoichiometry."
                )

    # Module 5
    m5_mof = result.get("w_mof_pct")
    additive = result.get("additive_name")
    if m5_mof is not None:
        parts.append(
            f"The MOF content in the composite was determined to be"
            f" {m5_mof:.1f} wt% by the residue ratio method, with the"
            f" balance attributed to {additive or 'the additive'}."
        )

    if not parts:
        return "(No results available for wording generation.)"

    return " ".join(parts)


def generate_full_report_text(
    result: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a complete human-readable report.

    Parameters
    ----------
    result : dict
    config : dict or None

    Returns
    -------
    str
    """
    lines = [
        "=" * 65,
        "TGA-MOF-Analyzer — Analysis Report",
        "=" * 65,
        f"Case: {result.get('case_id', 'unknown')}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Validation
    val = result.get("validation", {})
    if val.get("errors"):
        lines.append("VALIDATION ERRORS:")
        for e in val["errors"]:
            lines.append(f"  {e}")
        lines.append("")

    if val.get("warnings"):
        lines.append("VALIDATION WARNINGS:")
        for w in val["warnings"]:
            lines.append(f"  {w}")
        lines.append("")

    # Quality
    qual = result.get("quality", {})
    if qual:
        lines.append("DATA QUALITY:")
        lines.append(f"  Noise: {qual.get('noise_pct', '?'):.4f} wt%")
        lines.append(f"  Artifact: {'yes' if qual.get('has_artifact') else 'no'}")
        lines.append("")

    # Module 1
    m1 = result.get("module_1", {})
    if m1 and "error" not in m1:
        lines.append("THERMAL STABILITY:")
        lines.append(f"  T_onset (tangent): {m1.get('t_onset_tangent', '?')} °C")
        t_x = m1.get("t_x", {})
        for x, val in sorted(t_x.items(), key=lambda kv: int(kv[0])):
            lines.append(f"  T_{x}: {val} °C" if val else f"  T_{x}: not reached")
        lines.append(f"  Events: {m1.get('n_events', '?')}")
        lines.append(f"  Mass balance: {m1.get('mass_balance_pct', '?')}%")
        lines.append("")

    # Residue
    res = result.get("residue_analysis", {})
    if res and "error" not in res:
        lines.append("RESIDUE ANALYSIS:")
        lines.append(f"  Predicted: {res.get('predicted_residue_pct', '?')}%")
        lines.append(f"  Observed: {res.get('observed_residue_pct', '?')}%")
        lines.append(f"  Δ: {res.get('delta_pct', '?')} pp")
        lines.append(f"  {res.get('interpretation', '')}")
        lines.append("")

    # Suggested wording
    lines.append("-" * 65)
    lines.append("SUGGESTED WORDING (copy-paste for manuscript):")
    lines.append("-" * 65)
    lines.append("")

    atm = config.get("atmosphere", "air") if config else "air"
    hr_val = config.get("heating_rate") if config else None
    lines.append(generate_methods_wording(atmosphere=atm, heating_rate=hr_val))
    lines.append("")
    lines.append(generate_results_wording(result))
    lines.append("")
    lines.append("=" * 65)

    return "\n".join(lines)