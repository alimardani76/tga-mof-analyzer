"""
JSON-Driven Analysis Runner
=============================
Reads a JSON configuration, validates inputs, routes to
the appropriate modules, and produces structured output.

Input format: JSON with a "runs" array, each run containing
fields like tga_csv, formula, window_start, etc.

Output: For each run, a result dict with all computed values,
warnings, and metadata. Optionally saved as JSON.

This is the bridge between the user's input and our engine.
"""

import json
import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.tga_parser import TGAData
from core.tga_quality import run_quality_checks
from core.validator import validate_run_config, ValidationResult
from core.dtg import compute_dtg, detect_events
from core.uncertainty import (
    sigma_from_quality,
    guest_count_uncertainty,
    composition_uncertainty,
    composite_loading_uncertainty,
)


def run_from_json(
    json_path: str,
    output_dir: Optional[str] = None,
    save_json: bool = True,
    generate_plots: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run analysis from a JSON configuration file.

    Parameters
    ----------
    json_path : str
        Path to the JSON config file.
    output_dir : str or None
        Where to save outputs. If None, saves next to JSON.
    save_json : bool
        Whether to save result JSON files.
    generate_plots : bool
        Whether to generate plot PNGs.

    Returns
    -------
    list of dict
        One result dict per run in the JSON.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    runs = config.get("runs", [config])  # support single run or array
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(json_path))

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for i, run_config in enumerate(runs):
        case_id = run_config.get("case_id", f"run_{i:03d}")
        print(f"\n{'='*60}")
        print(f"  Run: {case_id}")
        print(f"{'='*60}")

        # --- Validate ---
        validation = validate_run_config(run_config)
        print(validation.summary())

        result = {
            "case_id": case_id,
            "validation": {
                "is_valid": validation.is_valid,
                "errors": [str(m) for m in validation.errors],
                "warnings": [str(m) for m in validation.warnings],
            },
            "timestamp": datetime.now().isoformat(),
        }

        if not validation.is_valid:
            result["status"] = "FAILED_VALIDATION"
            all_results.append(result)
            continue

        # --- Load data ---
        tga_csv = run_config.get("tga_csv") or run_config.get("input")
        try:
            tga = TGAData.from_file(
                tga_csv,
                heating_rate=run_config.get("heating_rate"),
                atmosphere=run_config.get("atmosphere", "unknown"),
            )
        except Exception as e:
            result["status"] = f"LOAD_ERROR: {e}"
            all_results.append(result)
            continue

        # --- Quality checks ---
        qr = run_quality_checks(tga)
        sigmas = sigma_from_quality(qr)
        result["quality"] = {
            "overall_pass": qr.overall_pass,
            "noise_pct": qr.noise_estimate_pct,
            "has_artifact": qr.has_initial_artifact,
            "sigma_mass_pct": sigmas["sigma_mass_pct"],
            "sigma_residue_pct": sigmas["sigma_residue_pct"],
        }

        # --- Module 1: Stability ---
        try:
            from modules.m1_thermal_stability import analyze_stability
            stab = analyze_stability(tga)
            result["module_1"] = {
                "t_onset_tangent": round(stab.t_onset, 2),
                "stability_window": [
                    round(stab.stability_window[0], 1),
                    round(stab.stability_window[1], 1),
                ],
                "activated_mass_pct": round(stab.activated_mass_pct, 2),
                "n_events": stab.events.n_events,
                "mass_balance_pct": round(
                    sum(e.mass_loss_pct for e in stab.events.events) + tga.residue_pct, 2
                ),
                "residue_pct": round(tga.residue_pct, 2),
                "t_x": {str(k): round(v, 1) if not np.isnan(v) else None
                        for k, v in stab.t_x.items()},
            }
        except Exception as e:
            result["module_1"] = {"error": str(e)}

        # --- Module 2: Windows ---
        try:
            from modules.m2_guest_content import analyze_guest_content_windows
            win = analyze_guest_content_windows(tga)
            result["module_2"] = {
                "windows": [
                    {
                        "name": w.name,
                        "t_min": w.t_min,
                        "t_max": w.t_max,
                        "loss_pct": round(w.mass_loss_pct, 3),
                        "sigma_loss_pct": round(sigmas["sigma_window_loss_pct"], 4),
                    }
                    for w in win.windows
                ],
                "total_loss_pct": round(win.total_loss_pct, 3),
                "mass_balance_pct": round(win.mass_balance_pct, 3),
            }
        except Exception as e:
            result["module_2"] = {"error": str(e)}

        # --- Residue prediction + Module 3/4 (if formula provided) ---
        formula = run_config.get("formula")
        if formula:
            try:
                from core.formula_parser import parse_formula, predict_residue
                from core.residue_analysis import analyze_residue

                parsed = parse_formula(formula)
                pred = predict_residue(formula, atmosphere=tga.atmosphere)
                res = analyze_residue(
                    tga.residue_pct, formula, atmosphere=tga.atmosphere,
                )

                result["residue_analysis"] = {
                    "formula": formula,
                    "framework_mass": round(parsed.molar_mass, 3),
                    "predicted_residue_pct": round(pred.residue_pct_guest_free, 2),
                    "observed_residue_pct": round(tga.residue_pct, 2),
                    "delta_pct": round(res.delta_pct, 2),
                    "interpretation": res.delta_interpretation,
                    "oxide_formula": pred.oxide_formula,
                }
            except Exception as e:
                result["residue_analysis"] = {"error": str(e)}

        result["status"] = "COMPLETE"
        result["input_config"] = run_config
        all_results.append(result)

        # --- Save individual result ---
        if save_json:
            out_path = os.path.join(output_dir, f"{case_id}_result.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

    return all_results