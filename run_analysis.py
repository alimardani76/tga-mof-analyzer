"""
TGA-MOF-Analyzer v0.2 — Full Analysis Runner
================================================
Honest, physics-grounded analysis of all 14 cases.

No pass/fail judgments. No parameter tuning.
Reports what the math says. Generates plots.
The chemist decides what's real.

Usage:
    python run_analysis.py

Output:
    - Console report for every case
    - PNG plots saved to output/ folder
    - Summary tables at the end

Requirements:
    pip install numpy scipy matplotlib
"""

import numpy as np
import sys
import os
import json

# --- Paths ---
BASE_DATA_PATH = r"C:\Users\KaraPardazesh\Desktop\TGA"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.tga_parser import TGAData
from core.tga_quality import run_quality_checks
from core.dtg import compute_dtg, detect_events
from core.rexp import compute_rexp, compute_rexp_curve, find_dh_plateau, compute_linkers
from modules.m1_thermal_stability import analyze_stability
from modules.m2_guest_content import analyze_guest_content_windows
from modules.m3_composition import analyze_composition, MOFComponents
from modules.m4_defect_quantification import analyze_defects
from modules.m5_composite_loading import analyze_composite
from validation.cross_check import validate_mass_balance

try:
    from viz.plotting import (
        plot_tga_dtg,
        plot_rexp_curve,
        plot_composition_sensitivity,
        plot_event_map,
        plot_composite_series,
    )
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not found. Plots will be skipped.")
    print("   Install with: pip install matplotlib\n")


# ==================================================================
# MOF definitions — from literature, not from fitting
# ==================================================================

# UiO-66-NH2: Zr6O4(OH)4(NH2-BDC)6
# Node (dehydrated, after ~300°C): Zr6O6
# Linker: 2-amino-1,4-benzenedicarboxylate, C8H5NO4, MW=179.13
# Residue: 6 ZrO2
# SBU charge: +12 (6×Zr4+ with bridging O/OH)
# Ref: Kandiah et al., Chem. Mater. 2010, 22, 6632

UIO66_NH2 = MOFComponents(
    metal="Zr",
    n_metals_per_sbu=6,
    sbu_formula="Zr6O6",
    M_node=6 * 91.224 + 6 * 15.999,    # 643.34 g/mol
    linker_name="NH2-BDC",
    M_linker=179.130,                    # C8H5NO4(2-)
    linker_charge=-2.0,
    residue_formula="ZrO2",
    M_residue_per_metal=123.218,
    sbu_charge=12.0,
)

UIO66_NH2.full_formula = "Zr6O4(OH)4(NH2BDC)6"
# ==================================================================
# Data loader
# ==================================================================

def load_tga(case_folder, subfolder="cases"):
    """Load standardized TGA CSV from the data package."""
    csv_path = os.path.join(
        BASE_DATA_PATH, subfolder, case_folder,
        "tga_curve_standardized.csv"
    )
    if not os.path.exists(csv_path):
        return None

    temperatures = []
    masses = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                try:
                    temperatures.append(float(parts[1]))
                    masses.append(float(parts[2]))
                except ValueError:
                    continue

    if len(temperatures) < 10:
        return None

    return TGAData.from_arrays(
        temperature=np.array(temperatures),
        mass=np.array(masses),
        mass_is_mg=False,
        heating_rate=10.0,
        atmosphere="air",
    )


# ==================================================================
# Printing helpers
# ==================================================================

def section(title):
    print()
    print("=" * 75)
    print(f"  {title}")
    print("=" * 75)
    print()


def hr():
    print("  " + "-" * 71)


# ==================================================================
# Analysis functions — report honestly, never force results
# ==================================================================

def run_full_analysis(case_id, case_folder, label, components=None,
                      subfolder="cases", is_composite=False,
                      r_mof=None, r_additive=None, additive_name=None):
    """
    Run all applicable modules on a single case.
    Returns a dict of results for the summary table.
    """
    section(f"{label}  [{case_id}]")

    tga = load_tga(case_folder, subfolder=subfolder)
    if tga is None:
        print(f"  ❌ Data not found: {case_folder}")
        return None

    result = {
        'case_id': case_id,
        'label': label,
        'n_points': tga.n_points,
        't_range': tga.t_range,
        'residue_pct': tga.residue_pct,
        'total_loss_pct': tga.total_mass_loss_pct,
    }

    print(f"  Data: {tga.n_points} points, "
          f"T = {tga.t_range[0]:.1f}–{tga.t_range[1]:.1f}°C")
    print(f"  Initial mass: {tga.mass_pct[0]:.2f}%, "
          f"Residue: {tga.residue_pct:.2f}%, "
          f"Total loss: {tga.total_mass_loss_pct:.2f}%")

    # ----------------------------------------------------------
    # Quality checks
    # ----------------------------------------------------------
    print()
    print("  QUALITY CHECKS")
    hr()
    qr = run_quality_checks(tga)
    for check in qr.checks:
        icon = "✅" if check.passed else ("⚠️" if check.severity == "warning" else "❌")
        print(f"  {icon} {check.name}: {check.message}")
    result['quality_pass'] = qr.overall_pass
    result['noise'] = qr.noise_estimate_pct
    result['has_artifact'] = qr.has_initial_artifact

    # ----------------------------------------------------------
    # Module 1: Thermal Stability
    # ----------------------------------------------------------
    print()
    print("  MODULE 1: Thermal Stability")
    hr()

    stab = analyze_stability(tga)
    result['t_onset'] = stab.t_onset
    result['n_events'] = stab.events.n_events
    result['stability_window'] = stab.stability_window

    print(f"  T_onset (tangent method): {stab.t_onset:.1f}°C")
    print(f"  Stability window: {stab.stability_window[0]:.1f}–"
          f"{stab.stability_window[1]:.1f}°C")
    print(f"  Activated mass: {stab.activated_mass_pct:.2f}%")

    # T_x values
    for x in sorted(stab.t_x.keys()):
        val = stab.t_x[x]
        if not np.isnan(val):
            print(f"  T_{x}: {val:.1f}°C")

    # Events
    print(f"\n  Detected {stab.events.n_events} events "
          f"[total captured: {stab.events.total_captured_pct:.2f}%]:")
    if stab.events.artifact_event:
        ae = stab.events.artifact_event
        print(f"  ⚠️ Initial artifact flagged: "
              f"{ae.t_start:.0f}–{ae.t_end:.0f}°C ({ae.mass_loss_pct:.2f}%)")

    for i, ev in enumerate(stab.events.events, 1):
        art_flag = ""
        if (stab.events.artifact_event and
                ev.t_start == stab.events.artifact_event.t_start):
            art_flag = " ◄ artifact"
        print(f"    {i:>2}. {ev.t_start:>6.1f}–{ev.t_end:>5.1f}°C  "
              f"{ev.mass_loss_pct:>6.2f}%  "
              f"peak={ev.t_peak:.1f}°C  "
              f"DTG={ev.peak_dtg_value:.4f}{art_flag}")

    # Mass balance
    event_losses = [ev.mass_loss_pct for ev in stab.events.events]
    mb = validate_mass_balance(event_losses, tga.residue_pct)
    print(f"\n  Mass balance: {mb.message}")
    result['mass_balance'] = mb.passed

    # ----------------------------------------------------------
    # Module 2: Window-based mass loss
    # ----------------------------------------------------------
    print()
    print("  MODULE 2: Mass Loss by Temperature Windows")
    hr()
    print("  Math: mass_loss(T1→T2) = m(T1) − m(T2)")
    print("        where m(T) is linearly interpolated from TGA data")
    print()

    try:
        win = analyze_guest_content_windows(tga)
        for w in win.windows:
            print(f"    {w.name:<15s}  "
                  f"{w.mass_start_pct:>7.2f}% → {w.mass_end_pct:>6.2f}%  "
                  f"Δ = {w.mass_loss_pct:>6.2f}%")
        print(f"\n    Total loss:   {win.total_loss_pct:.2f}%")
        print(f"    Residue:      {win.residue_pct:.2f}%")
        print(f"    Sum check:    {win.mass_balance_pct:.2f}%")
        result['window_balance'] = win.mass_balance_pct
    except Exception as e:
        print(f"    Error: {e}")
    
# ----------------------------------------------------------
    # Residue Analysis (if formula provided)
    # ----------------------------------------------------------
    if components is not None and not is_composite:
        print()
        print("  RESIDUE ANALYSIS: Observed vs. Predicted")
        hr()

        try:
            from core.formula_parser import parse_formula, predict_residue, formula_mass
            from core.residue_analysis import analyze_residue

            # Use the hydroxylated formula for residue prediction
            # (matches the convention in the literature)
            fw_formula = getattr(components, 'full_formula', None)

            # If no full_formula attribute, construct from node + linkers
            if fw_formula is None:
                fw_formula = (
                    f"{components.sbu_formula}"
                    f"({components.linker_name}){components.ideal_linkers:.0f}"
                )

            fw_parsed = parse_formula(fw_formula)
            print(f"  Framework formula: {fw_formula}")
            print(f"  Parsed MW: {fw_parsed.molar_mass:.3f} g/mol")
            if fw_parsed.aliases_expanded:
                for alias in fw_parsed.aliases_expanded:
                    print(f"    Alias expanded: {alias}")

            pred = predict_residue(fw_formula, atmosphere=tga.atmosphere)
            print(f"\n  Expected oxide: {pred.oxide_formula}")
            print(f"  Oxide mass: {pred.oxide_mass:.2f} g/mol")
            print(f"  Predicted residue (guest-free): "
                  f"{pred.residue_pct_guest_free:.2f}%")

            
            res = analyze_residue(
                observed_residue_pct=tga.residue_pct,
                framework_formula=fw_formula,
                atmosphere=tga.atmosphere,
            )

            print(f"  Observed residue: {res.observed_residue_pct:.2f}%")
            print(f"  Δ = {res.delta_pct:+.2f} pp")
            print(f"  {res.delta_interpretation}")

            result['predicted_residue'] = pred.residue_pct_guest_free
            result['residue_delta'] = res.delta_pct

            if res.warnings:
                for w in res.warnings:
                    print(f"  ⚠️ {w}")

        except Exception as e:
            print(f"  Error: {e}")

    # ----------------------------------------------------------
    # Module 3: Composition (only if components provided)
    # ----------------------------------------------------------
    comp_result = None
    best_comp = None
    if components is not None and not is_composite:
        print()
        print("  MODULE 3: Composition / Molecular Formula")
        hr()
        print(f"  Math: q = (R_exp_DH × M_residue − M_node) / M_linker")
        print(f"        R_exp_DH = m(T_DH) / m_residue")
        print(f"        M_residue = {components.M_residue:.2f} g/mol  "
              f"({components.n_metals_per_sbu}×{components.residue_formula})")
        print(f"        M_node = {components.M_node:.2f} g/mol  "
              f"({components.sbu_formula})")
        print(f"        M_linker = {components.M_linker:.2f} g/mol  "
              f"({components.linker_name})")
        print(f"        Ideal q = {components.ideal_linkers:.1f}")
        print()

        # Auto-plateau finder report
        plateau = find_dh_plateau(tga)
        t_auto = plateau['suggested_temp']
        flat = plateau['flatness']
        rel = plateau['is_reliable']
        print(f"  Auto-plateau finder:")
        print(f"    Suggested T_DH: {t_auto:.1f}°C")
        print(f"    Flatness: {flat:.5f} "
              f"({'reliable' if rel else 'no true plateau — user judgment needed'})")
        if plateau['warning']:
            print(f"    ⚠️ {plateau['warning'][:100]}")

        # Sensitivity sweep — the KEY diagnostic
        print(f"\n  Sensitivity analysis (q vs T_DH):")
        print(f"    {'T_DH':>6}  {'m_DH%':>7}  {'R_exp_DH':>9}  "
              f"{'q':>7}  {'deficiency':>11}")
        hr()

        sweep_temps = np.arange(
            max(150, tga.t_range[0] + 50),
            min(500, tga.t_range[1] - 50),
            20,
        )

        best_comp = None
        for t_dh in sweep_temps:
            try:
                m_at_t = tga.get_mass_at_temp(float(t_dh))
                r_dh = m_at_t / tga.residue_pct
                q = compute_linkers(r_dh, components.M_residue,
                                    components.M_node, components.M_linker)
                ideal = components.ideal_linkers
                deficiency = (ideal - q) / ideal * 100

                marker = ""
                if abs(t_dh - t_auto) < 15:
                    marker = " ◄ auto-plateau"
                if ideal * 0.7 < q < ideal * 1.05:
                    marker += " ✓"

                print(f"    {t_dh:>6.0f}  {m_at_t:>7.2f}  {r_dh:>9.4f}  "
                      f"{q:>7.3f}  {deficiency:>10.1f}%{marker}")

                # Track the result at auto-plateau temp
                if abs(t_dh - t_auto) < 15 and best_comp is None:
                    comp_result = analyze_composition(
                        tga, components=components,
                        t_dh_plateau=float(t_dh),
                    )
                    best_comp = comp_result
            except Exception:
                pass

        # Report the auto-plateau composition
        if best_comp is not None:
            print(f"\n  Composition at auto-plateau ({t_auto:.0f}°C):")
            print(f"    q = {best_comp.q_linkers:.3f} linkers per FU")
            print(f"    R_exp_DH = {best_comp.r_exp_dh:.4f}")
            print(f"    Formula: {best_comp.formula_string}")
            print(f"    Charge: {best_comp.charge_balance}")
            result['q_auto'] = best_comp.q_linkers
            result['r_exp_dh_auto'] = best_comp.r_exp_dh
            result['t_dh_auto'] = t_auto
            result['plateau_reliable'] = rel

            if best_comp.warnings:
                print(f"\n    Warnings:")
                for w in best_comp.warnings:
                    short = w.replace('\n', ' ')[:100]
                    print(f"      ⚠️ {short}")

        # User guidance
        print(f"\n  ➡ The correct T_DH depends on YOUR knowledge of this MOF.")
        print(f"    Look at the DTG plot: the DH plateau is the flat region")
        print(f"    AFTER guest loss but BEFORE linker combustion.")
        print(f"    Use: analyze_composition(tga, components, t_dh_plateau=YOUR_VALUE)")

    # ----------------------------------------------------------
    # Module 4: Defects (only if Module 3 ran)
    # ----------------------------------------------------------
    if best_comp is not None:
        print()
        print("  MODULE 4: Missing Linker Defect Quantification")
        hr()
        print(f"  Math: x = (M_ideal_DH − R_exp_DH × M_res) / "
              f"(M_linker − n_cap × M_cap)")
        print(f"        Each compensator model gives a different x because")
        print(f"        the net mass change per defect depends on what")
        print(f"        fills the vacancy.")
        print()

        try:
            defects = analyze_defects(composition_result=best_comp)
            print(f"  Simple estimate: {defects.n_missing_simple:.3f} "
                  f"missing ({defects.defect_pct:.1f}%)")
            print(f"  Coordination: {defects.total_coordination_exp:.1f} / "
                  f"{defects.max_coordination:.0f} max")
            print()

            print(f"    {'Model':<22s}  {'x_miss':>7}  {'q_eff':>7}  "
                  f"{'n_cap':>7}  {'Chg':>4}")
            hr()
            for cr in defects.compensator_results:
                chg = "✅" if cr.charge_balanced else "❌"
                print(f"    {cr.model.name:<22s}  {cr.x_missing:>7.3f}  "
                      f"{cr.q_effective:>7.3f}  {cr.n_compensators:>7.2f}  "
                      f"{chg:>4}")

            print()
            print(f"  ⚠️ TGA CANNOT distinguish missing-linker from "
                  f"missing-cluster defects.")
            print(f"     Cross-validate with N₂ sorption or HRTEM.")

            result['defect_pct'] = defects.defect_pct
        except Exception as e:
            print(f"    Error: {e}")

    # ----------------------------------------------------------
    # Module 5: Composite Loading (if applicable)
    # ----------------------------------------------------------
    if is_composite and r_mof is not None:
        print()
        print("  MODULE 5: Composite Loading")
        hr()
        r_add = r_additive if r_additive is not None else 0.0
        if abs(r_add) < 0.001:
            print(f"  Math: w_MOF = r_composite / r_MOF")
            print(f"        (additive leaves no residue)")
        else:
            print(f"  Math: w_MOF = (r_comp − r_add) / (r_MOF − r_add)")
        print(f"        r_MOF = {r_mof:.4f}")
        print(f"        r_{additive_name} = {r_add:.4f}")
        print(f"        r_composite = {tga.residue_pct / 100:.4f}")
        print()

        try:
            comp_load = analyze_composite(
                tga_composite=tga,
                r_mof=r_mof,
                r_additive=r_add,
                additive_name=additive_name or "additive",
                additive_leaves_residue=(r_add > 0.001),
            )
            print(f"  MOF loading: {comp_load.w_mof_pct:.2f}%")
            print(f"  {additive_name}: {comp_load.w_additive_pct:.2f}%")
            result['w_mof_pct'] = comp_load.w_mof_pct
            result['w_add_pct'] = comp_load.w_additive_pct

            if comp_load.warnings:
                for w in comp_load.warnings:
                    print(f"  ⚠️ {w[:100]}")
        except Exception as e:
            print(f"    Error: {e}")

    # ----------------------------------------------------------
    # Plots
    # ----------------------------------------------------------
    if HAS_MATPLOTLIB:
        print()
        print("  GENERATING PLOTS")
        hr()

        prefix = os.path.join(OUTPUT_DIR, case_id)

        # Plot 1: TGA + DTG
        try:
            dtg_result = compute_dtg(tga)
            fig, _ = plot_tga_dtg(
                tga, dtg_result=dtg_result, events=stab.events,
                title=f"{label} — TGA / DTG",
                save_path=f"{prefix}_tga_dtg.png",
            )
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"    ✅ {case_id}_tga_dtg.png")
        except Exception as e:
            print(f"    ❌ TGA/DTG plot: {e}")

        # Plot 2: Event map
        try:
            fig, _ = plot_event_map(
                tga, stab.events,
                title=f"{label} — Event Map",
                save_path=f"{prefix}_events.png",
            )
            plt.close(fig)
            print(f"    ✅ {case_id}_events.png")
        except Exception as e:
            print(f"    ❌ Event map: {e}")

        # Plot 3: R_exp curve (only for pure MOFs)
        if components is not None and not is_composite:
            try:
                fig, _ = plot_rexp_curve(
                    tga, plateau_info=plateau,
                    title=f"{label} — R_exp(T)",
                    save_path=f"{prefix}_rexp.png",
                )
                plt.close(fig)
                print(f"    ✅ {case_id}_rexp.png")
            except Exception as e:
                print(f"    ❌ R_exp plot: {e}")

            # Plot 4: Composition sensitivity
            try:
                fig, _ = plot_composition_sensitivity(
                    tga, components,
                    title=f"{label} — q vs T_DH",
                    save_path=f"{prefix}_sensitivity.png",
                )
                plt.close(fig)
                print(f"    ✅ {case_id}_sensitivity.png")
            except Exception as e:
                print(f"    ❌ Sensitivity plot: {e}")

    return result


# ==================================================================
# Main
# ==================================================================

def main():
    section("TGA-MOF-Analyzer v0.2 — Honest Analysis Runner")
    print(f"  Data source: Crickmore & Bradshaw")
    print(f"  DOI: 10.5258/SOTON/D2004")
    print(f"  Base path: {BASE_DATA_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    if not HAS_MATPLOTLIB:
        print(f"  ⚠️ matplotlib not installed — no plots")
    print()
    print(f"  This tool reports what the math says.")
    print(f"  The chemist decides what's real.")

    all_results = []

    # ==============================================================
    # BLANKS
    # ==============================================================
    section("BLANKS")

    blank_ffp = load_tga("blank_01_FFP", subfolder="blank_controls")
    blank_alg = load_tga("blank_02_Ca_Alg", subfolder="blank_controls")

    r_ffp = max(blank_ffp.residue_pct / 100.0, 0.0) if blank_ffp else 0.0
    r_alg = blank_alg.residue_pct / 100.0 if blank_alg else 0.089

    if blank_ffp:
        print(f"  FFP blank: residue = {blank_ffp.residue_pct:.2f}%  "
              f"→ r_FFP = {r_ffp:.4f}")
    if blank_alg:
        print(f"  Ca-Alginate blank: residue = {blank_alg.residue_pct:.2f}%  "
              f"→ r_Alg = {r_alg:.4f}")

    # ==============================================================
    # PURE MOF POWDERS
    # ==============================================================

    # Case 01
    r = run_full_analysis(
        "case_01", "case_01_UiO66NH2",
        "UiO-66-NH2 (PSM source)", components=UIO66_NH2)
    if r:
        all_results.append(r)
        r_mof_01 = r['residue_pct'] / 100.0
    else:
        r_mof_01 = 0.3288

    # Case 02 — PSM, unknown modified formula
    r = run_full_analysis(
        "case_02", "case_02_PSM",
        "Post-Synthetic Modification")
    if r:
        all_results.append(r)

    # Case 03
    r = run_full_analysis(
        "case_03", "case_03_UiO66NH2_2_FFP",
        "UiO-66-NH2 (FFP source)", components=UIO66_NH2)
    if r:
        all_results.append(r)

    # Case 09
    r = run_full_analysis(
        "case_09", "case_09_UiO66NH2_2_ALG",
        "UiO-66-NH2 (ALG source)", components=UIO66_NH2)
    if r:
        all_results.append(r)
        r_mof_09 = r['residue_pct'] / 100.0
    else:
        r_mof_09 = 0.2964

    # ==============================================================
    # FFP COMPOSITES
    # ==============================================================

    ffp_labels = []
    ffp_mof_pcts = []

    for case_id, folder, label in [
        ("case_04", "case_04_UiO66NH2_FFP_5cycle", "UiO66-NH2@FFP 5 cycles"),
        ("case_05", "case_05_UiO66NH2_FFP_10cycle", "UiO66-NH2@FFP 10 cycles"),
        ("case_06", "case_06_UiO66NH2_FFP_15cycle", "UiO66-NH2@FFP 15 cycles"),
        ("case_07", "case_07_UiO66NH2_FFP_20cycle", "UiO66-NH2@FFP 20 cycles"),
        ("case_08", "case_08_UiO66NH3_FFP", "UiO66-NH3+@FFP"),
    ]:
        r = run_full_analysis(
            case_id, folder, label,
            is_composite=True, r_mof=r_mof_01,
            r_additive=r_ffp, additive_name="FFP")
        if r:
            all_results.append(r)
            if r.get('w_mof_pct') is not None:
                ffp_labels.append(label.split("@")[1] if "@" in label else label)
                ffp_mof_pcts.append(r['w_mof_pct'])

    # ==============================================================
    # ALGINATE COMPOSITES
    # ==============================================================

    alg_labels = []
    alg_mof_pcts = []

    for case_id, folder, label in [
        ("case_10", "case_10_UiO66NH2_Alg", "UiO66-NH2@Alginate"),
        ("case_11", "case_11_UiO66NH3_Alg", "UiO66-NH3+@Alginate"),
    ]:
        r = run_full_analysis(
            case_id, folder, label,
            is_composite=True, r_mof=r_mof_09,
            r_additive=r_alg, additive_name="Ca-Alginate")
        if r:
            all_results.append(r)
            if r.get('w_mof_pct') is not None:
                alg_labels.append(label)
                alg_mof_pcts.append(r['w_mof_pct'])

    # ==============================================================
    # Zn-MOFs — Module 1 + 2 only
    # ==============================================================

    for case_id, folder, label in [
        ("case_12", "case_12_Zn-MOF-1", "Zn-MOF-1"),
        ("case_13", "case_13_Zn-LMOF-2", "Zn-LMOF-2"),
        ("case_14", "case_14_Zn-LMOF-3", "Zn-LMOF-3"),
    ]:
        r = run_full_analysis(case_id, folder, label)
        if r:
            all_results.append(r)

    # ==============================================================
    # COMPOSITE SERIES PLOTS
    # ==============================================================

    if HAS_MATPLOTLIB and ffp_mof_pcts:
        import matplotlib.pyplot as plt
        try:
            fig, _ = plot_composite_series(
                ffp_labels, ffp_mof_pcts,
                additive_name="FFP",
                title="FFP Composite Loading Series",
                save_path=os.path.join(OUTPUT_DIR, "ffp_series.png"),
            )
            plt.close(fig)
            print(f"\n  ✅ ffp_series.png")
        except Exception as e:
            print(f"\n  ❌ FFP series plot: {e}")

    if HAS_MATPLOTLIB and alg_mof_pcts:
        import matplotlib.pyplot as plt
        try:
            fig, _ = plot_composite_series(
                alg_labels, alg_mof_pcts,
                additive_name="Ca-Alginate",
                title="Alginate Composite Loading",
                save_path=os.path.join(OUTPUT_DIR, "alg_series.png"),
            )
            plt.close(fig)
            print(f"  ✅ alg_series.png")
        except Exception as e:
            print(f"  ❌ Alginate series plot: {e}")

    # ==============================================================
    # SUMMARY TABLES
    # ==============================================================

    section("SUMMARY TABLE: All Cases")

    print(f"  {'Case':<12} {'Label':<30} {'Res%':>6} {'T_ons':>6} "
          f"{'Nevt':>5} {'MB':>3} {'Qual':>4} {'q':>7} {'MOF%':>7}")
    print(f"  {'-'*12} {'-'*30} {'-'*6} {'-'*6} "
          f"{'-'*5} {'-'*3} {'-'*4} {'-'*7} {'-'*7}")

    for r in all_results:
        mb = "✅" if r.get('mass_balance', False) else "❌"
        qual = "✅" if r.get('quality_pass', False) else "⚠️"
        q_str = f"{r['q_auto']:.3f}" if r.get('q_auto') is not None else "    —"
        mof_str = f"{r['w_mof_pct']:.1f}" if r.get('w_mof_pct') is not None else "    —"
        t_onset = r.get('t_onset', float('nan'))
        t_str = f"{t_onset:.1f}" if not np.isnan(t_onset) else "  N/A"
        n_events = r.get('n_events', 0)

        print(f"  {r['case_id']:<12} {r['label']:<30} "
              f"{r['residue_pct']:>6.2f} {t_str:>6} "
              f"{n_events:>5d} {mb:>3} {qual:>4} "
              f"{q_str:>7} {mof_str:>7}")

    # ==============================================================
    # PHYSICS NOTES
    # ==============================================================

    section("NOTES ON THE PHYSICS")

    print("""
  1. MODULE 5 (Composite Loading) is the most robust module.
     It depends ONLY on residue masses — no event detection,
     no plateau selection.  The equation is exact:
       w_MOF = (r_comp − r_add) / (r_MOF − r_add)
     Limitation: r_MOF ≈ r_add → denominator → 0 → unreliable.

  2. MODULE 3 (Composition) is sensitive to the DH plateau choice.
     The sensitivity plots show EXACTLY how q varies with T_DH.
     There is no single "correct" T_DH — it depends on whether
     guests are fully removed at that temperature.
     For as-synthesized MOFs with heavy solvation, there may be
     NO true plateau.  The R_exp curve makes this visible.

  3. MODULE 4 (Defects) inherits Module 3's uncertainty AND adds
     the compensator ambiguity.  TGA alone cannot determine what
     caps the defect sites.  NMR, CHNO, or ICP are needed.

  4. MODULE 1 (Stability) T_onset depends on which event is
     identified as "decomposition."  For composites where the
     support decomposes at a different temperature than the MOF,
     T_onset reflects the dominant component.

  5. The valley-to-valley event detection guarantees 100% mass
     balance but does NOT guarantee chemically meaningful events.
     Events should be interpreted with DTG peak inspection.

  6. All equations in this toolkit are traceable to:
     - Abánades Lázaro, Eur. J. Inorg. Chem. 2020, 4284
     - Shearer et al., Chem. Mater. 2016, 28, 3749
     - Sannes et al., Chem. Mater. 2023, 35, 3793
     """)

    section("ANALYSIS COMPLETE")
    print(f"  Cases analyzed: {len(all_results)}")
    print(f"  Plots saved to: {OUTPUT_DIR}")
    print(f"  This tool reports what the math says.")
    print(f"  The chemist decides what's real.")


if __name__ == "__main__":
    main()