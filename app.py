"""
TGA-MOF-Analyzer — Streamlit Web Interface
=============================================
Run locally:  streamlit run app.py
Deploy:       Streamlit Community Cloud
"""

import streamlit as st
import numpy as np
import json
import os
import sys
import tempfile
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from core.tga_parser import TGAData
from core.tga_quality import run_quality_checks
from core.formula_parser import parse_formula, predict_residue, FORMULA_ALIASES
from core.residue_analysis import analyze_residue
from core.validator import validate_run_config
from core.uncertainty import sigma_from_quality
from core.dtg import compute_dtg, detect_events
from modules.m1_thermal_stability import analyze_stability
from modules.m2_guest_content import analyze_guest_content_windows
from core.report import generate_full_report_text


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="TGA-MOF-Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Helper: plot to bytes for download
# ------------------------------------------------------------------

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

st.sidebar.title("🔬 TGA-MOF-Analyzer")
st.sidebar.markdown("*Automated TGA analysis for MOFs*")
st.sidebar.markdown("---")

# --- 1. File upload ---
st.sidebar.header("1. Upload TGA Data")
uploaded_file = st.sidebar.file_uploader(
    "TGA data file (CSV or TXT)",
    type=["csv", "txt", "tsv", "dat"],
    help="CSV with temperature and mass columns. Format auto-detected.",
)

# --- 2. Sample info ---
st.sidebar.header("2. Sample Information")

case_id = st.sidebar.text_input(
    "Sample name",
    value="my_sample",
    help="Names your output files.",
)

formula_presets = {
    "(custom or none)": "",
    "UiO-66": "Zr6O4(OH)4(BDC)6",
    "UiO-66-NH2": "Zr6O4(OH)4(NH2BDC)6",
    "UiO-67": "Zr6O4(OH)4(BPDC)6",
    "MOF-5": "Zn4O(BDC)3",
    "HKUST-1": "Cu3(BTC)2",
    "ZIF-8": "Zn(MeIM)2",
    "MIL-53(Al)": "Al(OH)(BDC)",
    "MIL-101(Cr)": "Cr3O(OH)(BDC)3",
}

preset = st.sidebar.selectbox(
    "MOF type",
    list(formula_presets.keys()),
    index=0,
)

if preset == "(custom or none)":
    formula = st.sidebar.text_input(
        "Framework formula (guest-free)",
        value="",
        placeholder="e.g., Zr6O4(OH)4(BDC)6",
        help="Leave blank to skip composition analysis.",
    )
else:
    formula = formula_presets[preset]
    st.sidebar.code(formula, language=None)

# Live formula validation
formula_valid = False
if formula:
    try:
        parsed_formula = parse_formula(formula)
        st.sidebar.success(f"MW = {parsed_formula.molar_mass:.2f} g/mol")
        if parsed_formula.aliases_expanded:
            for a in parsed_formula.aliases_expanded:
                st.sidebar.caption(f"Alias: {a}")
        formula_valid = True
    except ValueError as e:
        st.sidebar.error(f"Parse error: {e}")
        formula = ""

atmosphere = st.sidebar.selectbox(
    "Atmosphere",
    ["air", "N2", "Ar", "O2", "unknown"],
    index=0,
)

heating_rate = st.sidebar.number_input(
    "Heating rate (°C/min)",
    min_value=0.1,
    max_value=100.0,
    value=10.0,
    step=1.0,
)

# --- 3. Advanced ---
with st.sidebar.expander("Advanced Options"):
    guests = st.text_input(
        "Guest candidates",
        value="H2O,DMF",
        help="Comma-separated: H2O,DMF,EtOH,MeOH,DEF,THF,acetone,DMSO",
    )
    smooth_window = st.slider(
        "Smoothing window",
        min_value=5,
        max_value=101,
        value=31,
        step=2,
        help="Savitzky-Golay window (must be odd).",
    )


# ------------------------------------------------------------------
# Main area: landing page
# ------------------------------------------------------------------

st.title("🔬 TGA-MOF-Analyzer")

if not uploaded_file:
    st.markdown(
        "Upload a TGA file → get thermal stability, composition, "
        "and publication-quality plots."
    )
    st.markdown("---")

    st.markdown("### How to use")
    st.markdown("""
    1. **Upload** your TGA data file (CSV) in the sidebar
    2. **Select** your MOF type or enter a custom formula
    3. **Click** "Run Analysis"
    4. **Download** results, plots, and suggested wording
    """)

    st.markdown("### Supported TGA formats")
    st.markdown("""
    Any CSV or TXT with **temperature** and **mass** columns.
    The parser auto-detects delimiter, headers, units, and column order.

    | Format | Supported |
    |--------|-----------|
    | Comma-separated (CSV) | ✅ |
    | Tab-separated (TSV) | ✅ |
    | Semicolon-separated | ✅ |
    | Temperature in °C or K | ✅ |
    | Mass in wt% or mg | ✅ |
    | Multiple header rows | ✅ (skipped automatically) |
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🌡️ Thermal Stability**")
        st.caption("T_onset (3 methods), T_1/T_2/T_5/T_10, stability window")
    with col2:
        st.markdown("**🧪 Composition**")
        st.caption("Residue prediction, formula verification, defect estimate")
    with col3:
        st.markdown("**📈 Plots**")
        st.caption("TGA+DTG overlay, event map — all publication-ready")

    with st.expander("📖 Available formula aliases (click to expand)"):
        alias_data = [
            {"Alias": name, "Formula": form}
            for name, form in sorted(FORMULA_ALIASES.items())
        ]
        st.dataframe(alias_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        "*Based on: Abánades Lázaro, Eur. J. Inorg. Chem. 2020; "
        "Shearer et al., Chem. Mater. 2016; "
        "Sannes et al., Chem. Mater. 2023*"
    )

    st.stop()


# ------------------------------------------------------------------
# Run button
# ------------------------------------------------------------------

run_button = st.sidebar.button(
    "🚀 Run Analysis",
    type="primary",
    use_container_width=True,
)

if run_button:
    st.session_state["run_triggered"] = True

if not st.session_state.get("run_triggered"):
    st.info("👈 Configure your analysis in the sidebar, then click **Run Analysis**.")
    st.stop()


# ------------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------------

# Save uploaded file to temp
with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
    tmp.write(uploaded_file.getvalue())
    tmp_path = tmp.name

try:
    # ============================
    # LOAD DATA
    # ============================
    with st.spinner("Loading TGA data..."):
        tga = TGAData.from_file(
            tmp_path,
            heating_rate=heating_rate,
            atmosphere=atmosphere,
        )

    # ============================
    # DATA OVERVIEW
    # ============================
    st.header("📊 Data Overview")
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Data Points", f"{tga.n_points}")
    ov2.metric(
        "T Range",
        f"{tga.temperature[0]:.0f}–{tga.temperature[-1]:.0f} °C",
    )
    ov3.metric("Total Loss", f"{tga.total_mass_loss_pct:.2f}%")
    ov4.metric("Residue", f"{tga.residue_pct:.2f}%")

    # ============================
    # QUALITY CHECKS
    # ============================
    st.header("✅ Quality Checks")
    qr = run_quality_checks(tga)
    sigmas = sigma_from_quality(qr)

    n_checks = len(qr.checks)
    n_cols = min(n_checks, 4)
    qc_cols = st.columns(n_cols)
    for i, check in enumerate(qr.checks):
        with qc_cols[i % n_cols]:
            if check.passed:
                st.markdown(f"✅ **{check.name}**")
            else:
                st.markdown(f"⚠️ **{check.name}**")
            # Show value if available
            if hasattr(check, "value") and check.value is not None:
                st.caption(f"{check.value}")

    st.caption(f"Noise estimate: {qr.noise_estimate_pct:.4f} wt%")

    # ============================
    # TGA + DTG PLOT
    # ============================
    st.header("📈 TGA + DTG")

    dtg_result = compute_dtg(tga)

    fig_tga, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(tga.temperature, tga.mass_pct, "b-", linewidth=1.5, label="TGA")
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.set_ylabel("Mass (%)", color="b", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="b")
    y_min = max(0, tga.mass_pct[-1] - 5)
    ax1.set_ylim(bottom=y_min)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        dtg_result.temperature, dtg_result.dtg,
        "r-", linewidth=1, alpha=0.7, label="DTG",
    )
    ax2.set_ylabel("DTG (%/°C)", color="r", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="r")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig_tga.tight_layout()
    st.pyplot(fig_tga)

    # Download button for plot
    st.download_button(
        "📥 Download TGA+DTG plot",
        data=fig_to_bytes(fig_tga),
        file_name=f"{case_id}_tga_dtg.png",
        mime="image/png",
    )
    plt.close(fig_tga)

    # ============================
    # EVENT DETECTION
    # ============================
    st.header("🔍 Mass-Loss Events")

    event_result = detect_events(tga, dtg_result=dtg_result)

    events_table = []
    for i, ev in enumerate(event_result.events, 1):
        events_table.append({
            "Event": i,
            "T_start (°C)": round(ev.t_start, 1),
            "T_end (°C)": round(ev.t_end, 1),
            "T_peak (°C)": round(ev.t_peak, 1),
            "Mass Loss (%)": round(ev.mass_loss_pct, 2),
            "Peak DTG": round(ev.peak_dtg_value, 4),
        })
    st.dataframe(events_table, use_container_width=True, hide_index=True)

    total_ev = sum(e.mass_loss_pct for e in event_result.events)
    mb = total_ev + tga.residue_pct
    st.success(
        f"**Mass balance:** events ({total_ev:.2f}%) + "
        f"residue ({tga.residue_pct:.2f}%) = {mb:.2f}%"
    )

    # Event map plot
    fig_ev, ax_ev = plt.subplots(figsize=(10, 4))
    n_ev = len(event_result.events)
    colors = plt.cm.Set3(np.linspace(0, 1, max(n_ev, 1)))

    for i, ev in enumerate(event_result.events):
        mask = (tga.temperature >= ev.t_start) & (tga.temperature <= ev.t_end)
        if mask.any():
            ax_ev.fill_between(
                tga.temperature[mask],
                tga.mass_pct[mask],
                tga.mass_pct[mask].min(),
                alpha=0.5,
                color=colors[i],
                label=f"E{i+1}: {ev.mass_loss_pct:.1f}%",
            )
    ax_ev.plot(tga.temperature, tga.mass_pct, "k-", linewidth=1)
    ax_ev.set_xlabel("Temperature (°C)", fontsize=12)
    ax_ev.set_ylabel("Mass (%)", fontsize=12)
    ax_ev.set_title("Mass-Loss Event Map", fontsize=13)
    ax_ev.grid(True, alpha=0.3)
    if n_ev <= 12:
        ax_ev.legend(fontsize=7, ncol=3, loc="upper right")
    fig_ev.tight_layout()
    st.pyplot(fig_ev)

    st.download_button(
        "📥 Download Event Map",
        data=fig_to_bytes(fig_ev),
        file_name=f"{case_id}_events.png",
        mime="image/png",
    )
    plt.close(fig_ev)

    # ============================
    # MODULE 1: STABILITY
    # ============================
    st.header("🌡️ Module 1: Thermal Stability")

    result = {"case_id": case_id}

    try:
        stab = analyze_stability(tga)

        s1, s2, s3 = st.columns(3)
        s1.metric("T_onset (tangent)", f"{stab.t_onset:.1f} °C")
        t_lo, t_hi = stab.stability_window
        if not np.isnan(t_lo) and not np.isnan(t_hi):
            s2.metric("Stability Window", f"{t_lo:.0f}–{t_hi:.0f} °C")
        else:
            s2.metric("Stability Window", "N/A")
        s3.metric("Activated Mass", f"{stab.activated_mass_pct:.2f}%")

        # T_x metrics
        tx_keys = sorted(stab.t_x.keys())
        tx_cols = st.columns(len(tx_keys))
        for i, x in enumerate(tx_keys):
            v = stab.t_x[x]
            with tx_cols[i]:
                if not np.isnan(v):
                    st.metric(f"T_{x}", f"{v:.1f} °C")
                else:
                    st.metric(f"T_{x}", "—")

        result["module_1"] = {
            "t_onset_tangent": round(stab.t_onset, 2),
            "stability_window": [
                round(t_lo, 1) if not np.isnan(t_lo) else None,
                round(t_hi, 1) if not np.isnan(t_hi) else None,
            ],
            "activated_mass_pct": round(stab.activated_mass_pct, 2),
            "n_events": stab.events.n_events,
            "t_x": {
                str(k): round(v, 1) if not np.isnan(v) else None
                for k, v in stab.t_x.items()
            },
        }
    except Exception as e:
        st.error(f"Stability analysis error: {e}")
        result["module_1"] = {"error": str(e)}

    # ============================
    # MODULE 2: WINDOWS
    # ============================
    st.header("📐 Module 2: Mass Loss by Temperature Windows")

    try:
        win = analyze_guest_content_windows(tga)
        win_table = []
        for w in win.windows:
            win_table.append({
                "Window": w.name,
                "Start Mass (%)": round(w.mass_start_pct, 2),
                "End Mass (%)": round(w.mass_end_pct, 2),
                "Loss (%)": round(w.mass_loss_pct, 2),
            })
        st.dataframe(win_table, use_container_width=True, hide_index=True)

        st.caption(
            f"Total loss: {win.total_loss_pct:.2f}% | "
            f"Residue: {tga.residue_pct:.2f}% | "
            f"Sum: {win.total_loss_pct + tga.residue_pct:.2f}%"
        )

        result["module_2"] = {
            "windows": [
                {"name": w.name, "loss_pct": round(w.mass_loss_pct, 3)}
                for w in win.windows
            ],
            "total_loss_pct": round(win.total_loss_pct, 3),
        }
    except Exception as e:
        st.error(f"Window analysis error: {e}")
        result["module_2"] = {"error": str(e)}

    # ============================
    # RESIDUE ANALYSIS
    # ============================
    if formula and formula_valid:
        st.header("🧪 Residue Analysis: Observed vs. Predicted")

        try:
            parsed = parse_formula(formula)
            pred = predict_residue(formula, atmosphere=atmosphere)
            res = analyze_residue(
                tga.residue_pct, formula, atmosphere=atmosphere,
            )

            r1, r2, r3 = st.columns(3)
            r1.metric(
                "Predicted Residue",
                f"{pred.residue_pct_guest_free:.2f}%",
                help=f"From formula: {pred.oxide_formula}",
            )
            r2.metric("Observed Residue", f"{tga.residue_pct:.2f}%")

            delta_val = res.delta_pct
            if abs(delta_val) < 2:
                r3.metric("Δ (obs − pred)", f"{delta_val:+.2f} pp")
            else:
                r3.metric(
                    "Δ (obs − pred)",
                    f"{delta_val:+.2f} pp",
                    delta=f"{delta_val:+.1f}",
                    delta_color="inverse" if delta_val > 0 else "normal",
                )

            # Interpretation
            if abs(delta_val) < 2:
                st.success(res.delta_interpretation)
            elif delta_val < 0:
                st.warning(res.delta_interpretation)
            else:
                st.error(res.delta_interpretation)

            st.markdown(
                f"**Framework:** {formula} | "
                f"**MW:** {parsed.molar_mass:.2f} g/mol | "
                f"**Oxide:** {pred.oxide_formula}"
            )

            if res.warnings:
                for w in res.warnings:
                    st.warning(w)

            result["residue_analysis"] = {
                "formula": formula,
                "framework_mass": round(parsed.molar_mass, 3),
                "predicted_residue_pct": round(pred.residue_pct_guest_free, 2),
                "observed_residue_pct": round(tga.residue_pct, 2),
                "delta_pct": round(res.delta_pct, 2),
                "oxide_formula": pred.oxide_formula,
                "interpretation": res.delta_interpretation,
            }
        except Exception as e:
            st.error(f"Residue analysis error: {e}")
            result["residue_analysis"] = {"error": str(e)}

    # ============================
    # DOWNLOADS
    # ============================
    st.header("📥 Download Results")

    result["status"] = "COMPLETE"
    result["input_config"] = {
        "case_id": case_id,
        "formula": formula,
        "atmosphere": atmosphere,
        "heating_rate": heating_rate,
    }

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        result_json = json.dumps(result, indent=2, default=str)
        st.download_button(
            "📄 Result JSON",
            data=result_json,
            file_name=f"{case_id}_result.json",
            mime="application/json",
            use_container_width=True,
        )

    with dl2:
        report_text = generate_full_report_text(result, result.get("input_config", {}))
        st.download_button(
            "📝 Report (TXT)",
            data=report_text,
            file_name=f"{case_id}_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with dl3:
        # Combine both plots into a zip? Or just offer individually
        st.caption("Plots available above via individual download buttons.")

    # ============================
    # SUGGESTED WORDING
    # ============================
    with st.expander("📋 Suggested Publication Wording"):
        st.markdown("#### Methods")
        methods_text = (
            f"Thermogravimetric analysis (TGA) was performed under {atmosphere} "
            f"atmosphere at a heating rate of {heating_rate:.0f} °C/min. "
            f"Data were analyzed using TGA-MOF-Analyzer v0.3, an open-source "
            f"toolkit for automated TGA interpretation of metal–organic "
            f"frameworks. Mass-loss events were detected using valley-to-valley "
            f"segmentation of the derivative thermogravimetric (DTG) curve, "
            f"ensuring 100% mass balance across all detected events."
        )
        st.markdown(methods_text)

        if formula and "residue_analysis" in result and "error" not in result["residue_analysis"]:
            ra = result["residue_analysis"]
            st.markdown("#### Results")
            direction = "deficit" if ra["delta_pct"] < 0 else "excess"
            implication = (
                "indicates excess organic content relative to ideal stoichiometry, "
                "consistent with residual guests, modulator, or missing-linker defects."
                if ra["delta_pct"] < 0
                else "suggests inorganic impurity or incomplete organic combustion."
            )
            results_text = (
                f"The observed TGA residue ({ra['observed_residue_pct']:.2f} wt%) "
                f"was compared to the predicted {ra['oxide_formula']} residue "
                f"({ra['predicted_residue_pct']:.2f} wt%) calculated from the "
                f"ideal guest-free framework formula "
                f"(MW = {ra['framework_mass']:.2f} g/mol). "
                f"The {direction} of {abs(ra['delta_pct']):.1f} percentage points "
                f"{implication}"
            )
            st.markdown(results_text)

        st.caption("*Copy and adapt for your manuscript.*")

    # ============================
    # REFERENCES
    # ============================
    st.markdown("---")
    st.markdown(
        "*References: "
        "Abánades Lázaro, Eur. J. Inorg. Chem. 2020, 4284; "
        "Shearer et al., Chem. Mater. 2016, 28, 3749; "
        "Sannes et al., Chem. Mater. 2023, 35, 3793*"
    )

finally:
    # Cleanup temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass