"""
Module 2 — Guest / Solvent Content & Activation Quality
=========================================================
Answers: "How much solvent is trapped in my MOF?  Is activation complete?
          What's actually in those pores?"

Physics
-------
MOFs are synthesized in solvent (DMF, water, MeOH, or mixtures).
After synthesis, pores are filled with guest molecules.  "Activation"
means removing these guests to open the porosity.  TGA quantifies
remaining guests by measuring low-temperature mass loss — everything
that leaves before the framework decomposes.

The challenge: MOFs are often synthesized in mixed solvents, and the
pores may contain a mixture of guests.  A single 12% mass loss at
80–180°C could be 2 DMF, or 1 DMF + 3 H2O, or 8 H2O — TGA alone
cannot distinguish these.

Key equations
-------------
Single guest (known identity):
    wt% = (m_initial − m_after_loss) / m_initial × 100
    n_per_FU = (wt%/100 × M_FU) / M_guest

Mixed guests (unknown composition):
    Uses the combinatorial guest solver (core.guest_solver).
    f_calc = sum(n_i × MW_i) / (M_FU + sum(n_i × MW_i))
    Note the denominator: total mass = framework + guests.

Activation quality:
    Q_act = R_exp_activated / R_exp_theo_DH × 100%
    100% = perfect activation
    >100% = something extra still present
    <100% = possible partial framework collapse during activation

Drug / cargo loading
--------------------
Drug loading in MOFs for drug delivery is physically identical to
guest/solvent content — the "guest" is a drug molecule.  Use this
module with guest_mw = drug molecular weight.  Important: use the
EXPERIMENTAL formula mass from Module 3 as M_framework, not the
idealized textbook value, to avoid systematic error in loading%.

References
----------
- Abánades Lázaro, I. Eur. J. Inorg. Chem. 2020, 4284-4294.
  DOI: 10.1002/ejic.202000656  (Section 2.1, solvent quantification)
- Zahabi, G. et al. Sci. Rep. 2025, 15, 45135.
  DOI: 10.1038/s41598-025-33316-9  (drug loading by TGA, 21.3% example)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.dtg import (
    compute_dtg,
    detect_events,
    TGAEvent,
    EventDetectionResult,
)
from core.guest_solver import (
    GuestCandidate,
    GuestSolution,
    enumerate_guest_combinations,
    COMMON_GUESTS,
)


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class GuestContentResult:
    """Result of guest / solvent content analysis.

    Attributes
    ----------
    total_guest_wt_pct : float
        Total mass loss attributed to guest removal (wt%).
    n_guest_per_fu : float or None
        Number of guest molecules per formula unit.
        Only computed if guest_mw and M_framework are both known.
    guest_name : str or None
        Identity of the guest (if single guest specified).
    guest_mw : float or None
        Molar mass of the guest (g/mol).
    M_framework : float or None
        Molar mass of the guest-free MOF FU (g/mol).
    activation_quality_pct : float or None
        Activation quality score.
        100% = matches ideal DH-MOF formula.
        >100% = extra mass present (residual guest/modulator).
        <100% = framework may have partially collapsed.
    guest_event : TGAEvent or None
        The TGA event attributed to guest loss.
    solver_results : list of GuestSolution or None
        Ranked candidate guest mixtures from the combinatorial solver.
    """
    total_guest_wt_pct: float
    n_guest_per_fu: Optional[float] = None
    guest_name: Optional[str] = None
    guest_mw: Optional[float] = None
    M_framework: Optional[float] = None
    activation_quality_pct: Optional[float] = None
    guest_event: Optional[TGAEvent] = None
    solver_results: Optional[List[GuestSolution]] = None

    def summary(self) -> str:
        """Human-readable summary report."""
        lines = [
            "=" * 60,
            "MODULE 2: Guest Content & Activation Quality",
            "=" * 60,
            f"  Total guest loss:    {self.total_guest_wt_pct:.2f} wt%",
        ]

        if self.guest_name is not None:
            mw_str = f"{self.guest_mw:.2f}" if self.guest_mw else "?"
            lines.append(
                f"  Guest identity:      {self.guest_name} "
                f"(MW = {mw_str} g/mol)"
            )

        if self.n_guest_per_fu is not None:
            lines.append(
                f"  Guests per FU:       {self.n_guest_per_fu:.2f} mol/FU"
            )

        if self.activation_quality_pct is not None:
            q = self.activation_quality_pct
            if abs(q - 100.0) < 2.0:
                note = "(excellent — matches ideal)"
            elif q > 100.0:
                note = "(>100%: residual guest or modulator likely present)"
            else:
                note = "(<100%: possible partial framework collapse)"
            lines.append(f"  Activation quality:  {q:.1f}%  {note}")

        if self.guest_event is not None:
            ev = self.guest_event
            lines.append(
                f"  Guest event:         {ev.t_start:.0f} – "
                f"{ev.t_end:.0f} °C  (peak at {ev.t_peak:.0f} °C)"
            )

        if self.solver_results:
            lines.append("")
            lines.append("  Candidate guest mixtures (ranked by plausibility):")
            lines.append("  " + "-" * 56)
            lines.append(
                f"  {'Rank':<5} {'Formula':<30} {'Error':>8} {'Score':>8}"
            )
            lines.append("  " + "-" * 56)
            for i, sol in enumerate(self.solver_results[:10], 1):
                lines.append(
                    f"  {i:<5} {sol.formula_string():<30} "
                    f"{sol.mass_error:>7.3f}% {sol.penalty_score:>8.4f}"
                )
            if len(self.solver_results) > 10:
                lines.append(
                    f"  ... and {len(self.solver_results) - 10} more candidates"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


# ------------------------------------------------------------------
# Single-guest analysis
# ------------------------------------------------------------------

def analyze_guest_content(
    tga_data,
    events: Optional[EventDetectionResult] = None,
    guest_name: Optional[str] = None,
    guest_mw: Optional[float] = None,
    M_framework: Optional[float] = None,
    R_exp_theo_dh: Optional[float] = None,
    guest_event_index: int = 0,
    sg_window: Optional[int] = None,
    sg_polyorder: int = 3,
) -> GuestContentResult:
    """
    Quantify guest content from TGA data (single known guest).

    Parameters
    ----------
    tga_data : TGAData
        Parsed TGA data.
    events : EventDetectionResult or None
        Pre-computed events.  If None, events are auto-detected.
    guest_name : str or None
        Name of the guest molecule (e.g., 'DMF', 'H2O').
    guest_mw : float or None
        Molar mass of the guest (g/mol).
    M_framework : float or None
        Molar mass of the guest-free MOF formula unit (g/mol).
        Use the experimental value from Module 3 if available,
        or the idealized value as a first approximation.
    R_exp_theo_dh : float or None
        Theoretical R_exp for the perfectly activated DH-MOF.
        Used to compute activation quality score.
        Can be obtained from MOFComponents.R_theo_dh (Module 3).
    guest_event_index : int
        Which detected event corresponds to guest loss.
        Default 0 = first event (lowest temperature).
    sg_window, sg_polyorder : int
        Savitzky-Golay parameters for DTG (if events not provided).

    Returns
    -------
    GuestContentResult
    """
    m = tga_data.mass_pct

    # --- Event detection ---
    if events is None:
        dtg_result = compute_dtg(
            tga_data, sg_window=sg_window, sg_polyorder=sg_polyorder,
        )
        events = detect_events(tga_data, dtg_result=dtg_result)

    if len(events.events) == 0:
        return GuestContentResult(
            total_guest_wt_pct=0.0,
            guest_name=guest_name,
            guest_mw=guest_mw,
            M_framework=M_framework,
        )

    # Get the guest event
    evt_idx = min(guest_event_index, len(events.events) - 1)
    guest_event = events.events[evt_idx]
    total_guest_wt_pct = guest_event.mass_loss_pct

    # --- Moles per formula unit ---
    n_per_fu = None
    if guest_mw is not None and guest_mw > 0 and M_framework is not None:
        # n = (wt%/100 × M_FU) / M_guest
        n_per_fu = (total_guest_wt_pct / 100.0 * M_framework) / guest_mw

    # --- Activation quality ---
    q_act = None
    if R_exp_theo_dh is not None:
        m_after_guest = guest_event.mass_end_pct
        m_residue = float(m[-1])
        if m_residue > 0:
            R_exp_act = m_after_guest / m_residue
            q_act = (R_exp_act / R_exp_theo_dh) * 100.0

    return GuestContentResult(
        total_guest_wt_pct=total_guest_wt_pct,
        n_guest_per_fu=n_per_fu,
        guest_name=guest_name,
        guest_mw=guest_mw,
        M_framework=M_framework,
        activation_quality_pct=q_act,
        guest_event=guest_event,
    )


# ------------------------------------------------------------------
# Multi-guest analysis (combinatorial solver)
# ------------------------------------------------------------------

def analyze_guest_content_mixed(
    tga_data,
    M_framework: float,
    candidates: Optional[List[GuestCandidate]] = None,
    events: Optional[EventDetectionResult] = None,
    guest_event_index: int = 0,
    step: float = 0.5,
    tolerance: float = 0.005,
    pore_volume_cm3_per_mol: Optional[float] = None,
    max_results: int = 20,
    sg_window: Optional[int] = None,
    sg_polyorder: int = 3,
) -> GuestContentResult:
    """
    Quantify guest content using the combinatorial solver for mixed
    or unknown guest compositions.

    Instead of assuming a single guest identity, this function enumerates
    ALL stoichiometric combinations of candidate guest molecules that
    reproduce the observed mass loss within tolerance.  Results are
    ranked by a penalty function that encodes chemical plausibility.

    Parameters
    ----------
    tga_data : TGAData
        Parsed TGA data.
    M_framework : float
        Molar mass of the guest-free MOF FU (g/mol).
    candidates : list of GuestCandidate or None
        Candidate guest molecules.  If None, uses COMMON_GUESTS
        (H2O, DMF, MeOH, EtOH, DEF, acetone, THF, formic acid,
        acetic acid).
    events : EventDetectionResult or None
        Pre-computed events.  If None, auto-detected.
    guest_event_index : int
        Which event = guest loss (default 0 = first).
    step : float
        Stoichiometry step size (default 0.5).
        Use 0.25 for higher precision (slower).
    tolerance : float
        Acceptance tolerance on fractional mass loss (default 0.005).
    pore_volume_cm3_per_mol : float or None
        Pore volume per FU in cm³/mol.  Improves penalty scoring by
        penalizing solutions that overstuff the pore.
    max_results : int
        Maximum solutions to return (default 20).
    sg_window, sg_polyorder : int
        SG parameters for DTG.

    Returns
    -------
    GuestContentResult with solver_results populated.

    Notes
    -----
    The solver uses the equation:
        f_calc = sum(n_i × MW_i) / (M_FU + sum(n_i × MW_i))

    Note the denominator: M_FU + guest_mass, NOT just M_FU.
    TGA reports loss as fraction of TOTAL initial mass (framework + guests).
    Getting this wrong introduces systematic overestimation.
    """
    m = tga_data.mass_pct

    # --- Events ---
    if events is None:
        dtg_result = compute_dtg(
            tga_data, sg_window=sg_window, sg_polyorder=sg_polyorder,
        )
        events = detect_events(tga_data, dtg_result=dtg_result)

    if len(events.events) == 0:
        return GuestContentResult(
            total_guest_wt_pct=0.0,
            M_framework=M_framework,
        )

    evt_idx = min(guest_event_index, len(events.events) - 1)
    guest_event = events.events[evt_idx]
    total_guest_wt_pct = guest_event.mass_loss_pct

    # Convert wt% to fractional loss
    loss_frac = total_guest_wt_pct / 100.0

    # --- Run combinatorial solver ---
    solutions = enumerate_guest_combinations(
        M_framework=M_framework,
        loss_fraction=loss_frac,
        candidates=candidates,
        step=step,
        max_count=10.0,
        tolerance=tolerance,
        event_temp=guest_event.t_peak,
        pore_volume_cm3_per_mol=pore_volume_cm3_per_mol,
        max_results=max_results,
    )

    return GuestContentResult(
        total_guest_wt_pct=total_guest_wt_pct,
        M_framework=M_framework,
        guest_event=guest_event,
        solver_results=solutions,
    )


# ------------------------------------------------------------------
# Window-based guest analysis (v0.2 addition — bypasses event detector)
# ------------------------------------------------------------------

@dataclass
class WindowResult:
    """Result for a single temperature window."""
    name: str
    t_min: float
    t_max: float
    mass_loss_pct: float
    mass_start_pct: float
    mass_end_pct: float

# ------------------------------------------------------------------
# Window-based guest analysis (v0.2 — bypasses event detector)
# ------------------------------------------------------------------

@dataclass
class WindowResult:
    """Result for a single temperature window."""
    name: str
    t_min: float
    t_max: float
    mass_loss_pct: float
    mass_start_pct: float
    mass_end_pct: float


@dataclass
class WindowAnalysisResult:
    """Result of window-based guest/mass-loss analysis."""
    windows: List[WindowResult]
    total_loss_pct: float
    residue_pct: float
    mass_balance_pct: float

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "MODULE 2 (Window Mode): Mass Loss by Temperature Windows",
            "=" * 60,
        ]
        lines.append(
            f"  {'Window':<20} {'T_min':>6} {'T_max':>6} "
            f"{'Loss%':>7} {'m_start':>8} {'m_end':>8}"
        )
        lines.append("  " + "-" * 58)
        for w in self.windows:
            lines.append(
                f"  {w.name:<20} {w.t_min:>6.0f} {w.t_max:>6.0f} "
                f"{w.mass_loss_pct:>7.2f} {w.mass_start_pct:>8.2f} "
                f"{w.mass_end_pct:>8.2f}"
            )
        lines.append("  " + "-" * 58)
        lines.append(f"  {'Total loss:':<20} {'':>6} {'':>6} "
                      f"{self.total_loss_pct:>7.2f}")
        lines.append(f"  {'Residue:':<20} {'':>6} {'':>6} "
                      f"{self.residue_pct:>7.2f}")
        lines.append(f"  {'Mass balance:':<20} {'':>6} {'':>6} "
                      f"{self.mass_balance_pct:>7.2f}  "
                      f"({'✅' if abs(self.mass_balance_pct - 100) < 2 else '❌'})")
        lines.append("=" * 60)
        return "\n".join(lines)


def analyze_guest_content_windows(
    tga_data,
    windows=None,
    M_framework=None,
    guest_mw=None,
    guest_name=None,
):
    """
    Compute mass loss in user-specified temperature windows.

    This BYPASSES the automatic event detector entirely.

    Parameters
    ----------
    tga_data : TGAData
    windows : list of tuples, optional
        Each element is (t_min, t_max) or (name, t_min, t_max).
        If None, uses default MOF windows.
    M_framework : float or None
    guest_mw : float or None
    guest_name : str or None

    Returns
    -------
    WindowAnalysisResult
    """
    T = tga_data.temperature
    m = tga_data.mass_pct
    t_start = float(T[0])
    t_end_data = float(T[-1])

    if windows is None:
        windows = [
            ("RT-120°C",  t_start, 120.0),
            ("120-250°C", 120.0, 250.0),
            ("250-400°C", 250.0, 400.0),
            ("400-600°C", 400.0, 600.0),
            ("600-end",   600.0, t_end_data),
        ]

    parsed_windows = []
    for w in windows:
        if len(w) == 2:
            t_lo, t_hi = w
            name = f"{t_lo:.0f}-{t_hi:.0f}°C"
        elif len(w) == 3:
            name, t_lo, t_hi = w
        else:
            raise ValueError(f"Window must be (t_min, t_max) or (name, t_min, t_max), got {w}")

        t_lo = max(t_lo, t_start)
        t_hi = min(t_hi, t_end_data)

        m_lo = float(np.interp(t_lo, T, m))
        m_hi = float(np.interp(t_hi, T, m))
        loss = m_lo - m_hi

        parsed_windows.append(WindowResult(
            name=name,
            t_min=t_lo,
            t_max=t_hi,
            mass_loss_pct=loss,
            mass_start_pct=m_lo,
            mass_end_pct=m_hi,
        ))

    total_loss = sum(w.mass_loss_pct for w in parsed_windows)
    residue = float(m[-1])
    mass_balance = total_loss + residue

    return WindowAnalysisResult(
        windows=parsed_windows,
        total_loss_pct=total_loss,
        residue_pct=residue,
        mass_balance_pct=mass_balance,
    )