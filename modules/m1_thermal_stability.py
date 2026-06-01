"""
Module 1 — Thermal Stability Profiling
========================================
Answers: "At what temperature does my MOF decompose?  How stable is it?"

Outputs
-------
- Onset decomposition temperature T_onset (tangent intersection method)
- T_x metrics: T1, T2, T5, T10  (temperature at x% mass loss from
  the activated reference mass)
- Stability window: temperature range between activation end and
  decomposition onset
- Number and characteristics of all mass-loss events via DTG

Physics
-------
A MOF decomposes in stages:
  1. RT → 100–250°C:  Guest / solvent removal
  2. 250–400°C:       Modulator loss, dehydroxylation of nodes
  3. 400–600°C:       Linker combustion (air) or carbonization (N2)
  4. >600°C:          Stable metal oxide residue (air)

The onset temperature T_onset is defined as the intersection of the
pre-decomposition baseline tangent and the steepest-descent tangent
during framework decomposition.  This follows ASTM/ISO convention.

The T_x metrics (T1, T2, T5, T10) give a standardized, reproducible
stability fingerprint.  These are standard in polymer science and
ceramics but underused in MOFs — including them improves cross-lab
comparability.

Heating rate dependence
-----------------------
T_onset shifts ~10–30°C higher at 10°C/min vs. 2°C/min due to
thermal lag.  Always report the heating rate alongside T_onset.

Atmosphere dependence
---------------------
The same MOF gives different T_onset in air vs. N2 because oxidative
decomposition is thermodynamically favored.  In N2, the framework
may survive to higher T but produces different products (carbides,
metal, amorphous carbon).

References
----------
- Healy, C. et al. Coord. Chem. Rev. 2020, 419, 213388.
  DOI: 10.1016/j.ccr.2020.213388
- Moseson, D.E. et al. Int. J. Pharm. 2020, 590, 119916.
  DOI: 10.1016/j.ijpharm.2020.119916
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.dtg import (
    compute_dtg,
    detect_events,
    DTGResult,
    TGAEvent,
    EventDetectionResult,
)


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class StabilityResult:
    """Complete thermal stability analysis output.

    Attributes
    ----------
    t_onset : float
        Decomposition onset temperature (°C) by tangent method.
    t_x : dict
        {x: T_x} where T_x is the temperature at which x% of the
        activated mass has been lost.  Keys: 1, 2, 5, 10.
    stability_window : tuple of (float, float)
        (T_end_activation, T_onset_decomp) in °C.
        The usable operating range of the activated MOF.
    events : EventDetectionResult
        All detected mass-loss events from DTG analysis.
    total_mass_loss_pct : float
        Total mass lost from initial to final residue (wt%).
    residue_pct : float
        Final residue mass (wt%).
    activated_mass_pct : float
        Reference mass after guest removal (wt%).
        This is the denominator for T_x calculations.
    """

    t_onset: float
    t_onset_threshold: float    # ADD THIS
    t_dtg_max: float            # ADD THIS
    t_x: Dict[int, float]
    stability_window: Tuple[float, float]
    events: EventDetectionResult
    total_mass_loss_pct: float
    residue_pct: float
    activated_mass_pct: float


    def summary(self) -> str:
        """Human-readable summary report."""
        lines = [
            "=" * 60,
            "MODULE 1: Thermal Stability Profiling",
            "=" * 60,
            f"  T_onset (tangent):    {self.t_onset:.1f} °C",
            
            f"  T_onset (threshold):  {self.t_onset_threshold:.1f} °C",
            f"  T_DTG_max:           {self.t_dtg_max:.1f} °C",

        ]
        for x in sorted(self.t_x.keys()):
            val = self.t_x[x]
            val_str = f"{val:.1f} °C" if not np.isnan(val) else "N/A (not reached)"
            lines.append(f"  T_{x}:                {val_str}")

        t_lo, t_hi = self.stability_window
        if not np.isnan(t_lo) and not np.isnan(t_hi):
            delta = t_hi - t_lo
            lines.append(
                f"  Stability window:    {t_lo:.1f} – {t_hi:.1f} °C "
                f"(Δ = {delta:.0f} °C)"
            )
        else:
            lines.append("  Stability window:    Could not be determined")

        lines.extend([
            f"  Activated mass:      {self.activated_mass_pct:.2f} wt%",
            f"  Residue mass:        {self.residue_pct:.2f} wt%",
            f"  Total mass loss:     {self.total_mass_loss_pct:.2f} wt%",
            "",
            self.events.summary_table(),
            "=" * 60,
        ])
        return "\n".join(lines)


# ------------------------------------------------------------------
# T_x computation
# ------------------------------------------------------------------

def _compute_tx(
    temperature: np.ndarray,
    mass_pct: np.ndarray,
    ref_mass: float,
    x_values: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Compute T_x: temperature at which x% of the reference mass is lost.

    T_x = T where mass = ref_mass * (1 - x/100)

    Parameters
    ----------
    temperature : np.ndarray
        Temperature array (°C).
    mass_pct : np.ndarray
        Mass array (wt%).
    ref_mass : float
        Reference mass (wt%) — the activated (guest-free) mass.
    x_values : list of int or None
        Percent-loss values to compute.  Default [1, 2, 5, 10].

    Returns
    -------
    dict {x: T_x}
    """
    if x_values is None:
        x_values = [1, 2, 5, 10]

    t_x = {}
    for x in x_values:
        target_mass = ref_mass * (1.0 - x / 100.0)

        # Find where mass curve first drops below target_mass
        below = np.where(mass_pct <= target_mass)[0]
        if len(below) == 0:
            t_x[x] = float("nan")
            continue
        idx = below[0]
        if idx == 0:
            t_x[x] = float(temperature[0])
            continue

        # Linear interpolation between idx-1 and idx
        m0, m1 = mass_pct[idx - 1], mass_pct[idx]
        t0, t1 = temperature[idx - 1], temperature[idx]
        if abs(m0 - m1) < 1e-12:
            t_x[x] = float(t0)
        else:
            frac = (m0 - target_mass) / (m0 - m1)
            t_x[x] = float(t0 + frac * (t1 - t0))

    return t_x


# ------------------------------------------------------------------
# Tangent-intersection T_onset
# ------------------------------------------------------------------

def _compute_t_onset_tangent(
    temperature: np.ndarray,
    mass_pct: np.ndarray,
    dtg: np.ndarray,
    decomp_peak_idx: int,
    baseline_mass: float,
) -> float:
    """
    Compute T_onset via tangent intersection method.

    The tangent intersection method defines T_onset as the point where:
      - A horizontal line at y = baseline_mass  (the DH plateau)
      - Intersects the tangent line drawn through the steepest point
        of the decomposition curve.

    The tangent line at the DTG peak has slope = -DTG_peak  (in m-vs-T
    space, since DTG = -dm/dT, the slope dm/dT = -DTG_peak).

    Tangent line:  m(T) = m_pk - DTG_pk * (T - T_pk)
    Set m(T) = baseline_mass:
      T_onset = T_pk + (m_pk - baseline_mass) / DTG_pk

    Parameters
    ----------
    temperature : np.ndarray
    mass_pct : np.ndarray
    dtg : np.ndarray
        DTG array (positive = mass loss).
    decomp_peak_idx : int
        Index of the DTG peak for the decomposition event.
    baseline_mass : float
        Mass at the pre-decomposition plateau (wt%).

    Returns
    -------
    float
        T_onset in °C.  NaN if DTG peak is non-positive.
    """
    T_pk = temperature[decomp_peak_idx]
    m_pk = mass_pct[decomp_peak_idx]
    dtg_pk = dtg[decomp_peak_idx]

    if dtg_pk <= 0:
        return float("nan")

    # Tangent line: m(T) = m_pk - dtg_pk * (T - T_pk)
    # Set m(T) = baseline_mass and solve for T:
    # baseline_mass = m_pk - dtg_pk * (T_onset - T_pk)
    # T_onset = T_pk + (m_pk - baseline_mass) / dtg_pk
    t_onset = T_pk + (m_pk - baseline_mass) / dtg_pk

    return float(t_onset)


# ------------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Additional onset methods (Stage 5)
# ------------------------------------------------------------------

def _compute_t_onset_threshold(
    temperature: np.ndarray,
    dtg: np.ndarray,
    decomp_start_idx: int,
    decomp_end_idx: int,
) -> float:
    """
    Compute T_onset by threshold method.

    T_onset = first temperature where DTG exceeds a threshold
    defined as max(0.05 × DTG_max, median(DTG) + 0.5 × std(DTG))
    within the decomposition window.

    This is more robust than tangent for noisy data or data
    without a clear single decomposition peak.

    Parameters
    ----------
    temperature : np.ndarray
    dtg : np.ndarray
    decomp_start_idx : int
        Start index of the decomposition window.
    decomp_end_idx : int
        End index of the decomposition window.

    Returns
    -------
    float
        T_onset in °C. NaN if threshold never exceeded.
    """
    window_dtg = dtg[decomp_start_idx:decomp_end_idx]
    window_T = temperature[decomp_start_idx:decomp_end_idx]

    if len(window_dtg) == 0:
        return float("nan")

    dtg_max = np.max(window_dtg)
    dtg_median = np.median(window_dtg)
    dtg_std = np.std(window_dtg)

    threshold = max(0.05 * dtg_max, dtg_median + 0.5 * dtg_std)

    above = np.where(window_dtg >= threshold)[0]
    if len(above) == 0:
        return float("nan")

    return float(window_T[above[0]])


def _compute_t_dtg_max(
    temperature: np.ndarray,
    dtg: np.ndarray,
    decomp_start_idx: int,
    decomp_end_idx: int,
) -> float:
    """
    Find the temperature of maximum DTG in the decomposition window.

    This is the simplest and most reproducible decomposition metric.
    It corresponds to the temperature of maximum mass-loss rate.

    Parameters
    ----------
    temperature : np.ndarray
    dtg : np.ndarray
    decomp_start_idx : int
    decomp_end_idx : int

    Returns
    -------
    float
        Temperature in °C. NaN if window is empty.
    """
    window_dtg = dtg[decomp_start_idx:decomp_end_idx]
    window_T = temperature[decomp_start_idx:decomp_end_idx]

    if len(window_dtg) == 0:
        return float("nan")

    peak_idx = np.argmax(window_dtg)
    return float(window_T[peak_idx])

    
def analyze_stability(
    tga_data,
    activated_mass_pct: Optional[float] = None,
    sg_window: Optional[int] = None,
    sg_polyorder: int = 3,
    prominence_factor: float = 0.03,
    min_event_loss: float = 1.0,
    tx_values: Optional[List[int]] = None,
) -> StabilityResult:
    """
    Perform complete thermal stability analysis.

    Parameters
    ----------
    tga_data : TGAData
        Parsed TGA data (from core.tga_parser).
    activated_mass_pct : float or None
        Reference mass (wt%) after guest removal.
        If None, automatically taken as the mass at the end of the
        first detected event (assumed to be guest/solvent loss).
    sg_window : int or None
        Savitzky-Golay window for DTG.  Auto-scaled if None.
    sg_polyorder : int
        Savitzky-Golay polynomial order (default 3).
    prominence_factor : float
        DTG peak detection sensitivity (fraction of max peak).
    min_event_loss : float
        Minimum mass loss (wt%) to count as an event.
    tx_values : list of int or None
        Which T_x values to compute.  Default [1, 2, 5, 10].

    Returns
    -------
    StabilityResult
    """
    T = tga_data.temperature
    m = tga_data.mass_pct

    # --- Step 1: Compute DTG and detect events ---
    dtg_result = compute_dtg(
        tga_data,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
    )
    event_result = detect_events(
        tga_data,
        dtg_result=dtg_result,
        prominence_factor=prominence_factor,
        min_mass_loss=min_event_loss,
    )

    events = event_result.events
    dtg = dtg_result.dtg

    # --- Basic outputs ---
    residue_pct = float(m[-1])
    total_loss = float(m[0] - m[-1])

    # --- Handle edge case: no events detected ---
    if len(events) == 0:
        if activated_mass_pct is None:
            activated_mass_pct = float(m[0])
        return StabilityResult(
            t_onset=float("nan"),
            t_x={x: float("nan") for x in (tx_values or [1, 2, 5, 10])},
            stability_window=(float("nan"), float("nan")),
            events=event_result,
            total_mass_loss_pct=total_loss,
            residue_pct=residue_pct,
            activated_mass_pct=activated_mass_pct,
        )

    # --- Step 2: Identify the first non-artifact event ---
    # The first event is typically guest/solvent loss.
    # If an artifact was flagged, skip it.
    if (event_result.artifact_event is not None
            and len(events) > 1
            and events[0].t_start == event_result.artifact_event.t_start):
        first_real_event = events[1]
    else:
        first_real_event = events[0]

    # --- Step 3: Determine activated mass ---
    # The "activated mass" is the mass AFTER the first guest-loss event.
    # This is the reference point for T_x calculations.
    # The user can override this with an explicit value.
    if activated_mass_pct is None:
        activated_mass_pct = first_real_event.mass_end_pct

    # Temperature at end of activation (= end of first real event)
    t_end_activation = first_real_event.t_end

    # --- Step 4: Identify the decomposition event ---
    # The "decomposition event" is the dominant mass-loss event
    # at HIGH temperature (above guest-loss region).
    #
    # We pick the event above 250°C with the highest DTG peak
    # (steepest mass-loss rate).  This is more robust than picking
    # the largest total mass loss, because a single broad guest-loss
    # event can be larger than the decomposition step.
    #
    # 250°C is a conservative threshold — above typical guest
    # desorption but below linker combustion for essentially all
    # MOFs (Zr: 350–550°C, Zn: 300–450°C, Cu: 250–350°C).

    decomp_min_temp = 250.0

    high_t_events = [
        ev for ev in events if ev.t_peak >= decomp_min_temp
    ]

    if high_t_events:
        decomp_event = max(high_t_events, key=lambda e: e.peak_dtg_value)
    else:
        # Fallback: no events above 250°C — use the event with
        # the highest DTG peak overall
        decomp_event = max(events, key=lambda e: e.peak_dtg_value)

    # --- Step 5: Compute T_onset via tangent intersection ---
    # Find the DTG peak index within the decomposition event
    decomp_mask = (T >= decomp_event.t_start) & (T <= decomp_event.t_end)
    decomp_indices = np.where(decomp_mask)[0]

    if len(decomp_indices) > 0:
        local_dtg = dtg[decomp_indices]
        peak_local_idx = np.argmax(local_dtg)
        peak_global_idx = decomp_indices[peak_local_idx]
    else:
        peak_global_idx = np.argmax(dtg)

    # Baseline mass = mass at the start of the decomposition event.
    # This represents the pre-decomposition plateau.
    baseline_mass = decomp_event.mass_start_pct

    t_onset = _compute_t_onset_tangent(
        T, m, dtg,
        decomp_peak_idx=peak_global_idx,
        baseline_mass=baseline_mass,
    )

    
    # --- Step 5b: Additional onset methods ---
    # Decomposition window for threshold/DTG_max methods
    decomp_start_idx = np.searchsorted(T, decomp_event.t_start)
    decomp_end_idx = np.searchsorted(T, decomp_event.t_end)

    t_onset_threshold = _compute_t_onset_threshold(
        T, dtg, decomp_start_idx, decomp_end_idx,
    )
    t_dtg_max = _compute_t_dtg_max(
        T, dtg, decomp_start_idx, decomp_end_idx,
    )


    # --- Step 6: Compute T_x metrics ---
    t_x = _compute_tx(T, m, ref_mass=activated_mass_pct, x_values=tx_values)

    # --- Step 7: Stability window ---
    # Lower bound: end of first guest-loss event (activation complete)
    # Upper bound: T_onset of decomposition (or start of decomp event)
    #
    # If T_onset is valid, use it.  Otherwise use the start of the
    # decomposition event as the upper bound.
    t_decomp_start = decomp_event.t_start

    if not np.isnan(t_onset):
        upper = min(t_onset, t_decomp_start)
    else:
        upper = t_decomp_start

    stability_window = (t_end_activation, upper)


    return StabilityResult(
        t_onset=t_onset,
        t_onset_threshold=t_onset_threshold,
        t_dtg_max=t_dtg_max,
        t_x=t_x,
        stability_window=stability_window,
        events=event_result,
        total_mass_loss_pct=total_loss,
        residue_pct=residue_pct,
        activated_mass_pct=activated_mass_pct,
    )
