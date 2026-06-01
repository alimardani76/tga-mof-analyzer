"""
DTG (Derivative Thermogravimetry) Engine — v0.2
=================================================
Valley-to-valley event detection with artifact filtering.

Changes from v0.1
------------------
- REWRITTEN detect_events: uses valley-to-valley boundaries
  instead of threshold-based boundary walking.
  Guarantees 100% mass-loss capture.
- NEW: _find_valley_boundaries — locates DTG minima between peaks.
- NEW: _merge_small_events — iteratively merges tiny events.
- NEW: artifact detection — flags initial stabilization artifacts.

Algorithm (v0.2)
----------------
1. Savitzky-Golay smoothed derivative (unchanged).
2. Prominence-based peak finding (unchanged).
3. Valley finding: for each pair of adjacent peaks, find the
   DTG minimum between them.  These minima are the natural
   event boundaries.
4. Build events from valley to valley.  First event starts at
   data start; last event ends at data end.  This guarantees
   that sum(event losses) + residue = initial mass.
5. Merge events with mass_loss < threshold into their larger
   neighbor.
6. Optionally filter initial stabilization artifact.

Why valley-to-valley works
--------------------------
The old boundary walker asked: "where does this event's signal
become negligible?"  This is a hard question — the answer depends
on an arbitrary threshold, and mass between events is lost.

Valley-to-valley asks: "where does one event end and the next
begin?"  The answer is unambiguous — it's where the DTG is at
a local minimum between two peaks.  Every data point belongs
to exactly one event.  100% of mass is always accounted for.

References
----------
- Savitzky, A.; Golay, M.J.E. Anal. Chem. 1964, 36, 1627-1639.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.signal import savgol_filter, find_peaks


# ------------------------------------------------------------------
# Data containers (unchanged from v0.1)
# ------------------------------------------------------------------

@dataclass
class DTGResult:
    """Smoothed derivative thermogravimetry result."""
    temperature: np.ndarray
    dtg: np.ndarray
    sg_window: int
    sg_polyorder: int


@dataclass
class TGAEvent:
    """A single detected mass-loss event."""
    t_start: float
    t_peak: float
    t_end: float
    mass_loss_pct: float
    mass_start_pct: float
    mass_end_pct: float
    peak_dtg_value: float


@dataclass
class EventDetectionResult:
    """Complete event-detection output."""
    events: List[TGAEvent]
    dtg_result: DTGResult
    n_events: int = 0
    artifact_event: Optional[TGAEvent] = None
    total_captured_pct: float = 0.0

    def __post_init__(self):
        self.n_events = len(self.events)
        self.total_captured_pct = sum(
            ev.mass_loss_pct for ev in self.events
        )

    def summary_table(self) -> str:
        """Human-readable summary of detected events."""
        lines = []
        lines.append(f"Detected {self.n_events} mass-loss event(s)  "
                      f"[total captured: {self.total_captured_pct:.2f}%]:")
        if self.artifact_event is not None:
            ae = self.artifact_event
            lines.append(
                f"  (Initial artifact removed: {ae.t_start:.0f}–"
                f"{ae.t_end:.0f}°C, {ae.mass_loss_pct:.2f}%)"
            )
        lines.append("-" * 80)
        lines.append(
            f"{'Event':>5}  {'T_start':>8}  {'T_peak':>7}  {'T_end':>6}  "
            f"{'Loss%':>6}  {'m_start%':>8}  {'m_end%':>7}  {'DTG_peak':>9}"
        )
        lines.append("-" * 80)
        for i, ev in enumerate(self.events, 1):
            lines.append(
                f"{i:>5}  {ev.t_start:>8.1f}  {ev.t_peak:>7.1f}  "
                f"{ev.t_end:>6.1f}  {ev.mass_loss_pct:>6.2f}  "
                f"{ev.mass_start_pct:>8.2f}  {ev.mass_end_pct:>7.2f}  "
                f"{ev.peak_dtg_value:>9.4f}"
            )
        lines.append("-" * 80)
        return "\n".join(lines)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _round_odd(x: float) -> int:
    """Round to nearest odd integer >= 5."""
    n = max(5, int(round(x)))
    if n % 2 == 0:
        n += 1
    return n


def _find_valley_boundaries(
    peak_indices: np.ndarray,
    dtg: np.ndarray,
    n: int,
) -> List[int]:
    """
    Find valley (DTG minimum) indices between adjacent peaks.

    Returns a list of boundary indices:
      [0, valley_1, valley_2, ..., valley_{N-1}, n-1]

    The first boundary is always index 0 (data start).
    The last boundary is always index n-1 (data end).
    Between each pair of adjacent peaks, the boundary is placed
    at the index of minimum DTG.

    Parameters
    ----------
    peak_indices : array of int
        Indices of detected DTG peaks, sorted by temperature.
    dtg : np.ndarray
        DTG array (positive = mass loss).
    n : int
        Length of the data.

    Returns
    -------
    list of int
        Boundary indices.
    """
    boundaries = [0]

    for i in range(len(peak_indices) - 1):
        pk1 = peak_indices[i]
        pk2 = peak_indices[i + 1]

        if pk2 - pk1 <= 1:
            # Adjacent peaks — place valley at the lower one
            valley = pk1 if dtg[pk1] < dtg[pk2] else pk2
        else:
            # Find minimum DTG between the two peaks
            segment = dtg[pk1 + 1 : pk2]
            valley_local = np.argmin(segment)
            valley = pk1 + 1 + valley_local

        boundaries.append(valley)

    boundaries.append(n - 1)
    return boundaries


def _merge_small_events(
    events: List[TGAEvent],
    min_mass_loss: float,
) -> List[TGAEvent]:
    """
    Iteratively merge the smallest event into its larger neighbor
    until all events have mass_loss >= min_mass_loss.

    When merging two adjacent events, the combined event spans
    from the earlier t_start to the later t_end, with the t_peak
    taken from whichever had the larger DTG peak.

    Parameters
    ----------
    events : list of TGAEvent
    min_mass_loss : float
        Minimum mass loss (wt%) to keep an event.

    Returns
    -------
    list of TGAEvent
    """
    if len(events) <= 1:
        return events

    while True:
        # Find the smallest event
        smallest_idx = None
        smallest_loss = float("inf")
        for i, ev in enumerate(events):
            if ev.mass_loss_pct < smallest_loss:
                smallest_loss = ev.mass_loss_pct
                smallest_idx = i

        # Stop if all events are above threshold
        if smallest_loss >= min_mass_loss or len(events) <= 1:
            break

        # Choose merge target: adjacent event with larger mass loss
        i = smallest_idx
        if i == 0:
            merge_with = 1
        elif i == len(events) - 1:
            merge_with = i - 1
        else:
            if events[i - 1].mass_loss_pct >= events[i + 1].mass_loss_pct:
                merge_with = i - 1
            else:
                merge_with = i + 1

        # Combine: ensure a < b (a is the earlier event)
        a, b = sorted([i, merge_with])
        ev_a, ev_b = events[a], events[b]

        # The peak of the combined event is whichever had larger DTG
        if ev_a.peak_dtg_value >= ev_b.peak_dtg_value:
            peak_t = ev_a.t_peak
            peak_v = ev_a.peak_dtg_value
        else:
            peak_t = ev_b.t_peak
            peak_v = ev_b.peak_dtg_value

        combined = TGAEvent(
            t_start=ev_a.t_start,
            t_peak=peak_t,
            t_end=ev_b.t_end,
            mass_loss_pct=ev_a.mass_loss_pct + ev_b.mass_loss_pct,
            mass_start_pct=ev_a.mass_start_pct,
            mass_end_pct=ev_b.mass_end_pct,
            peak_dtg_value=peak_v,
        )

        events = events[:a] + [combined] + events[b + 1 :]

    return events


def _detect_initial_artifact(
    events: List[TGAEvent],
    data_start_temp: float,
    max_temp_offset: float = 50.0,
    min_rate: float = 0.15,
) -> tuple:
    """
    Detect and remove initial mass stabilization artifact.

    Many TGA instruments force-normalize the first data point to
    100%, but the sample has already lost surface moisture before
    the measurement stabilized.  This creates a sharp, artificial
    mass drop in the first few degrees that is not a real thermal
    event.

    Detection criteria:
    - The event starts within max_temp_offset °C of data start.
    - The mass-loss rate (wt%/°C) exceeds min_rate.
    - The event's temperature span is < 60°C.

    Parameters
    ----------
    events : list of TGAEvent
    data_start_temp : float
        Temperature of the first data point (°C).
    max_temp_offset : float
        Maximum distance from data start to consider (°C).
    min_rate : float
        Minimum mass-loss rate to flag as artifact (%/°C).

    Returns
    -------
    (filtered_events, artifact_event_or_None)
    """
    if not events:
        return events, None

    first = events[0]
    dt = first.t_end - first.t_start
    if dt <= 0:
        return events, None

    rate = first.mass_loss_pct / dt
    starts_near_beginning = (first.t_start - data_start_temp) < max_temp_offset
    short_span = dt < 60.0

    if starts_near_beginning and rate > min_rate and short_span:
        return events[1:], first

    return events, None


# ------------------------------------------------------------------
# compute_dtg — UNCHANGED from v0.1
# ------------------------------------------------------------------

def compute_dtg(
    tga_data,
    sg_window: Optional[int] = None,
    sg_polyorder: int = 3,
) -> DTGResult:
    """
    Compute the smoothed derivative thermogravimetry (DTG) curve.

    DTG = -d(mass_pct)/dT   [units: %/°C]

    Positive peaks in DTG correspond to mass-loss events.

    Parameters
    ----------
    tga_data : TGAData
        Parsed TGA data.
    sg_window : int or None
        Savitzky-Golay window size (must be odd, >= 5).
        If None, auto-scaled from data density.
    sg_polyorder : int
        Polynomial order for SG filter (default 3).

    Returns
    -------
    DTGResult
    """
    T = tga_data.temperature
    m = tga_data.mass_pct

    if len(T) < 10:
        raise ValueError("Need at least 10 data points for DTG.")

    dt_avg = np.mean(np.diff(T))
    if dt_avg <= 0:
        raise ValueError("Temperature must be monotonically increasing.")

    if sg_window is None:
        hr = tga_data.heating_rate if tga_data.heating_rate else 10.0
        raw_w = 15.0 / (hr * dt_avg) if (hr * dt_avg) > 0 else 15.0
        sg_window = _round_odd(raw_w)

    sg_window = _round_odd(min(sg_window, len(T) - 1))
    if sg_polyorder >= sg_window:
        sg_polyorder = sg_window - 1

    dm_dT = savgol_filter(
        m,
        window_length=sg_window,
        polyorder=sg_polyorder,
        deriv=1,
        delta=dt_avg,
    )

    dtg = -dm_dT

    return DTGResult(
        temperature=T,
        dtg=dtg,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
    )


# ------------------------------------------------------------------
# detect_events — v0.2 REWRITE (valley-to-valley)
# ------------------------------------------------------------------


def detect_events(
    tga_data,
    dtg_result: Optional[DTGResult] = None,

    prominence_factor: float = 0.03,
    min_mass_loss: float = 1.0,

    skip_initial_artifact: bool = True,
    artifact_max_temp_offset: float = 50.0,
    artifact_min_rate: float = 0.15,
) -> EventDetectionResult:
    """
    Detect discrete mass-loss events using valley-to-valley boundaries.

    v0.2 algorithm — guarantees 100% mass-loss capture.

    Steps
    -----
    1. Find DTG peaks using prominence-based detection.
    2. Find valleys (DTG minima) between each pair of adjacent peaks.
    3. Build events from valley to valley.  First event starts at
       data start; last event ends at data end.
    4. Merge events with mass_loss < min_mass_loss into neighbors.
    5. Optionally remove initial stabilization artifact.

    Parameters
    ----------
    tga_data : TGAData
        Parsed TGA data.
    dtg_result : DTGResult or None
        Pre-computed DTG.  If None, computed with default parameters.
    prominence_factor : float
        Minimum prominence as fraction of max DTG peak (default 0.03).
        Lower = more peaks detected = finer event boundaries.
    min_mass_loss : float
        Minimum mass loss (wt%) per event after merging (default 0.5).
    skip_initial_artifact : bool
        If True, detect and remove initial mass stabilization artifact
        (the sharp drop in the first few degrees from instrument
        normalization).  Default True.
    artifact_max_temp_offset : float
        Maximum distance from data start to consider as artifact (°C).
    artifact_min_rate : float
        Minimum mass-loss rate (%/°C) to flag as artifact.

    Returns
    -------
    EventDetectionResult
        Includes artifact_event if one was detected and removed.
    """
    if dtg_result is None:
        dtg_result = compute_dtg(tga_data)

    T = dtg_result.temperature
    dtg = dtg_result.dtg
    m_pct = tga_data.mass_pct
    n = len(T)

    # --- Step 1: Find peaks ---
    dtg_max = np.max(dtg)
    if dtg_max <= 0:
        return EventDetectionResult(events=[], dtg_result=dtg_result)

    min_prominence = prominence_factor * dtg_max
    peak_indices, _ = find_peaks(dtg, prominence=min_prominence)

    if len(peak_indices) == 0:
        return EventDetectionResult(events=[], dtg_result=dtg_result)

    # Sort peaks by temperature
    peak_indices = peak_indices[np.argsort(T[peak_indices])]

    # --- Step 2: Find valley boundaries ---
    boundaries = _find_valley_boundaries(peak_indices, dtg, n)

    # --- Step 3: Build events from boundary to boundary ---
    raw_events = []
    for i in range(len(boundaries) - 1):
        left = boundaries[i]
        right = boundaries[i + 1]

        if left >= right:
            continue

        t_start = float(T[left])
        t_end = float(T[right])
        mass_start = float(m_pct[left])
        mass_end = float(m_pct[right])
        mass_loss = mass_start - mass_end

        # Find the highest DTG peak within this segment
        segment_dtg = dtg[left : right + 1]
        local_peak_idx = np.argmax(segment_dtg)
        peak_global_idx = left + local_peak_idx
        t_peak = float(T[peak_global_idx])
        peak_value = float(dtg[peak_global_idx])

        raw_events.append(TGAEvent(
            t_start=t_start,
            t_peak=t_peak,
            t_end=t_end,
            mass_loss_pct=mass_loss,
            mass_start_pct=mass_start,
            mass_end_pct=mass_end,
            peak_dtg_value=peak_value,
        ))

    # --- Step 4: Merge small events ---
    events = _merge_small_events(raw_events, min_mass_loss)

    # --- Step 5: Filter initial artifact ---
    
    # --- Step 5: Flag initial artifact (but keep in events for mass balance) ---
    artifact = None
    if skip_initial_artifact and events:
        # Detect artifact but DON'T remove it from events list.
        # Removing it breaks mass balance (the mass is real, even if
        # it's an instrument artifact — it still left the sample).
        # We flag it in artifact_event so downstream code can handle
        # it appropriately (e.g., Module 1 can skip it when
        # determining the "activated mass" reference point).
        test_events = list(events)  # copy
        _, artifact = _detect_initial_artifact(
            test_events,
            data_start_temp=float(T[0]),
            max_temp_offset=artifact_max_temp_offset,
            min_rate=artifact_min_rate,
        )
        # artifact_event is set but events list is untouched

    return EventDetectionResult(
        events=events,
        dtg_result=dtg_result,
        artifact_event=artifact,
    )
