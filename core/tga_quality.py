"""
TGA Data Quality Checks
=========================
Automated validation of parsed TGA data.

These checks catch common data problems BEFORE any analysis runs.
Every check returns a clear pass/fail with an explanation.

The checks are:
1. Sufficient data points (≥50 for meaningful DTG)
2. Temperature range (must span ≥200°C)
3. Monotonic temperature (must be increasing)
4. Mass range sanity (should start near 100%, end > 0%)
5. Noise level estimation (from local variance)
6. Residue stability (last 5% of data should be flat)
7. Initial stabilization artifact detection

Design: every check returns a QualityCheck object.
The user decides what to do with warnings — we never
silently discard data or modify values.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QualityCheck:
    """Result of a single quality check."""
    name: str
    passed: bool
    value: float
    threshold: float
    message: str
    severity: str  # 'info', 'warning', 'error'


@dataclass
class QualityReport:
    """Complete quality assessment of a TGA dataset."""
    checks: List[QualityCheck]
    overall_pass: bool
    noise_estimate_pct: float
    has_initial_artifact: bool
    artifact_size_pct: float

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "TGA Data Quality Report",
            "=" * 60,
        ]
        for check in self.checks:
            icon = "✅" if check.passed else ("⚠️" if check.severity == "warning" else "❌")
            lines.append(f"  {icon} {check.name}: {check.message}")

        lines.append("")
        lines.append(f"  Noise estimate:     {self.noise_estimate_pct:.4f} wt%")
        lines.append(f"  Initial artifact:   "
                      f"{'yes (' + f'{self.artifact_size_pct:.2f}%' + ')' if self.has_initial_artifact else 'none detected'}")
        lines.append(f"  Overall:            "
                      f"{'PASS' if self.overall_pass else 'ISSUES FOUND'}")
        lines.append("=" * 60)
        return "\n".join(lines)


def run_quality_checks(tga_data) -> QualityReport:
    """
    Run all quality checks on a TGAData object.

    Parameters
    ----------
    tga_data : TGAData

    Returns
    -------
    QualityReport
    """
    T = tga_data.temperature
    m = tga_data.mass_pct
    checks = []

    # --- Check 1: Sufficient data points ---
    n = len(T)
    checks.append(QualityCheck(
        name="Data points",
        passed=n >= 50,
        value=n,
        threshold=50,
        message=f"{n} points ({'sufficient' if n >= 50 else 'too few for reliable DTG'})",
        severity="error" if n < 50 else "info",
    ))

    # --- Check 2: Temperature range ---
    t_span = T[-1] - T[0]
    checks.append(QualityCheck(
        name="Temperature range",
        passed=t_span >= 200,
        value=t_span,
        threshold=200,
        message=f"{T[0]:.1f}–{T[-1]:.1f}°C (span: {t_span:.1f}°C)",
        severity="warning" if t_span < 200 else "info",
    ))

    # --- Check 3: Monotonic temperature ---
    diffs = np.diff(T)
    n_decreasing = np.sum(diffs < 0)
    frac_mono = 1.0 - n_decreasing / len(diffs)
    checks.append(QualityCheck(
        name="Temperature monotonicity",
        passed=n_decreasing == 0,
        value=frac_mono * 100,
        threshold=100.0,
        message=(f"{'Strictly increasing' if n_decreasing == 0 else f'{n_decreasing} non-increasing steps ({frac_mono*100:.1f}% monotonic)'}"),
        severity="error" if n_decreasing > len(diffs) * 0.05 else ("warning" if n_decreasing > 0 else "info"),
    ))

    # --- Check 4: Mass range sanity ---
    m_start = m[0]
    m_end = m[-1]
    start_ok = 80 < m_start < 120
    end_ok = m_end > -5  # allow small negative from drift
    checks.append(QualityCheck(
        name="Mass range",
        passed=start_ok and end_ok,
        value=m_start,
        threshold=100.0,
        message=f"Start: {m_start:.2f}%, End: {m_end:.2f}%"
                + ("" if start_ok else " ⚠️ unusual start value")
                + ("" if end_ok else " ⚠️ negative final mass"),
        severity="warning" if not (start_ok and end_ok) else "info",
    ))

    # --- Check 5: Noise level ---
    # Estimate noise from local variance using a sliding window
    window = min(11, n // 10)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1

    local_residuals = []
    for i in range(window // 2, n - window // 2):
        segment = m[i - window // 2 : i + window // 2 + 1]
        # Fit linear trend and measure residual
        x = np.arange(len(segment))
        coeffs = np.polyfit(x, segment, 1)
        fitted = np.polyval(coeffs, x)
        residual = np.std(segment - fitted)
        local_residuals.append(residual)

    noise = np.median(local_residuals) if local_residuals else 0.0
    checks.append(QualityCheck(
        name="Noise level",
        passed=noise < 0.5,
        value=noise,
        threshold=0.5,
        message=f"{noise:.4f} wt% (median local σ)"
                + ("" if noise < 0.5 else " ⚠️ noisy data"),
        severity="warning" if noise >= 0.5 else "info",
    ))

    # --- Check 6: Residue stability ---
    # Last 5% of data should be relatively flat
    n_tail = max(10, n // 20)
    tail = m[-n_tail:]
    tail_range = tail.max() - tail.min()
    tail_std = np.std(tail)
    checks.append(QualityCheck(
        name="Residue stability",
        passed=tail_range < 2.0,
        value=tail_range,
        threshold=2.0,
        message=f"Last {n_tail} points: range={tail_range:.3f}%, "
                f"σ={tail_std:.3f}%"
                + ("" if tail_range < 2.0 else " ⚠️ residue not stable"),
        severity="warning" if tail_range >= 2.0 else "info",
    ))

    # --- Check 7: Initial artifact ---
    # Sharp drop in first few degrees
    has_artifact = False
    artifact_size = 0.0
    if n > 10:
        # Check first 10 points
        initial_drop = m[0] - m[min(10, n - 1)]
        initial_dt = T[min(10, n - 1)] - T[0]
        if initial_dt > 0:
            rate = initial_drop / initial_dt
            if rate > 0.15 and initial_drop > 1.0:
                has_artifact = True
                artifact_size = initial_drop

    checks.append(QualityCheck(
        name="Initial artifact",
        passed=not has_artifact,
        value=artifact_size,
        threshold=1.0,
        message=("None detected" if not has_artifact
                 else f"Detected: {artifact_size:.2f}% drop in first "
                      f"{T[min(10, n-1)] - T[0]:.1f}°C ",
                      f"(likely instrument stabilization, not thermal event)"),
        severity="warning" if has_artifact else "info",
    ))

    overall = all(c.passed or c.severity != "error" for c in checks)

    return QualityReport(
        checks=checks,
        overall_pass=overall,
        noise_estimate_pct=noise,
        has_initial_artifact=has_artifact,
        artifact_size_pct=artifact_size,
    )