"""
Uncertainty Propagation
========================
Analytical error propagation for every computed quantity.

Every number the toolkit reports should have an uncertainty.
This module provides the math to compute σ for:
  - Guest count (n per formula unit)
  - Composition (q linkers)
  - Composite loading (w_MOF)
  - Residue comparison (Δ)

Method: Linear (first-order Taylor) error propagation.
For a function y = f(x1, x2, ...):
  σ_y² = Σ (∂f/∂xi)² σ_xi²

This assumes independent, normally-distributed errors.
For TGA data, the dominant error sources are:
  1. Mass measurement noise (from quality check noise estimate)
  2. Temperature accuracy (typically ±1°C for modern instruments)
  3. Residue drift (from quality check residue stability)
  4. Molar mass rounding (typically ±0.01 g/mol, negligible)

References
----------
- JCGM 100:2008 (GUM) — Guide to Expression of Uncertainty
- Taylor, J.R. "An Introduction to Error Analysis" (1997)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------

@dataclass
class UncertainValue:
    """A value with its uncertainty.

    Display: value ± uncertainty (to appropriate sig figs)
    """
    value: float
    sigma: float
    unit: str = ""

    def __repr__(self) -> str:
        if self.sigma > 0:
            # Determine decimal places from sigma magnitude
            if self.sigma >= 1:
                dp = 1
            elif self.sigma >= 0.1:
                dp = 2
            else:
                dp = 3
            return f"{self.value:.{dp}f} ± {self.sigma:.{dp}f}{' ' + self.unit if self.unit else ''}"
        return f"{self.value:.3f}{' ' + self.unit if self.unit else ''}"

    @property
    def relative_pct(self) -> float:
        """Relative uncertainty as percentage."""
        if abs(self.value) < 1e-12:
            return float("inf")
        return abs(self.sigma / self.value) * 100.0


# ------------------------------------------------------------------
# Guest count uncertainty
# ------------------------------------------------------------------

def guest_count_uncertainty(
    mass_loss_pct: float,
    framework_mass: float,
    guest_mass: float,
    sigma_mass_loss_pct: float = 0.5,
    sigma_framework_mass: float = 0.1,
    sigma_guest_mass: float = 0.01,
) -> UncertainValue:
    """
    Uncertainty in guest count per formula unit.

    Equation:
        n = (f / (1-f)) × (M_F / M_G)
    where f = mass_loss_pct / 100

    Partial derivatives:
        ∂n/∂f = M_F / (M_G × (1-f)²)
        ∂n/∂M_F = f / (M_G × (1-f))
        ∂n/∂M_G = -f × M_F / (M_G² × (1-f))

    Parameters
    ----------
    mass_loss_pct : float
        Observed mass loss in the window (wt%).
    framework_mass : float
        Guest-free framework molar mass (g/mol).
    guest_mass : float
        Guest molar mass (g/mol).
    sigma_mass_loss_pct : float
        Uncertainty in mass loss (wt%). Default 0.5% — typical
        for well-calibrated TGA at ~5 mg sample mass.
    sigma_framework_mass : float
        Uncertainty in framework mass (g/mol).
    sigma_guest_mass : float
        Uncertainty in guest mass (g/mol).

    Returns
    -------
    UncertainValue
        Guest count ± uncertainty (mol/mol).
    """
    if guest_mass <= 0 or framework_mass <= 0:
        return UncertainValue(0.0, float("inf"), "mol/mol")

    f = mass_loss_pct / 100.0
    sigma_f = sigma_mass_loss_pct / 100.0
    M_F = framework_mass
    M_G = guest_mass

    if f >= 1.0 or f <= 0.0:
        n = 0.0 if f <= 0 else float("inf")
        return UncertainValue(n, float("inf"), "mol/mol")

    # Value
    n = (f / (1 - f)) * (M_F / M_G)

    # Partial derivatives
    dn_df = M_F / (M_G * (1 - f) ** 2)
    dn_dMF = f / (M_G * (1 - f))
    dn_dMG = -f * M_F / (M_G ** 2 * (1 - f))

    # Propagate
    sigma_n = np.sqrt(
        (dn_df * sigma_f) ** 2
        + (dn_dMF * sigma_framework_mass) ** 2
        + (dn_dMG * sigma_guest_mass) ** 2
    )

    return UncertainValue(n, float(sigma_n), "mol/mol")


# ------------------------------------------------------------------
# Composition (q linkers) uncertainty
# ------------------------------------------------------------------

def composition_uncertainty(
    mass_at_plateau_pct: float,
    residue_pct: float,
    M_residue: float,
    M_node: float,
    M_linker: float,
    sigma_mass_pct: float = 0.5,
    sigma_residue_pct: float = 0.2,
) -> UncertainValue:
    """
    Uncertainty in linker count q.

    Equation:
        R = m_plateau / m_residue
        q = (R × M_residue - M_node) / M_linker

    Combined:
        q = (m_plateau × M_residue / (m_residue × M_linker))
            - M_node / M_linker

    Partial derivatives:
        ∂q/∂m_plateau = M_residue / (m_residue × M_linker)
        ∂q/∂m_residue = -m_plateau × M_residue / (m_residue² × M_linker)

    Parameters
    ----------
    mass_at_plateau_pct : float
        Mass at DH plateau (wt%).
    residue_pct : float
        Final residue mass (wt%).
    M_residue : float
        Total residue molar mass (g/mol), e.g., 6×M(ZrO2).
    M_node : float
        Dehydroxylated node mass (g/mol).
    M_linker : float
        Linker molar mass (g/mol).
    sigma_mass_pct : float
        Uncertainty in mass at plateau (wt%). Estimated from
        local noise at the plateau temperature.
    sigma_residue_pct : float
        Uncertainty in residue (wt%). Estimated from residue
        stability (std of last N points).

    Returns
    -------
    UncertainValue
        Linker count q ± uncertainty.
    """
    if residue_pct <= 0 or M_linker <= 0:
        return UncertainValue(0.0, float("inf"), "linkers/FU")

    m_p = mass_at_plateau_pct
    m_r = residue_pct

    # Value
    R = m_p / m_r
    q = (R * M_residue - M_node) / M_linker

    # Partials
    dq_dmp = M_residue / (m_r * M_linker)
    dq_dmr = -m_p * M_residue / (m_r ** 2 * M_linker)

    sigma_q = np.sqrt(
        (dq_dmp * sigma_mass_pct) ** 2
        + (dq_dmr * sigma_residue_pct) ** 2
    )

    return UncertainValue(float(q), float(sigma_q), "linkers/FU")


# ------------------------------------------------------------------
# Composite loading uncertainty
# ------------------------------------------------------------------

def composite_loading_uncertainty(
    r_composite: float,
    r_mof: float,
    r_additive: float,
    sigma_r_composite: float = 0.005,
    sigma_r_mof: float = 0.005,
    sigma_r_additive: float = 0.005,
) -> UncertainValue:
    """
    Uncertainty in MOF weight fraction in a composite.

    Equation:
        w = (r_comp - r_add) / (r_MOF - r_add)

    Partial derivatives:
        ∂w/∂r_comp = 1 / (r_MOF - r_add)
        ∂w/∂r_MOF  = -(r_comp - r_add) / (r_MOF - r_add)²
        ∂w/∂r_add  = (r_comp - r_MOF) / (r_MOF - r_add)²

    Parameters
    ----------
    r_composite, r_mof, r_additive : float
        Residue fractions (0-1 scale).
    sigma_r_* : float
        Uncertainties in residue fractions.

    Returns
    -------
    UncertainValue
        MOF weight fraction ± uncertainty (as fraction 0-1).
    """
    denom = r_mof - r_additive
    if abs(denom) < 1e-6:
        return UncertainValue(0.0, float("inf"), "fraction")

    w = (r_composite - r_additive) / denom

    dw_drc = 1.0 / denom
    dw_drm = -(r_composite - r_additive) / denom ** 2
    dw_dra = (r_composite - r_mof) / denom ** 2

    sigma_w = np.sqrt(
        (dw_drc * sigma_r_composite) ** 2
        + (dw_drm * sigma_r_mof) ** 2
        + (dw_dra * sigma_r_additive) ** 2
    )

    return UncertainValue(float(w), float(sigma_w), "fraction")


# ------------------------------------------------------------------
# Window mass loss uncertainty
# ------------------------------------------------------------------

def window_mass_loss_uncertainty(
    noise_estimate_pct: float,
) -> float:
    """
    Estimate uncertainty in a window mass loss measurement.

    A window loss = m(T1) - m(T2). Each interpolated mass has
    uncertainty ≈ noise_estimate. By error propagation:
        σ_loss = sqrt(2) × σ_mass ≈ 1.41 × noise_estimate

    Parameters
    ----------
    noise_estimate_pct : float
        Noise estimate from quality checks (wt%).

    Returns
    -------
    float
        σ for window mass loss (wt%).
    """
    return np.sqrt(2) * noise_estimate_pct


# ------------------------------------------------------------------
# Estimate σ from TGA quality report
# ------------------------------------------------------------------

def sigma_from_quality(quality_report) -> Dict[str, float]:
    """
    Extract uncertainty estimates from a QualityReport.

    Returns a dict of sigma values that can be fed into
    the propagation functions above.

    Parameters
    ----------
    quality_report : QualityReport
        From core.tga_quality.run_quality_checks()

    Returns
    -------
    dict with keys:
        sigma_mass_pct : noise-based mass uncertainty
        sigma_residue_pct : residue stability uncertainty
        sigma_window_loss_pct : window mass loss uncertainty
    """
    noise = quality_report.noise_estimate_pct

    # Residue stability: use the std of the tail
    sigma_res = noise  # conservative default
    for check in quality_report.checks:
        if check.name == "Residue stability":
            # The check stores range; std ≈ range/4 for normal dist
            sigma_res = check.value / 4.0
            break

    return {
        "sigma_mass_pct": noise,
        "sigma_residue_pct": max(sigma_res, noise),
        "sigma_window_loss_pct": window_mass_loss_uncertainty(noise),
    }