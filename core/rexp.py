"""
R_exp Engine — Residue-Anchored Stoichiometry
==============================================
Core algebra for MOF composition determination from TGA data.

Central concept
---------------
    R_exp = m_MOF / m_residue

Since the metal oxide residue mass per formula unit (FU) is fixed
and predictable from the metal identity, R_exp encodes the total
organic-to-inorganic mass ratio.  All composition equations derive
from this single measurable quantity.

Master equation
---------------
    q = (R_exp_DH * M_residue - M_node) / M_linker

    where:
      q          = number of linkers per formula unit
      R_exp_DH   = m(DH-MOF) / m(residue)   — from TGA
      M_residue  = n_metals * M(metal oxide per atom)
      M_node     = molar mass of dehydrated inorganic node
      M_linker   = molar mass of one linker anion

When modulators are present (2 unknowns: q linkers, p modulators):
  Strategy A — NMR ratio: p = r * q, solve 1 equation / 1 unknown.
  Strategy B — Charge balance: |z_L|*q + |z_M|*p = Q_SBU,
               combined with R_exp -> 2 equations / 2 unknowns.

References
----------
- Abánades Lázaro, I. Eur. J. Inorg. Chem. 2020, 4284-4294.
  DOI: 10.1002/ejic.202000656
- Shearer, G.C. et al. Chem. Mater. 2016, 28, 3749-3761.
  DOI: 10.1021/acs.chemmater.6b00602
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


# ------------------------------------------------------------------
# Data container
# ------------------------------------------------------------------

@dataclass
class RexpResult:
    """
    Result of R_exp computation from a TGA curve.

    Attributes
    ----------
    r_exp : float
        m_initial / m_residue  (using wt% values).
    r_exp_dh : float
        m_DH_plateau / m_residue.
    m_initial_pct : float
        Initial mass in wt% (should be ~100%).
    m_dh_pct : float
        Mass at the dehydrated plateau in wt%.
    m_residue_pct : float
        Final residue mass in wt%.
    t_dh_plateau : float or None
        Temperature at which DH plateau was read (deg C).
    """
    r_exp: float
    r_exp_dh: float
    m_initial_pct: float
    m_dh_pct: float
    m_residue_pct: float
    t_dh_plateau: Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"RexpResult(R_exp={self.r_exp:.4f}, R_exp_DH={self.r_exp_dh:.4f}, "
            f"m_init={self.m_initial_pct:.2f}%, m_DH={self.m_dh_pct:.2f}%, "
            f"m_res={self.m_residue_pct:.2f}%, T_DH={self.t_dh_plateau})"
        )


# ------------------------------------------------------------------
# R_exp computation
# ------------------------------------------------------------------

def compute_rexp(
    tga_data,
    m_residue_pct: Optional[float] = None,
    m_dh_pct: Optional[float] = None,
    t_dh_plateau: Optional[float] = None,
) -> RexpResult:
    """
    Compute experimental R_exp values from TGA data.

    Parameters
    ----------
    tga_data : TGAData
        Parsed and normalized TGA data.
    m_residue_pct : float or None
        Residue mass in wt%.  If None, uses the last data point.
    m_dh_pct : float or None
        Mass at the dehydrated (DH) plateau in wt%.
        If None and t_dh_plateau is given, interpolated from data.
        If both are None, defaults to the initial mass (no DH correction).
    t_dh_plateau : float or None
        Temperature of the DH plateau (deg C).  Used to look up m_dh_pct.

    Returns
    -------
    RexpResult

    Notes
    -----
    The DH (dehydrated/dehydroxylated) MOF is the state after all guests,
    water, and hydroxyl groups have been removed but before linker
    combustion begins.  Typically visible as a plateau in the TGA curve
    between ~250-400 deg C for Zr-MOFs, ~200-300 deg C for Zn-MOFs.
    """
    m_initial_pct = float(tga_data.mass_pct[0])

    if m_residue_pct is None:
        m_residue_pct = float(tga_data.mass_pct[-1])

    if m_residue_pct <= 0:
        raise ValueError("Residue mass must be > 0 wt%.")

    # DH plateau mass
    if m_dh_pct is not None:
        pass  # User provided directly
    elif t_dh_plateau is not None:
        m_dh_pct = tga_data.get_mass_at_temp(t_dh_plateau)
    else:
        m_dh_pct = m_initial_pct  # Fallback: no guest correction

    r_exp = m_initial_pct / m_residue_pct
    r_exp_dh = m_dh_pct / m_residue_pct

    return RexpResult(
        r_exp=r_exp,
        r_exp_dh=r_exp_dh,
        m_initial_pct=m_initial_pct,
        m_dh_pct=m_dh_pct,
        m_residue_pct=m_residue_pct,
        t_dh_plateau=t_dh_plateau,
    )


# ------------------------------------------------------------------
# Linker count — simple case (no modulators)
# ------------------------------------------------------------------

def compute_linkers(
    r_exp_dh: float,
    M_residue: float,
    M_node: float,
    M_linker: float,
) -> float:
    """
    Compute number of linkers per formula unit from R_exp_DH.

    Master equation:
        q = (R_exp_DH * M_residue - M_node) / M_linker

    Parameters
    ----------
    r_exp_dh : float
        Experimental ratio of DH-MOF mass to residue mass.
    M_residue : float
        Molar mass of the metal oxide residue per FU (g/mol).
        Example: for Zr6 cluster -> 6 * M(ZrO2) = 6 * 123.22 = 739.3
    M_node : float
        Molar mass of the dehydrated inorganic node (g/mol).
        Example: Zr6O6 = 6*91.22 + 6*16.00 = 643.3
    M_linker : float
        Molar mass of ONE linker anion (g/mol).
        Example: BDC2- = C8H4O4 = 164.1

    Returns
    -------
    float
        Number of linkers per formula unit (q).
        Ideal UiO-66 -> 6.0.

    Notes
    -----
    Derivation:
        M_DH  = M_node + q * M_linker
        M_res = known (from metal oxide stoichiometry)
        R_exp_DH = M_DH / M_res
        => q = (R_exp_DH * M_res - M_node) / M_linker

    Assumptions:
    1. Complete combustion to the expected metal oxide.
    2. No modulators present in the DH-MOF.
    3. Residue is phase-pure metal oxide.
    """
    if M_linker <= 0:
        raise ValueError("Linker molar mass must be > 0.")
    if M_residue <= 0:
        raise ValueError("Residue molar mass must be > 0.")

    q = (r_exp_dh * M_residue - M_node) / M_linker
    return q


# ------------------------------------------------------------------
# Linker count — with modulators (2 unknowns)
# ------------------------------------------------------------------

def compute_linkers_with_modulator(
    r_exp_dh: float,
    M_residue: float,
    M_node: float,
    M_linker: float,
    M_modulator: float,
    mod_linker_ratio: Optional[float] = None,
    sbu_charge: Optional[float] = None,
    linker_charge: float = -2.0,
    mod_charge: float = -1.0,
) -> Dict:
    """
    Compute linkers (q) and modulators (p) per FU when both are present.

    The DH-MOF formula:  [Node] * (Linker)_q * (Modulator)_p

        M_DH = M_node + q * M_linker + p * M_modulator
        R_exp_DH = M_DH / M_residue

    One equation, two unknowns.  Need a second constraint.

    Strategy A — NMR molar ratio (preferred):
        Given r = p/q from digestion NMR:
        q = (R_exp_DH * M_res - M_node) / (M_linker + r * M_modulator)
        p = r * q

    Strategy B — Charge balance:
        |linker_charge| * q + |mod_charge| * p = |sbu_charge|
        Combined with R_exp equation -> solve 2x2 linear system.

    Parameters
    ----------
    r_exp_dh : float
    M_residue, M_node, M_linker, M_modulator : float
        Molar masses (g/mol).
    mod_linker_ratio : float or None
        p/q ratio from NMR (Strategy A).
    sbu_charge : float or None
        Total positive charge to balance (Strategy B).
    linker_charge : float
        Charge per linker (default -2).
    mod_charge : float
        Charge per modulator (default -1).

    Returns
    -------
    dict with keys:
        'q' : float — linkers per FU
        'p' : float — modulators per FU
        'method' : str — 'nmr_ratio', 'charge_balance', or 'error'
        'M_DH' : float — computed DH formula mass
        'charge_balance_residual' : float or None
    """
    result = {
        'q': None, 'p': None, 'method': 'error',
        'M_DH': None, 'charge_balance_residual': None,
    }

    # Strategy A: NMR ratio
    if mod_linker_ratio is not None:
        r = mod_linker_ratio  # p/q
        denom = M_linker + r * M_modulator
        if abs(denom) < 1e-10:
            result['method'] = 'error: denominator near zero'
            return result

        q = (r_exp_dh * M_residue - M_node) / denom
        p = r * q

        result['q'] = q
        result['p'] = p
        result['method'] = 'nmr_ratio'
        result['M_DH'] = M_node + q * M_linker + p * M_modulator

        # Optional cross-check
        if sbu_charge is not None:
            charge_used = abs(linker_charge) * q + abs(mod_charge) * p
            result['charge_balance_residual'] = abs(sbu_charge) - charge_used

        return result

    # Strategy B: Charge balance
    if sbu_charge is not None:
        # Eq 1 (mass):   M_linker * q + M_modulator * p = R_exp_DH * M_res - M_node
        # Eq 2 (charge): |z_L| * q + |z_M| * p = |Q_SBU|
        A_val = r_exp_dh * M_residue - M_node
        C_val = abs(sbu_charge)
        a1, b1 = M_linker, M_modulator
        a2, b2 = abs(linker_charge), abs(mod_charge)

        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            result['method'] = 'error: singular system (parallel constraints)'
            return result

        q = (A_val * b2 - C_val * b1) / det
        p = (a1 * C_val - a2 * A_val) / det

        result['q'] = q
        result['p'] = p
        result['method'] = 'charge_balance'
        result['M_DH'] = M_node + q * M_linker + p * M_modulator
        result['charge_balance_residual'] = 0.0  # Exact by construction

        return result

    # Fallback: neither constraint provided
    result['method'] = 'error: need mod_linker_ratio or sbu_charge'
    return result


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def compute_formula_mass(
    M_node: float,
    q: float,
    M_linker: float,
    p: float = 0.0,
    M_modulator: float = 0.0,
) -> float:
    """
    Compute molar mass of the DH-MOF formula unit.

    M_DH = M_node + q * M_linker + p * M_modulator
    """
    return M_node + q * M_linker + p * M_modulator


def compute_theoretical_rexp(
    M_node: float,
    q: float,
    M_linker: float,
    M_residue: float,
    p: float = 0.0,
    M_modulator: float = 0.0,
) -> float:
    """
    Compute theoretical R_exp_DH for a given formula.

    R_theo = (M_node + q*M_linker + p*M_mod) / M_residue

    Useful for comparing expected vs. measured R_exp.
    """
    M_DH = compute_formula_mass(M_node, q, M_linker, p, M_modulator)
    return M_DH / M_residue


# ------------------------------------------------------------------
# R_exp curve and plateau finder (v0.2 additions)
# ------------------------------------------------------------------

def compute_rexp_curve(
    tga_data,
    m_residue_pct: Optional[float] = None,
) -> Dict:
    """
    Compute R_exp as a continuous function of temperature.

    R_exp(T) = mass(T) / mass(residue)

    This is more informative than a single R_exp value because
    plateaus in the R_exp curve correspond to chemically distinct
    states of the MOF (with guests, without guests, dehydrated, etc.).

    Parameters
    ----------
    tga_data : TGAData
        Parsed TGA data.
    m_residue_pct : float or None
        Residue mass in wt%.  If None, uses the last data point.

    Returns
    -------
    dict with keys:
        'temperature' : np.ndarray — temperature axis (deg C)
        'r_exp' : np.ndarray — R_exp at each temperature
        'dr_dt' : np.ndarray — dR/dT at each temperature
    """
    T = tga_data.temperature
    m = tga_data.mass_pct

    if m_residue_pct is None:
        m_residue_pct = float(m[-1])

    if m_residue_pct <= 0:
        raise ValueError("Residue mass must be > 0.")

    r_exp = m / m_residue_pct

    # Numerical derivative of R_exp
    dr_dt = np.gradient(r_exp, T)

    return {
        'temperature': T,
        'r_exp': r_exp,
        'dr_dt': dr_dt,
    }


def find_dh_plateau(
    tga_data,
    m_residue_pct: Optional[float] = None,
    search_min_C: float = 150.0,
    search_max_C: float = 500.0,
    window_C: float = 30.0,
) -> Dict:
    """
    Automatically find the dehydrated (DH) plateau in a TGA curve.

    The DH plateau is the temperature range where the R_exp curve
    is flattest — meaning nothing is entering or leaving the MOF.
    This corresponds to the fully activated, guest-free framework
    before linker decomposition begins.

    Algorithm
    ---------
    1. Compute R_exp(T) at every point.
    2. Compute |dR/dT| — the absolute rate of change.
    3. Slide a window of width window_C across the search range.
    4. For each window position, compute mean |dR/dT|.
    5. The window with minimum mean |dR/dT| is the flattest region.
    6. Report the center temperature and the flatness metric.

    Parameters
    ----------
    tga_data : TGAData
    m_residue_pct : float or None
        Residue mass in wt%.
    search_min_C : float
        Lower bound of search range (deg C).  Default 150.
    search_max_C : float
        Upper bound of search range (deg C).  Default 500.
    window_C : float
        Width of the sliding window (deg C).  Default 30.

    Returns
    -------
    dict with keys:
        'suggested_temp' : float — center of the flattest window (deg C)
        'suggested_range' : tuple — (T_lo, T_hi) of the flattest window
        'r_exp_at_plateau' : float — R_exp value at the suggested temp
        'mass_at_plateau' : float — mass (wt%) at the suggested temp
        'flatness' : float — mean |dR/dT| in the window (lower = flatter)
        'is_reliable' : bool — True if flatness < 0.002 (genuinely flat)
        'warning' : str or None — warning message if plateau is unreliable
    """
    T = tga_data.temperature
    m = tga_data.mass_pct

    if m_residue_pct is None:
        m_residue_pct = float(m[-1])

    if m_residue_pct <= 0:
        raise ValueError("Residue mass must be > 0.")

    r_exp = m / m_residue_pct
    dr_dt = np.gradient(r_exp, T)
    abs_dr = np.abs(dr_dt)

    # Restrict to search range
    mask = (T >= search_min_C) & (T <= search_max_C)
    T_search = T[mask]
    abs_dr_search = abs_dr[mask]

    if len(T_search) < 5:
        return {
            'suggested_temp': (search_min_C + search_max_C) / 2,
            'suggested_range': (search_min_C, search_max_C),
            'r_exp_at_plateau': float(np.interp(
                (search_min_C + search_max_C) / 2, T, r_exp)),
            'mass_at_plateau': float(np.interp(
                (search_min_C + search_max_C) / 2, T, m)),
            'flatness': float("inf"),
            'is_reliable': False,
            'warning': "Insufficient data points in search range.",
        }

    # Sliding window: find the window with minimum mean |dR/dT|
    best_flatness = float("inf")
    best_center = float(T_search[len(T_search) // 2])

    for i in range(len(T_search)):
        t_center = T_search[i]
        t_lo = t_center - window_C / 2
        t_hi = t_center + window_C / 2

        window_mask = (T_search >= t_lo) & (T_search <= t_hi)
        if np.sum(window_mask) < 3:
            continue

        mean_abs_dr = float(np.mean(abs_dr_search[window_mask]))

        if mean_abs_dr < best_flatness:
            best_flatness = mean_abs_dr
            best_center = float(t_center)

    # Read values at the best center
    r_at_plateau = float(np.interp(best_center, T, r_exp))
    m_at_plateau = float(np.interp(best_center, T, m))
    t_lo_best = best_center - window_C / 2
    t_hi_best = best_center + window_C / 2

    # Reliability check
    is_reliable = best_flatness < 0.002
    warning = None
    if not is_reliable:
        warning = (
            f"No genuinely flat DH plateau found (flatness = "
            f"{best_flatness:.4f}, threshold = 0.002).  "
            f"This sample may have overlapping guest removal and "
            f"framework decomposition.  The suggested temperature "
            f"({best_center:.0f} deg C) is the LEAST SLOPED region, "
            f"not a true plateau.  Consider specifying t_dh_plateau "
            f"manually based on DTG inspection, or use window-based "
            f"analysis instead."
        )

    return {
        'suggested_temp': best_center,
        'suggested_range': (t_lo_best, t_hi_best),
        'r_exp_at_plateau': r_at_plateau,
        'mass_at_plateau': m_at_plateau,
        'flatness': best_flatness,
        'is_reliable': is_reliable,
        'warning': warning,
    }