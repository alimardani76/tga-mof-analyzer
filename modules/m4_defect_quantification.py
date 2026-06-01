"""
Module 4 — Missing Linker Defect Quantification
==================================================
Answers: "How many linker defects does my MOF have?
          What is capping those open sites?"

The Multi-Compensator Engine
----------------------------
For each missing ditopic linker (charge −2), the framework needs +2
in compensating charge.  This module tests ALL chemically plausible
compensator models simultaneously:

    x = (M_ideal_DH − R_exp_DH × M_residue) / (M_linker − n_cap × M_cap)

    where:
      x           = number of missing linkers per formula unit
      M_ideal_DH  = molar mass of ideal dehydrated FU
      R_exp_DH × M_res = experimentally implied formula mass
      M_linker    = molar mass of the linker being removed
      n_cap       = compensating groups replacing one linker (usually 2)
      M_cap       = molar mass of one compensating group

Each compensator model gives a DIFFERENT value of x because the
net mass change per defect depends on what fills the vacancy.

M-400 Sanity Check
------------------
The M-400 method (Shearer et al. 2016) assumes all modulators are
gone by 400°C.  When they're not (benzoic acid, TFA, amino-BDC),
the calculation forces residual modulator mass into the "linker"
bucket, producing IMPOSSIBLE formulas where total coordination
exceeds the SBU maximum (e.g., >12 for Zr₆).

This module automatically checks: if total_coordination > max,
a warning is emitted suggesting the M-200 method.

CRITICAL CAVEAT
---------------
TGA CANNOT distinguish between:
  - Missing linker defects (linker absent, cluster intact)
  - Missing cluster defects (entire SBU absent with its linkers)
Both produce similar R_exp values.  Cross-validate with:
  - N₂ sorption (missing clusters → mesopores >2 nm)
  - HRTEM (direct imaging of cluster vacancies)

References
----------
- Shearer, G.C. et al. Chem. Mater. 2016, 28, 3749-3761.
  DOI: 10.1021/acs.chemmater.6b00602
- Sannes, D.K. et al. Chem. Mater. 2023, 35, 3793-3800.
  DOI: 10.1021/acs.chemmater.2c03744
- Liu, L. et al. Nature Chemistry 2019, 11, 622-628.
  DOI: 10.1038/s41557-019-0263-4
- Valenzano, L. et al. Chem. Mater. 2011, 23, 1700-1718.
  DOI: 10.1021/cm2007855
- Bueken, B. et al. Chem. Mater. 2017, 29, 10478-10486.
  DOI: 10.1021/acs.chemmater.7b04128
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.rexp import compute_linkers


# ------------------------------------------------------------------
# Compensator models
# ------------------------------------------------------------------

@dataclass
class CompensatorModel:
    """A charge-compensating species that replaces missing linkers.

    Attributes
    ----------
    name : str
        Human-readable name.
    n_cap_per_missing_linker : int
        Number of compensator units replacing one multidentate linker.
        Typically 2 for monocarboxylates replacing a dicarboxylate
        (each monocarboxylate provides −1 charge, linker provided −2).
    M_cap : float
        Molar mass of one compensating group (g/mol).
    charge_per_cap : float
        Charge per compensating group (negative for anions).
    """
    name: str
    n_cap_per_missing_linker: int
    M_cap: float
    charge_per_cap: float = -1.0


# The five standard compensator models
DEFAULT_COMPENSATORS = [
    CompensatorModel(
        name="OH⁻ / H₂O",
        n_cap_per_missing_linker=2,
        M_cap=17.008,          # OH⁻
        charge_per_cap=-1.0,
    ),
    CompensatorModel(
        name="Formate (HCO₂⁻)",
        n_cap_per_missing_linker=2,
        M_cap=45.017,          # HCO₂⁻
        charge_per_cap=-1.0,
    ),
    CompensatorModel(
        name="Acetate (CH₃CO₂⁻)",
        n_cap_per_missing_linker=2,
        M_cap=59.044,          # CH₃CO₂⁻
        charge_per_cap=-1.0,
    ),
    CompensatorModel(
        name="Chloride (Cl⁻)",
        n_cap_per_missing_linker=2,
        M_cap=35.453,          # Cl⁻
        charge_per_cap=-1.0,
    ),
    CompensatorModel(
        name="Vacancy (no cap)",
        n_cap_per_missing_linker=0,
        M_cap=0.0,
        charge_per_cap=0.0,
    ),
]


# ------------------------------------------------------------------
# Per-compensator result
# ------------------------------------------------------------------

@dataclass
class CompensatorResult:
    """Result for a single compensator model.

    Attributes
    ----------
    model : CompensatorModel
    x_missing : float
        Missing linkers per FU under this model.
    q_effective : float
        Actual linkers per FU = ideal_q − x_missing.
    n_compensators : float
        Total compensating groups per FU.
    formula_string : str
    charge_balanced : bool
    mass_discrepancy_pct : float
        Deviation between model-predicted and observed formula mass.
    """
    model: CompensatorModel
    x_missing: float
    q_effective: float
    n_compensators: float
    formula_string: str
    charge_balanced: bool
    mass_discrepancy_pct: float


# ------------------------------------------------------------------
# Main result container
# ------------------------------------------------------------------

@dataclass
class DefectResult:
    """Complete defect quantification output.

    Attributes
    ----------
    q_experimental : float
        Linkers per FU from R_exp (Module 3, no compensator assumed).
    q_ideal : float
        Ideal linkers in defect-free structure.
    n_missing_simple : float
        Simple difference: ideal − experimental.
    defect_pct : float
        Defect percentage: n_missing / q_ideal × 100.
    compensator_results : list of CompensatorResult
        Results for each compensator model tested.
    coordination_check_passed : bool
        True if total coordination ≤ SBU maximum.
    max_coordination : float
    total_coordination_exp : float
    warnings : list of str
    """
    q_experimental: float
    q_ideal: float
    n_missing_simple: float
    defect_pct: float
    compensator_results: List[CompensatorResult]
    coordination_check_passed: bool
    max_coordination: float
    total_coordination_exp: float
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary report."""
        lines = [
            "=" * 65,
            "MODULE 4: Defect Quantification (Missing Linkers)",
            "=" * 65,
            f"  Linkers (experimental):  {self.q_experimental:.3f}",
            f"  Linkers (ideal):         {self.q_ideal:.1f}",
            f"  Missing (simple):        {self.n_missing_simple:.3f}  "
            f"({self.defect_pct:.1f}%)",
            "",
            f"  Coordination check:      "
            f"{'✅ PASS' if self.coordination_check_passed else '❌ FAIL'}  "
            f"(exp = {self.total_coordination_exp:.1f}, "
            f"max = {self.max_coordination:.0f})",
            "",
            "  Multi-compensator analysis:",
            "  " + "-" * 61,
            f"  {'Model':<22} {'x_miss':>7} {'q_eff':>7} "
            f"{'n_cap':>7} {'Δm%':>7} {'Chg?':>5}",
            "  " + "-" * 61,
        ]

        for cr in self.compensator_results:
            chg = "✅" if cr.charge_balanced else "❌"
            lines.append(
                f"  {cr.model.name:<22} {cr.x_missing:>7.3f} "
                f"{cr.q_effective:>7.3f} {cr.n_compensators:>7.2f} "
                f"{cr.mass_discrepancy_pct:>7.3f} {chg:>5}"
            )

        lines.append("  " + "-" * 61)
        lines.append("")
        lines.append("  Formulas per model:")
        for cr in self.compensator_results:
            lines.append(f"    {cr.model.name:<22}  {cr.formula_string}")

        if self.warnings:
            lines.append("")
            lines.append("  ⚠️  WARNINGS:")
            for w in self.warnings:
                lines.append(f"     - {w}")

        lines.append("=" * 65)
        return "\n".join(lines)


# ------------------------------------------------------------------
# Core defect equation
# ------------------------------------------------------------------

def _compute_missing_for_model(
    M_ideal_dh: float,
    R_exp_dh: float,
    M_residue: float,
    M_linker: float,
    model: CompensatorModel,
) -> float:
    """
    Compute missing linkers under a specific compensator model.

    x = (M_ideal_DH − R_exp_DH × M_residue) / (M_linker − n_cap × M_cap)

    The numerator is the mass discrepancy between ideal and observed.
    The denominator is the net mass change per missing linker:
      mass of linker removed MINUS mass of compensator added.

    Parameters
    ----------
    M_ideal_dh : float
        Molar mass of ideal dehydrated FU (g/mol).
    R_exp_dh : float
        Experimental R_exp at DH plateau.
    M_residue : float
        Molar mass of residue per FU (g/mol).
    M_linker : float
        Molar mass of one linker (g/mol).
    model : CompensatorModel

    Returns
    -------
    float
        Number of missing linkers (x).
    """
    numerator = M_ideal_dh - R_exp_dh * M_residue
    denominator = M_linker - model.n_cap_per_missing_linker * model.M_cap

    if abs(denominator) < 1e-10:
        return float("nan")

    return numerator / denominator


# ------------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------------

def analyze_defects(
    composition_result=None,
    tga_data=None,
    components=None,
    r_exp_dh: Optional[float] = None,
    m_dh_pct: Optional[float] = None,
    m_residue_pct: Optional[float] = None,
    compensators: Optional[List[CompensatorModel]] = None,
    max_coordination: Optional[float] = None,
    ideal_q: Optional[float] = None,
) -> DefectResult:
    """
    Quantify missing linker defects across multiple compensator models.

    This function can be called in two ways:

    1. From Module 3 output (preferred):
       analyze_defects(composition_result=my_comp_result)

    2. From raw values:
       analyze_defects(components=my_components, r_exp_dh=2.05)

    Parameters
    ----------
    composition_result : CompositionResult or None
        If provided, extracts all needed values from Module 3 output.
    tga_data : TGAData or None
        Raw TGA data (fallback if composition_result not given).
    components : MOFComponents or None
        MOF specification (fallback).
    r_exp_dh : float or None
        Override R_exp_DH value.
    m_dh_pct, m_residue_pct : float or None
        Override DH plateau and residue masses (to compute R_exp_DH).
    compensators : list of CompensatorModel or None
        Models to test.  If None, uses DEFAULT_COMPENSATORS (5 models).
    max_coordination : float or None
        Maximum coordination number of the SBU.
        If None, inferred as 2 × ideal_q (works for most MOFs).
    ideal_q : float or None
        Ideal linkers per FU.  If None, from components.

    Returns
    -------
    DefectResult
    """
    warnings = []

    # --- Extract values ---
    if composition_result is not None:
        c = composition_result.components
        q_exp = composition_result.q_linkers
        _r_exp_dh = composition_result.r_exp_dh
        _ideal_q = composition_result.ideal_q
        p_mod = composition_result.p_modulators
    elif components is not None:
        c = components
        _ideal_q = ideal_q if ideal_q is not None else c.ideal_linkers

        # Determine R_exp_DH
        if r_exp_dh is not None:
            _r_exp_dh = r_exp_dh
        elif m_dh_pct is not None and m_residue_pct is not None:
            if m_residue_pct <= 0:
                raise ValueError("m_residue_pct must be > 0.")
            _r_exp_dh = m_dh_pct / m_residue_pct
        elif tga_data is not None:
            # Very rough fallback — not recommended
            m = tga_data.mass_pct
            _r_exp_dh = float(m[len(m) // 2]) / float(m[-1])
            warnings.append(
                "R_exp_DH was auto-estimated from the curve midpoint.  "
                "For accurate results, specify m_dh_pct or t_dh_plateau "
                "and run Module 3 first."
            )
        else:
            raise ValueError(
                "Must provide r_exp_dh, or (m_dh_pct + m_residue_pct), "
                "or composition_result."
            )

        q_exp = compute_linkers(
            _r_exp_dh, c.M_residue, c.M_node, c.M_linker,
        )
        p_mod = 0.0
    else:
        raise ValueError(
            "Either composition_result or (components + r_exp_dh) required."
        )

    # Allow override
    if r_exp_dh is not None:
        _r_exp_dh = r_exp_dh
    if ideal_q is not None:
        _ideal_q = ideal_q

    # --- Derived quantities ---
    M_res = c.M_residue
    M_ideal_dh = c.M_node + _ideal_q * c.M_linker
    n_missing = _ideal_q - q_exp
    defect_pct = (n_missing / _ideal_q * 100.0) if _ideal_q > 0 else 0.0

    # --- Multi-compensator engine ---
    if compensators is None:
        compensators = DEFAULT_COMPENSATORS

    comp_results = []
    for model in compensators:
        x = _compute_missing_for_model(
            M_ideal_dh, _r_exp_dh, M_res, c.M_linker, model,
        )

        q_eff = _ideal_q - x
        n_cap = x * model.n_cap_per_missing_linker

        # Formula string
        formula = f"{c.sbu_formula}({c.linker_name}){q_eff:.2f}"
        if model.n_cap_per_missing_linker > 0 and x > 0.01:
            # Extract a short label from the model name
            short_name = model.name.split("(")[0].strip()
            formula += f"({short_name}){n_cap:.2f}"

        # Charge balance check
        total_neg = abs(c.linker_charge) * q_eff
        if model.charge_per_cap != 0:
            total_neg += abs(model.charge_per_cap) * n_cap
        sbu_q = c.sbu_charge if c.sbu_charge is not None else 0
        chg_balanced = abs(sbu_q - total_neg) < 0.5

        # Mass discrepancy
        M_model = c.M_node + q_eff * c.M_linker + n_cap * model.M_cap
        M_exp = _r_exp_dh * M_res
        if M_exp > 0:
            mass_disc = (M_model - M_exp) / M_exp * 100.0
        else:
            mass_disc = 0.0

        comp_results.append(CompensatorResult(
            model=model,
            x_missing=x,
            q_effective=q_eff,
            n_compensators=n_cap,
            formula_string=formula,
            charge_balanced=chg_balanced,
            mass_discrepancy_pct=mass_disc,
        ))

    # --- Coordination sanity check ---
    if max_coordination is None:
        max_coordination = 2.0 * _ideal_q  # 12 for Zr6 with 6 ditopic linkers

    total_coord = abs(c.linker_charge) * q_exp
    if p_mod and c.M_modulator:
        total_coord += abs(c.mod_charge) * p_mod
    coord_ok = total_coord <= max_coordination + 0.5

    if not coord_ok:
        warnings.append(
            f"Total coordination ({total_coord:.1f}) EXCEEDS SBU maximum "
            f"({max_coordination:.0f}).  This is the hallmark of the M-400 "
            f"failure mode: incomplete modulator removal before the DH "
            f"plateau.  The mass attributed to 'linker' includes residual "
            f"modulator, inflating q_exp.\n"
            f"  → Consider using M-200 activation (200°C) with qNMR "
            f"support.  See: Sannes et al., Chem. Mater. 2023, 35, 3793."
        )

    # --- Mandatory caveats ---
    warnings.append(
        "IMPORTANT: TGA cannot distinguish missing-linker from missing-"
        "cluster defects.  The values above assume ALL defects are "
        "missing linkers (clusters intact).  If missing clusters are "
        "present, the true defect topology differs.\n"
        "  → Cross-validate with N₂ sorption (mesopores >2 nm indicate "
        "missing clusters) or HRTEM.\n"
        "  → See: Liu et al., Nature Chemistry 2019, 11, 622-628."
    )

    if n_missing < -0.1:
        warnings.append(
            f"Negative missing linkers ({n_missing:.3f}): the sample "
            f"has MORE linkers than ideal.  Possible causes:\n"
            f"  (a) Incomplete guest removal inflating the DH plateau mass\n"
            f"  (b) Incorrect residue formula\n"
            f"  (c) Residual inorganic species (Cl⁻, NO₃⁻) adding hidden "
            f"mass to the residue"
        )

    for cr in comp_results:
        if np.isnan(cr.x_missing):
            warnings.append(
                f"Model '{cr.model.name}': denominator is zero "
                f"(M_linker ≈ n_cap × M_cap).  This model is "
                f"mathematically degenerate for this linker."
            )

    return DefectResult(
        q_experimental=q_exp,
        q_ideal=_ideal_q,
        n_missing_simple=n_missing,
        defect_pct=defect_pct,
        compensator_results=comp_results,
        coordination_check_passed=coord_ok,
        max_coordination=max_coordination,
        total_coordination_exp=total_coord,
        warnings=warnings,
    )