"""
Module 3 — Framework Composition / Molecular Formula Determination
====================================================================
Answers: "What is the actual chemical formula of my MOF?  How many
          linkers per metal cluster?  Does it match the ideal?"

This is the CORE MODULE — the most scientifically valuable thing
this toolkit does.

The Master Equation
-------------------
    q = (R_exp_DH × M_residue − M_node) / M_linker

    where:
      R_exp_DH  = m(DH-MOF) / m(residue)     — measured from TGA
      M_residue = n_metals × M(metal oxide)   — known from metal identity
      M_node    = molar mass of dehydrated SBU — known from crystal structure
      M_linker  = molar mass of linker anion   — known from chemistry

When modulators are present (2 unknowns: q linkers, p modulators):
  Strategy A — NMR ratio:  p/q known → 1 equation, 1 unknown.
  Strategy B — Charge balance:  |z_L|×q + |z_M|×p = Q_SBU
               Combined with R_exp → 2 equations, 2 unknowns.
  Strategy C — DTG separation: if modulator has a distinct decomposition
               step, its mass loss gives p×M_mod directly.

Worked Example — UiO-66(Zr)
----------------------------
  Ideal: Zr6O4(OH)4(BDC)6
  After dehydroxylation (~300°C): Zr6O6(BDC)6
    M_node   = 6×91.224 + 6×16.00  = 643.3 g/mol
    M_linker = 164.1 g/mol  (BDC²⁻ = C8H4O4)
    M_residue = 6 × 123.22 = 739.3 g/mol  (6 ZrO2)
    R_theo_DH = (643.3 + 6×164.1) / 739.3 = 2.202

  If experimental R_exp_DH = 2.05:
    q = (2.05 × 739.3 − 643.3) / 164.1 = 5.32
    → 0.68 missing linkers per node → ~11% linker deficiency

Limitations (TGA-only, "Bronze tier")
--------------------------------------
- Requires TGA in AIR to achieve full combustion to metal oxide.
- Residue must be pure, known metal oxide (verify by XRD if possible).
- Hidden inorganic ions (Cl⁻, NO₃⁻) from metal precursors are NOT
  detected by TGA alone.  See: Pulparayil Mathew et al., Adv. Sci.
  2025, DOI: 10.1002/advs.202504713.
- Fe-based MOFs may show mass GAIN (Fe²⁺→Fe³⁺ oxidation).
- Zn/Cd MOFs: metal may volatilize at very high T under N2.

References
----------
- Abánades Lázaro, I. Eur. J. Inorg. Chem. 2020, 4284-4294.
  DOI: 10.1002/ejic.202000656
- Shearer, G.C. et al. Chem. Mater. 2016, 28, 3749-3761.
  DOI: 10.1021/acs.chemmater.6b00602
- Sannes, D.K. et al. Chem. Mater. 2023, 35, 3793-3800.
  DOI: 10.1021/acs.chemmater.2c03744
- Pulparayil Mathew, J. et al. Adv. Sci. 2025, 12, e04713.
  DOI: 10.1002/advs.202504713
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.rexp import (
    compute_rexp,
    compute_linkers,
    compute_linkers_with_modulator,
    compute_formula_mass,
    compute_theoretical_rexp,
)
from core.charge_balance import (
    check_charge_balance,
    compute_compensator_needed,
    ChargeBalanceResult,
)


# ------------------------------------------------------------------
# MOF component specification
# ------------------------------------------------------------------

@dataclass
class MOFComponents:
    """Complete specification of a MOF's building blocks.

    All molar masses in g/mol.  Charges are signed (linker_charge is
    typically negative, sbu_charge is positive).

    Attributes
    ----------
    metal : str
        Element symbol (e.g., 'Zr', 'Zn', 'Cu').
    n_metals_per_sbu : int
        Number of metal atoms per secondary building unit.
    sbu_formula : str
        Formula of the dehydrated SBU (e.g., 'Zr6O6').
    M_node : float
        Molar mass of the dehydrated inorganic node (g/mol).
    linker_name : str
        Short name of the linker (e.g., 'BDC', 'BTC').
    M_linker : float
        Molar mass of ONE linker anion as coordinated (g/mol).
    linker_charge : float
        Formal charge per linker (e.g., -2 for dicarboxylates).
    residue_formula : str
        Expected metal oxide residue (e.g., 'ZrO2').
    M_residue_per_metal : float
        Molar mass of residue per metal atom (g/mol).
    modulator_name : str or None
        Short name of modulator if used (e.g., 'formate', 'acetate').
    M_modulator : float or None
        Molar mass of ONE modulator anion (g/mol).
    mod_charge : float
        Formal charge per modulator (default -1).
    sbu_charge : float or None
        Total positive charge of the SBU that must be balanced
        by linkers + modulators + compensators.
    """
    metal: str
    n_metals_per_sbu: int
    sbu_formula: str
    M_node: float
    linker_name: str
    M_linker: float
    linker_charge: float = -2.0
    residue_formula: str = ""
    M_residue_per_metal: float = 0.0
    modulator_name: Optional[str] = None
    M_modulator: Optional[float] = None
    mod_charge: float = -1.0
    sbu_charge: Optional[float] = None

    def __post_init__(self):
        """Compute total residue mass per formula unit."""
        self.M_residue = self.n_metals_per_sbu * self.M_residue_per_metal

    @property
    def R_theo_dh(self) -> float:
        """Theoretical R_exp_DH for ideal defect-free MOF (no modulators)."""
        iq = self.ideal_linkers
        M_ideal = self.M_node + iq * self.M_linker
        if self.M_residue <= 0:
            return float("nan")
        return M_ideal / self.M_residue

    @property
    def ideal_linkers(self) -> float:
        """Ideal number of linkers for defect-free structure."""
        if (self.sbu_charge is not None
                and abs(self.linker_charge) > 0):
            return abs(self.sbu_charge) / abs(self.linker_charge)
        return 6.0  # Fallback for common 12-connected Zr MOFs

    @property
    def M_ideal_dh(self) -> float:
        """Molar mass of ideal dehydrated formula unit (g/mol)."""
        return self.M_node + self.ideal_linkers * self.M_linker


# ------------------------------------------------------------------
# Preset MOF component libraries
# ------------------------------------------------------------------

UIO_66_ZR = MOFComponents(
    metal="Zr",
    n_metals_per_sbu=6,
    sbu_formula="Zr6O6",
    M_node=6 * 91.224 + 6 * 15.999,          # 643.34
    linker_name="BDC",
    M_linker=164.116,                          # C8H4O4(2-)
    linker_charge=-2.0,
    residue_formula="ZrO2",
    M_residue_per_metal=123.218,               # ZrO2
    sbu_charge=12.0,
)

UIO_67_ZR = MOFComponents(
    metal="Zr",
    n_metals_per_sbu=6,
    sbu_formula="Zr6O6",
    M_node=6 * 91.224 + 6 * 15.999,
    linker_name="BPDC",
    M_linker=240.212,                          # C14H8O4(2-)
    linker_charge=-2.0,
    residue_formula="ZrO2",
    M_residue_per_metal=123.218,
    sbu_charge=12.0,
)

MOF_808_ZR = MOFComponents(
    metal="Zr",
    n_metals_per_sbu=6,
    sbu_formula="Zr6O6",
    M_node=6 * 91.224 + 6 * 15.999,
    linker_name="BTC",
    M_linker=208.124,                          # C9H3O6(3-)
    linker_charge=-3.0,
    residue_formula="ZrO2",
    M_residue_per_metal=123.218,
    modulator_name="formate",
    M_modulator=45.017,                        # HCO2(-)
    mod_charge=-1.0,
    sbu_charge=12.0,
)

MOF_5_ZN = MOFComponents(
    metal="Zn",
    n_metals_per_sbu=4,
    sbu_formula="Zn4O",
    M_node=4 * 65.38 + 15.999,                # 277.52
    linker_name="BDC",
    M_linker=164.116,
    linker_charge=-2.0,
    residue_formula="ZnO",
    M_residue_per_metal=81.379,
    sbu_charge=6.0,
)

HKUST_1_CU = MOFComponents(
    metal="Cu",
    n_metals_per_sbu=2,
    sbu_formula="Cu2",
    M_node=2 * 63.546,                        # 127.09
    linker_name="BTC",
    M_linker=208.124,
    linker_charge=-3.0,
    residue_formula="CuO",
    M_residue_per_metal=79.545,
    sbu_charge=4.0,
)

MIL_125_TI = MOFComponents(
    metal="Ti",
    n_metals_per_sbu=8,
    sbu_formula="Ti8O8(OH)4",
    M_node=8 * 47.867 + 8 * 15.999 + 4 * 17.008,  # 578.96
    linker_name="BDC",
    M_linker=164.116,
    linker_charge=-2.0,
    residue_formula="TiO2",
    M_residue_per_metal=79.866,
    sbu_charge=12.0,
)

ZIF_8_ZN = MOFComponents(
    metal="Zn",
    n_metals_per_sbu=1,
    sbu_formula="Zn",
    M_node=65.38,
    linker_name="2-mIm",
    M_linker=81.076,                           # C4H5N2(-)
    linker_charge=-1.0,
    residue_formula="ZnO",
    M_residue_per_metal=81.379,
    sbu_charge=2.0,
)

MIL_53_AL = MOFComponents(
    metal="Al",
    n_metals_per_sbu=1,
    sbu_formula="AlOH",
    M_node=26.982 + 16.00 + 1.008,            # 43.99
    linker_name="BDC",
    M_linker=164.116,
    linker_charge=-2.0,
    residue_formula="Al2O3",
    M_residue_per_metal=50.982,                # Al2O3 / 2
    sbu_charge=2.0,
)

PRESET_MOFS = {
    "UiO-66(Zr)":   UIO_66_ZR,
    "UiO-67(Zr)":   UIO_67_ZR,
    "MOF-808(Zr)":  MOF_808_ZR,
    "MOF-5(Zn)":    MOF_5_ZN,
    "HKUST-1(Cu)":  HKUST_1_CU,
    "MIL-125(Ti)":  MIL_125_TI,
    "ZIF-8(Zn)":    ZIF_8_ZN,
    "MIL-53(Al)":   MIL_53_AL,
}


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class CompositionResult:
    """Result of MOF composition analysis.

    Attributes
    ----------
    components : MOFComponents
        The MOF building blocks used.
    r_exp : float
        Experimental R_exp (m_initial / m_residue).
    r_exp_dh : float
        Experimental R_exp at DH plateau.
    q_linkers : float
        Calculated linkers per formula unit.
    p_modulators : float or None
        Calculated modulators per FU (if applicable).
    method_used : str
        How the system was solved.
    M_formula_exp : float
        Experimental DH formula mass (g/mol).
    M_formula_ideal : float
        Ideal (defect-free) DH formula mass (g/mol).
    formula_string : str
        Human-readable experimental formula.
    charge_balance : ChargeBalanceResult
    n_compensator : float
        Compensating species needed (OH⁻, etc.).
    ideal_q : float
    warnings : list of str
    """
    components: MOFComponents
    r_exp: float
    r_exp_dh: float
    q_linkers: float
    p_modulators: Optional[float]
    method_used: str
    M_formula_exp: float
    M_formula_ideal: float
    formula_string: str
    charge_balance: ChargeBalanceResult
    n_compensator: float
    ideal_q: float
    warnings: List[str] = field(default_factory=list)

    @property
    def deficiency_pct(self) -> float:
        """Linker deficiency as percent of ideal."""
        if self.ideal_q > 0:
            return (self.ideal_q - self.q_linkers) / self.ideal_q * 100.0
        return 0.0

    def summary(self) -> str:
        """Human-readable summary report."""
        c = self.components
        lines = [
            "=" * 60,
            "MODULE 3: Composition / Molecular Formula",
            "=" * 60,
            f"  MOF system:          {c.sbu_formula} + {c.linker_name}",
            f"  Metal:               {c.metal} "
            f"(n = {c.n_metals_per_sbu} per SBU)",
            f"  Residue:             {c.residue_formula} "
            f"(M = {c.M_residue:.2f} g/mol per FU)",
            "",
            "  R_exp values:",
            f"    R_exp (as-synth):  {self.r_exp:.4f}",
            f"    R_exp_DH:          {self.r_exp_dh:.4f}",
            f"    R_theo_DH (ideal): {c.R_theo_dh:.4f}",
            "",
            f"  Linkers per FU:      {self.q_linkers:.3f}  "
            f"(ideal: {self.ideal_q:.1f})",
        ]

        if self.p_modulators is not None:
            lines.append(
                f"  Modulators per FU:   {self.p_modulators:.3f}  "
                f"({c.modulator_name})"
            )

        lines.extend([
            f"  Compensators (OH⁻):  {self.n_compensator:.2f}",
            f"  Deficiency:          {self.deficiency_pct:.1f}%",
            f"  Method:              {self.method_used}",
            "",
            f"  Experimental DH formula mass: {self.M_formula_exp:.1f} g/mol",
            f"  Ideal DH formula mass:        {self.M_formula_ideal:.1f} g/mol",
            "",
            f"  Formula: {self.formula_string}",
            "",
            f"  Charge balance: {self.charge_balance}",
        ])

        if self.warnings:
            lines.append("")
            lines.append("  ⚠️  WARNINGS:")
            for w in self.warnings:
                lines.append(f"     - {w}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ------------------------------------------------------------------
# Formula string builder
# ------------------------------------------------------------------

def _build_formula_string(
    c: MOFComponents,
    q: float,
    p: Optional[float],
    n_comp: float,
) -> str:
    """Build human-readable formula string."""
    parts = [f"{c.sbu_formula}"]
    parts.append(f"({c.linker_name}){q:.2f}")
    if p is not None and c.modulator_name and abs(p) > 0.01:
        parts.append(f"({c.modulator_name}){p:.2f}")
    if n_comp > 0.05:
        parts.append(f"(OH){n_comp:.2f}")
    return "".join(parts)


# ------------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------------

def analyze_composition(
    tga_data,
    components: MOFComponents,
    m_residue_pct: Optional[float] = None,
    m_dh_pct: Optional[float] = None,
    t_dh_plateau: Optional[float] = None,
    nmr_mod_linker_ratio: Optional[float] = None,
    ideal_q: Optional[float] = None,
    auto_find_plateau: bool = False,
) -> CompositionResult:
    """
    Determine MOF molecular formula from TGA data.

    Parameters
    ----------
    tga_data : TGAData
        Parsed TGA data.  Must be in AIR and reach full combustion
        (~800°C) for reliable residue.
    components : MOFComponents
        Specification of MOF building blocks.
    m_residue_pct : float or None
        Final residue mass in wt%.  If None, uses the last data point.
    m_dh_pct : float or None
        Mass at the dehydrated plateau in wt%.
        If None, t_dh_plateau or auto_find_plateau must be used.
    t_dh_plateau : float or None
        Temperature (°C) at which to read the DH plateau mass.
    nmr_mod_linker_ratio : float or None
        NMR-derived molar ratio p/q (modulators per linker).
    ideal_q : float or None
        Expected linkers in defect-free structure.
    auto_find_plateau : bool
        If True and both m_dh_pct and t_dh_plateau are None,
        automatically find the DH plateau using the R_exp curve
        flatness method.  Default False.

    Returns
    -------
    CompositionResult
    """
    T = tga_data.temperature
    m = tga_data.mass_pct
    c = components
    warnings = []

    # --- Ideal linkers ---
    if ideal_q is None:
        ideal_q = c.ideal_linkers

    # --- Read key mass values ---
    if m_residue_pct is None:
        m_residue_pct = float(m[-1])
    if m_residue_pct <= 0:
        raise ValueError(
            "Residue mass is zero or negative.  TGA data may be "
            "incomplete (didn't reach full combustion) or wrong "
            "atmosphere was used."
        )

    # --- Determine DH plateau mass ---
    if m_dh_pct is not None:
        pass  # User provided directly
    elif t_dh_plateau is not None:
        m_dh_pct = float(np.interp(t_dh_plateau, T, m))
    elif auto_find_plateau:
        # v0.2: automatic plateau detection
        from core.rexp import find_dh_plateau
        plateau = find_dh_plateau(tga_data, m_residue_pct=m_residue_pct)
        t_dh_plateau = plateau['suggested_temp']
        m_dh_pct = plateau['mass_at_plateau']

        if plateau['is_reliable']:
            warnings.append(
                f"DH plateau auto-detected at {t_dh_plateau:.1f}°C "
                f"(mass = {m_dh_pct:.2f}%, flatness = "
                f"{plateau['flatness']:.4f}).  Verify this matches "
                f"a visible plateau in your TGA/DTG curve."
            )
        else:
            warnings.append(
                f"⚠️ AUTO-PLATEAU WARNING: {plateau['warning']}"
            )
    else:
        raise ValueError(
            "Either m_dh_pct, t_dh_plateau, or auto_find_plateau=True "
            "must be specified.\n"
            "This is the mass at the dehydrated plateau — after guest "
            "loss but before linker combustion.  Check your DTG for a "
            "plateau region, typically 300-450°C for Zr-MOFs.\n"
            "Or set auto_find_plateau=True to let the toolkit find it."
        )

    # --- Compute R_exp ---
    r_exp = float(m[0]) / m_residue_pct
    r_exp_dh = m_dh_pct / m_residue_pct
    M_res = c.M_residue

    # --- Determine if modulators are present ---
    has_modulator = (
        c.M_modulator is not None
        and c.M_modulator > 0
        and c.modulator_name is not None
    )

    # --- Solve for composition ---
    p_mod = None
    method = "linker_only"

    if not has_modulator:
        q = compute_linkers(r_exp_dh, M_res, c.M_node, c.M_linker)
        method = "linker_only"
    else:
        result = compute_linkers_with_modulator(
            r_exp_dh=r_exp_dh,
            M_residue=M_res,
            M_node=c.M_node,
            M_linker=c.M_linker,
            M_modulator=c.M_modulator,
            mod_linker_ratio=nmr_mod_linker_ratio,
            sbu_charge=c.sbu_charge,
            linker_charge=c.linker_charge,
            mod_charge=c.mod_charge,
        )

        if result["q"] is not None:
            q = result["q"]
            p_mod = result["p"]
            method = result["method"]
        else:
            q = compute_linkers(r_exp_dh, M_res, c.M_node, c.M_linker)
            method = f"linker_only (fallback: {result['method']})"
            warnings.append(
                f"Could not resolve modulator content: {result['method']}. "
                f"Falling back to linker-only calculation.  Provide "
                f"nmr_mod_linker_ratio or sbu_charge for accurate results."
            )

    # --- Charge balance ---
    n_comp = 0.0
    if c.sbu_charge is not None:
        n_comp = compute_compensator_needed(
            sbu_charge=c.sbu_charge,
            q=q,
            linker_charge=c.linker_charge,
            p=p_mod if p_mod is not None else 0.0,
            mod_charge=c.mod_charge if has_modulator else 0.0,
            comp_charge=-1.0,
        )

    cb = check_charge_balance(
        sbu_charge=c.sbu_charge or 0,
        q=q,
        linker_charge=c.linker_charge,
        p=p_mod if p_mod is not None else 0.0,
        mod_charge=c.mod_charge if has_modulator else 0.0,
        n_compensator=max(n_comp, 0),
        comp_charge=-1.0,
    )

    # --- Formula mass ---
    M_exp = c.M_node + q * c.M_linker
    if p_mod is not None and c.M_modulator:
        M_exp += p_mod * c.M_modulator
    M_ideal = c.M_node + ideal_q * c.M_linker

    # --- Formula string ---
    formula = _build_formula_string(c, q, p_mod, max(n_comp, 0))

    # --- Warnings ---
    if q > ideal_q + 0.1:
        warnings.append(
            f"Computed linkers ({q:.2f}) EXCEEDS ideal ({ideal_q:.1f}).  "
            f"This typically indicates: (a) incomplete guest removal "
            f"before the DH plateau — the mass is too high, inflating "
            f"R_exp_DH; or (b) incorrect residue formula — the expected "
            f"oxide may not match what actually formed."
        )

    if q < 0:
        warnings.append(
            f"Computed linkers ({q:.2f}) is NEGATIVE.  Check that "
            f"m_dh_pct ({m_dh_pct:.2f}%) > m_residue_pct "
            f"({m_residue_pct:.2f}%) and that the residue formula "
            f"({c.residue_formula}) and node formula ({c.sbu_formula}) "
            f"are correct."
        )

    if not cb.balanced and c.sbu_charge is not None:
        warnings.append(
            f"Charge imbalance: residual = {cb.residual:+.2f}.  "
            f"Formula may require additional compensating species "
            f"beyond OH⁻, or the linker/modulator charges may be wrong."
        )

    if n_comp < -0.5:
        warnings.append(
            f"Negative compensator count ({n_comp:.2f}) suggests the "
            f"formula already has MORE negative charge than the SBU "
            f"can accommodate.  This is physically impossible — "
            f"re-check inputs."
        )

    if c.metal in ("Fe", "Co", "Mn"):
        warnings.append(
            f"⚡ CAUTION: {c.metal}-based MOFs can show mass GAIN "
            f"during TGA in air due to metal oxidation (e.g., "
            f"Fe²⁺ → Fe³⁺).  Verify the TGA curve is monotonically "
            f"decreasing."
        )

    if c.metal in ("Zn", "Cd"):
        warnings.append(
            f"⚡ CAUTION: {c.metal} can volatilize at high T under "
            f"inert atmosphere.  Use AIR atmosphere for composition."
        )

    return CompositionResult(
        components=c,
        r_exp=r_exp,
        r_exp_dh=r_exp_dh,
        q_linkers=q,
        p_modulators=p_mod,
        method_used=method,
        M_formula_exp=M_exp,
        M_formula_ideal=M_ideal,
        formula_string=formula,
        charge_balance=cb,
        n_compensator=n_comp,
        ideal_q=ideal_q,
        warnings=warnings,
    )