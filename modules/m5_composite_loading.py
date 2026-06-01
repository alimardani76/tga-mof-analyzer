"""
Module 5 — Composite Loading Analysis
========================================
Answers: "What fraction of my composite is actually MOF?"

Physics
-------
In composites (MOF@polymer, MOF@silica, mixed-matrix membranes,
pellets with binder), TGA residue encodes the MOF loading because
only the MOF contributes a predictable metal oxide residue.

Key equations
-------------
Define residue fractions for pure components:
    r_A = m_residue(pure MOF) / m_initial(pure MOF)
    r_B = m_residue(pure additive) / m_initial(pure additive)
    r_comp = m_residue(composite) / m_initial(composite)

The composite residue is a weighted sum:
    r_comp = w_A × r_A + (1 − w_A) × r_B

Solving for MOF weight fraction:
    w_A = (r_comp − r_B) / (r_A − r_B)

Special case — polymer/organic additive (r_B ≈ 0):
    w_A = r_comp / r_A

This is the simplest and most common case.  The composite residue
comes entirely from the MOF.

References
----------
- Abánades Lázaro, I. Eur. J. Inorg. Chem. 2020, 4284-4294.
  DOI: 10.1002/ejic.202000656  (Section on composites, Figure 4)

Known limitations
-----------------
- If MOF and additive have similar residue fractions (r_A ≈ r_B),
  the denominator approaches zero and TGA cannot determine loading.
  Use ICP or elemental analysis instead.
- If MOF and polymer react at high T (e.g., carbon from polymer
  reduces metal oxide), the linear mixing model fails.
- Inhomogeneous mixing: TGA samples ~5-20 mg.  If the composite
  is not well-mixed, multiple measurements are needed.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class CompositeResult:
    """Result of composite loading analysis.

    Attributes
    ----------
    w_mof_pct : float
        MOF weight fraction in the composite (%).
    w_additive_pct : float
        Additive weight fraction (%).
    r_composite : float
        Residue fraction of the composite (0–1).
    r_mof : float
        Residue fraction of pure MOF (0–1).
    r_additive : float
        Residue fraction of pure additive (0–1).
    case : str
        'no_residue_additive' or 'both_leave_residue'.
    additive_name : str
        Label for the additive.
    warnings : list of str
    """
    w_mof_pct: float
    w_additive_pct: float
    r_composite: float
    r_mof: float
    r_additive: float
    case: str
    additive_name: str = "additive"
    warnings: List[str] = field(default_factory=list)

    @property
    def predicted_residue_pct(self) -> float:
        """Predicted composite residue from the computed loading."""
        w = self.w_mof_pct / 100.0
        return (w * self.r_mof + (1 - w) * self.r_additive) * 100.0

    def summary(self) -> str:
        """Human-readable summary report."""
        lines = [
            "=" * 60,
            "MODULE 5: Composite Loading Analysis",
            "=" * 60,
            f"  MOF loading:         {self.w_mof_pct:.2f} wt%",
            f"  {self.additive_name} content:  "
            f"{self.w_additive_pct:.2f} wt%",
            f"  Analysis case:       {self.case}",
            "",
            "  Residue fractions (mass_residue / mass_initial):",
            f"    Composite:         {self.r_composite:.4f}  "
            f"({self.r_composite * 100:.2f}%)",
            f"    Pure MOF:          {self.r_mof:.4f}  "
            f"({self.r_mof * 100:.2f}%)",
            f"    {self.additive_name}:  "
            f"{self.r_additive:.4f}  "
            f"({self.r_additive * 100:.2f}%)",
            "",
            f"  Predicted composite residue: "
            f"{self.predicted_residue_pct:.2f}%  "
            f"(observed: {self.r_composite * 100:.2f}%)",
        ]

        if self.warnings:
            lines.append("")
            lines.append("  ⚠️  WARNINGS:")
            for w in self.warnings:
                lines.append(f"     - {w}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ------------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------------

def analyze_composite(
    tga_composite,
    tga_pure_mof=None,
    tga_pure_additive=None,
    r_mof: Optional[float] = None,
    r_additive: Optional[float] = None,
    additive_name: str = "additive",
    additive_leaves_residue: bool = False,
) -> CompositeResult:
    """
    Determine MOF loading in a composite from TGA data.

    Parameters
    ----------
    tga_composite : TGAData
        TGA of the composite sample.
    tga_pure_mof : TGAData or None
        TGA of pure MOF.  Used to compute r_mof if not given directly.
    tga_pure_additive : TGAData or None
        TGA of pure additive.  Used to compute r_additive.
    r_mof : float or None
        Residue fraction of pure MOF (0–1).
        Overrides tga_pure_mof if both given.
    r_additive : float or None
        Residue fraction of pure additive (0–1).
        Overrides tga_pure_additive if both given.
        Set to 0.0 explicitly for fully decomposing polymers.
    additive_name : str
        Human-readable label for the additive (e.g., 'PVDF', 'silica').
    additive_leaves_residue : bool
        If True, expect r_additive > 0 (e.g., silica, alumina).
        If False, assume r_additive = 0 unless explicitly provided.

    Returns
    -------
    CompositeResult

    Raises
    ------
    ValueError
        If r_mof cannot be determined.

    Notes
    -----
    IMPORTANT: The pure MOF and composite TGA must be run under the
    SAME conditions (atmosphere, heating rate, final temperature) for
    the residue fractions to be comparable.
    """
    warnings = []

    # --- Composite residue fraction ---
    r_comp = tga_composite.residue_pct / 100.0

    # --- Pure MOF residue fraction ---
    if r_mof is not None:
        pass  # User provided directly
    elif tga_pure_mof is not None:
        r_mof = tga_pure_mof.residue_pct / 100.0
    else:
        raise ValueError(
            "Must provide r_mof or tga_pure_mof.  r_mof is the "
            "residue fraction of the pure MOF (residue_mass / initial_mass)."
        )

    if r_mof <= 0:
        raise ValueError(
            "Pure MOF residue fraction must be > 0.  If the MOF "
            "leaves no residue (e.g., purely organic framework), "
            "this method cannot determine loading."
        )

    # --- Additive residue fraction ---
    if r_additive is not None:
        pass  # User provided directly
    elif tga_pure_additive is not None:
        r_additive = tga_pure_additive.residue_pct / 100.0
    elif additive_leaves_residue:
        raise ValueError(
            f"additive_leaves_residue=True but no r_additive or "
            f"tga_pure_additive provided.  Cannot determine "
            f"{additive_name} residue fraction."
        )
    else:
        r_additive = 0.0
        warnings.append(
            f"{additive_name} residue fraction assumed 0% "
            f"(fully decomposing organic).  If {additive_name} "
            f"leaves char or inorganic residue, provide r_additive "
            f"or tga_pure_additive for accurate results."
        )

    # --- Solve for MOF fraction ---
    if abs(r_additive) < 1e-6:
        # Case 1: additive leaves no residue
        w_mof = r_comp / r_mof
        case = "no_residue_additive"
    else:
        # Case 2: both leave residue
        denom = r_mof - r_additive
        if abs(denom) < 0.01:
            warnings.append(
                f"MOF and {additive_name} have very similar residue "
                f"fractions (r_MOF = {r_mof:.4f}, "
                f"r_{additive_name} = {r_additive:.4f}).  "
                f"The denominator is near zero — TGA cannot reliably "
                f"determine loading.  Use ICP or elemental analysis."
            )
            w_mof = 0.5  # Unreliable placeholder
        else:
            w_mof = (r_comp - r_additive) / denom
        case = "both_leave_residue"

    # --- Validity checks ---
    if w_mof < 0:
        warnings.append(
            f"Computed MOF fraction ({w_mof:.4f}) is NEGATIVE.  "
            f"This means the composite residue is LESS than the "
            f"pure additive residue — check that the pure-component "
            f"TGA data are correct and measured under the same "
            f"conditions."
        )
        w_mof = 0.0
    elif w_mof > 1.0:
        warnings.append(
            f"Computed MOF fraction ({w_mof:.4f}) exceeds 100%.  "
            f"The composite residue is HIGHER than pure MOF — "
            f"possible causes: (a) additive-MOF reaction producing "
            f"extra residue, (b) wrong pure MOF reference, "
            f"(c) inhomogeneous sample."
        )
        w_mof = 1.0

    return CompositeResult(
        w_mof_pct=w_mof * 100.0,
        w_additive_pct=(1.0 - w_mof) * 100.0,
        r_composite=r_comp,
        r_mof=r_mof,
        r_additive=r_additive,
        case=case,
        additive_name=additive_name,
        warnings=warnings,
    )