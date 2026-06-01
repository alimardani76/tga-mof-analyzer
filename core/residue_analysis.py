"""
Residue Analysis — Observed vs. Predicted
============================================
Compares the experimental TGA residue to the theoretical oxide
residue predicted from the framework formula.

The difference between observed and predicted residue is a
powerful diagnostic:

  Δ = observed_residue - predicted_residue

  Δ ≈ 0:   Framework is fully intact, ideal stoichiometry
  Δ < 0:   Less residue than expected → excess organic content
            (guests, modulators, extra linkers, amorphous phase)
  Δ > 0:   More residue than expected → inorganic impurity
            (unreacted metal salt, extra metal oxide, support)

This analysis is INDEPENDENT of the DH plateau choice —
it uses only the final residue mass and the formula.
It therefore provides a cross-check on Module 3 results.

References
----------
- Shearer et al., Chem. Mater. 2016, 28, 3749–3761
- Abánades Lázaro et al., Eur. J. Inorg. Chem. 2020, 4284–4294
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class ResidueAnalysisResult:
    """Result of observed vs. predicted residue comparison.

    Attributes
    ----------
    observed_residue_pct : float
        Observed final residue from TGA (wt%).
    predicted_residue_pct : float
        Predicted residue from framework formula (wt% of guest-free mass).
    predicted_residue_pct_initial : float or None
        Predicted residue as wt% of initial mass (if guest mass known).
    delta_pct : float
        observed - predicted (percentage points).
    delta_interpretation : str
        Human-readable interpretation of delta.
    oxide_formula : str
        Expected oxide phase(s).
    oxide_mass_gmol : float
        Total oxide residue molar mass.
    framework_mass_gmol : float
        Guest-free framework molar mass.
    formula_used : str
        The formula string used for prediction.
    atmosphere : str
        TGA atmosphere.
    warnings : list of str
    """
    observed_residue_pct: float
    predicted_residue_pct: float
    predicted_residue_pct_initial: Optional[float]
    delta_pct: float
    delta_interpretation: str
    oxide_formula: str
    oxide_mass_gmol: float
    framework_mass_gmol: float
    formula_used: str
    atmosphere: str
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "RESIDUE ANALYSIS: Observed vs. Predicted",
            "=" * 60,
            f"  Formula:     {self.formula_used}",
            f"  M_framework: {self.framework_mass_gmol:.2f} g/mol",
            f"  Atmosphere:  {self.atmosphere}",
            f"",
            f"  Expected oxide:     {self.oxide_formula}",
            f"  Oxide mass:         {self.oxide_mass_gmol:.2f} g/mol",
            f"",
            f"  Predicted residue (guest-free basis): "
            f"{self.predicted_residue_pct:.2f}%",
        ]
        if self.predicted_residue_pct_initial is not None:
            lines.append(
                f"  Predicted residue (initial basis):    "
                f"{self.predicted_residue_pct_initial:.2f}%"
            )
        lines.extend([
            f"  Observed residue:  {self.observed_residue_pct:.2f}%",
            f"",
            f"  Δ = observed − predicted = {self.delta_pct:+.2f} pp",
            f"  Interpretation: {self.delta_interpretation}",
        ])
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  ⚠️ {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


def analyze_residue(
    observed_residue_pct: float,
    framework_formula: str,
    atmosphere: str = "air",
    guest_mass_per_fu: float = 0.0,
    tolerance_pct: float = 5.0,
) -> ResidueAnalysisResult:
    """
    Compare observed TGA residue to predicted oxide residue.

    Parameters
    ----------
    observed_residue_pct : float
        Experimental final residue from TGA (wt%).
    framework_formula : str
        Guest-free framework formula (e.g., "Zr6O4(OH)4(BDC)6").
    atmosphere : str
        TGA atmosphere.
    guest_mass_per_fu : float
        Guest mass per formula unit (g/mol) for initial-basis calc.
    tolerance_pct : float
        Tolerance for "close match" (percentage points).

    Returns
    -------
    ResidueAnalysisResult
    """
    from core.formula_parser import predict_residue

    pred = predict_residue(
        framework_formula,
        atmosphere=atmosphere,
        guest_mass_per_fu=guest_mass_per_fu,
    )

    # Use guest-free basis for comparison
    # (observed residue is wt% of initial mass which includes guests,
    # but predicted_pct_guest_free is wt% of framework-only mass.
    # For a fair comparison when guest mass is known, use initial basis.)
    if pred.residue_pct_initial is not None and guest_mass_per_fu > 0:
        predicted = pred.residue_pct_initial
    else:
        predicted = pred.residue_pct_guest_free

    delta = observed_residue_pct - predicted

    # Interpret delta
    abs_delta = abs(delta)
    if abs_delta < 2.0:
        interpretation = (
            f"Excellent match (Δ = {delta:+.2f} pp). "
            f"Observed residue is consistent with the ideal framework formula."
        )
    elif abs_delta < tolerance_pct:
        if delta < 0:
            interpretation = (
                f"Observed residue is {abs_delta:.1f} pp BELOW predicted. "
                f"Possible causes: (a) missing-linker defects (compensated by "
                f"lighter species like OH/H2O), (b) incomplete combustion, "
                f"(c) amorphous/non-crystalline phase with different composition."
            )
        else:
            interpretation = (
                f"Observed residue is {abs_delta:.1f} pp ABOVE predicted. "
                f"Possible causes: (a) unreacted metal salt or metal oxide "
                f"impurity, (b) inert support/substrate contributing to residue, "
                f"(c) incomplete organic combustion leaving carbonaceous residue."
            )
    else:
        if delta < 0:
            interpretation = (
                f"Large deficit ({abs_delta:.1f} pp below predicted). "
                f"This sample likely contains significant guest/solvent content "
                f"that was not removed before TGA, OR the framework has "
                f"substantial defects (>20% missing linkers). "
                f"Cross-check with N₂ sorption and elemental analysis."
            )
        else:
            interpretation = (
                f"Large excess ({abs_delta:.1f} pp above predicted). "
                f"Significant inorganic impurity or unreacted precursor. "
                f"Cross-check with PXRD and ICP."
            )

    return ResidueAnalysisResult(
        observed_residue_pct=observed_residue_pct,
        predicted_residue_pct=predicted,
        predicted_residue_pct_initial=pred.residue_pct_initial,
        delta_pct=delta,
        delta_interpretation=interpretation,
        oxide_formula=pred.oxide_formula,
        oxide_mass_gmol=pred.oxide_mass,
        framework_mass_gmol=pred.framework_mass,
        formula_used=framework_formula,
        atmosphere=atmosphere,
        warnings=pred.warnings,
    )