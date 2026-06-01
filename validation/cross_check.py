"""
Cross-Validation Checks
=========================
Independent verification of TGA-derived composition against
supplementary analytical data.

Three checks are provided:

1. Mass Balance
   Sum of all mass-loss events + residue should equal 100%.
   Deviations indicate baseline drift, incomplete data, or
   undetected events.

2. CHNO Elemental Analysis
   Predicted C, H, N content from the TGA-derived formula is
   compared to experimental CHN analyzer results.  Deviations
   indicate unaccounted species (e.g., residual inorganic ions,
   unreacted linker, or wrong guest assignment).

3. ICP Metal Content
   Predicted metal wt% from the formula is compared to ICP-OES/MS.
   Deviations indicate wrong residue assumption, incomplete
   digestion, or hidden inorganic species.

Usage
-----
These checks are OPTIONAL but strongly recommended.  They catch
systematic errors that TGA alone cannot detect.

References
----------
- Pulparayil Mathew, J. et al. Adv. Sci. 2025, 12, e04713.
  DOI: 10.1002/advs.202504713
  (demonstrates that CHNO + ICP + UV-Vis catch Cl⁻/NO₃⁻ that
  TGA misses entirely)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of a single validation check.

    Attributes
    ----------
    check_name : str
        Name of the check (e.g., 'Mass Balance').
    passed : bool
        True if all criteria are within tolerance.
    details : dict
        Numeric details (predicted, experimental, deviation, etc.).
    message : str
        Human-readable summary.
    """
    check_name: str
    passed: bool
    details: Dict[str, float]
    message: str

    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"ValidationResult({self.check_name}: {status})"


# ------------------------------------------------------------------
# Check 1: Mass Balance
# ------------------------------------------------------------------

def validate_mass_balance(
    event_losses: List[float],
    residue_pct: float,
    initial_pct: float = 100.0,
    tolerance: float = 2.0,
) -> ValidationResult:
    """
    Check that sum of all event mass losses + residue = initial mass.

    This catches:
    - Baseline drift in the TGA instrument
    - Undetected mass-loss events (hidden between assigned events)
    - Incomplete TGA data (didn't reach final temperature)

    Parameters
    ----------
    event_losses : list of float
        Mass loss per detected event in wt% (from Module 1).
    residue_pct : float
        Final residue mass in wt%.
    initial_pct : float
        Initial mass (should be ~100%).
    tolerance : float
        Acceptable deviation in wt% (default 2.0).

    Returns
    -------
    ValidationResult
    """
    total_loss = sum(event_losses)
    reconstructed = total_loss + residue_pct
    deviation = abs(reconstructed - initial_pct)
    unaccounted = initial_pct - reconstructed

    passed = deviation <= tolerance

    details = {
        "total_event_loss_pct": total_loss,
        "residue_pct": residue_pct,
        "reconstructed_total_pct": reconstructed,
        "deviation_pct": deviation,
        "unaccounted_pct": unaccounted,
    }

    if passed:
        msg = (
            f"Mass balance OK: events ({total_loss:.2f}%) + "
            f"residue ({residue_pct:.2f}%) = {reconstructed:.2f}% "
            f"(deviation: {deviation:.2f}%, within ±{tolerance}%)"
        )
    else:
        msg = (
            f"Mass balance FAILED: events ({total_loss:.2f}%) + "
            f"residue ({residue_pct:.2f}%) = {reconstructed:.2f}% "
            f"(deviation: {deviation:.2f}%, exceeds ±{tolerance}%).  "
            f"Unaccounted mass: {unaccounted:.2f}%.  "
            f"Possible causes: baseline drift, undetected event, "
            f"or incomplete TGA run."
        )

    return ValidationResult(
        check_name="Mass Balance",
        passed=passed,
        details=details,
        message=msg,
    )


# ------------------------------------------------------------------
# Check 2: CHNO Elemental Analysis
# ------------------------------------------------------------------

# Standard atomic masses
_ATOMIC_MASS = {
    "C": 12.011,
    "H": 1.008,
    "N": 14.007,
    "O": 15.999,
    "S": 32.065,
}


def validate_chno(
    formula_mass: float,
    n_metals: int,
    M_metal: float,
    q_linkers: float,
    linker_atoms: Dict[str, int],
    p_modulators: float = 0.0,
    modulator_atoms: Optional[Dict[str, int]] = None,
    n_compensator_oh: float = 0.0,
    n_water: float = 0.0,
    C_exp: Optional[float] = None,
    H_exp: Optional[float] = None,
    N_exp: Optional[float] = None,
    O_exp: Optional[float] = None,
    tolerance_pct: float = 2.0,
) -> ValidationResult:
    """
    Compare predicted vs experimental CHNO elemental composition.

    This catches:
    - Wrong linker count (TGA overestimates organic content)
    - Unaccounted species (Cl⁻, NO₃⁻, extra solvent)
    - Wrong modulator assignment

    Parameters
    ----------
    formula_mass : float
        Total formula mass of the MOF including node, linkers,
        modulators, compensators, and water (g/mol).
    n_metals : int
        Number of metal atoms per FU.
    M_metal : float
        Atomic mass of the metal (g/mol).
    q_linkers : float
        Linkers per FU.
    linker_atoms : dict
        Atoms per linker molecule, e.g., {"C": 8, "H": 4, "O": 4}
        for BDC²⁻.
    p_modulators : float
        Modulators per FU (default 0).
    modulator_atoms : dict or None
        Atoms per modulator, e.g., {"C": 1, "H": 1, "O": 2} for
        formate.
    n_compensator_oh : float
        OH⁻ groups per FU (default 0).
    n_water : float
        Adsorbed water molecules per FU (default 0).
    C_exp, H_exp, N_exp, O_exp : float or None
        Experimental elemental wt% from CHN(O) analyzer.
        Only elements with experimental data are compared.
    tolerance_pct : float
        Acceptable deviation per element in wt% (default 2.0).

    Returns
    -------
    ValidationResult
    """
    # --- Compute predicted atom counts per FU ---
    atom_counts = {"C": 0.0, "H": 0.0, "N": 0.0, "O": 0.0, "S": 0.0}

    # From linkers
    for elem, count in linker_atoms.items():
        if elem in atom_counts:
            atom_counts[elem] += q_linkers * count

    # From modulators
    if modulator_atoms is not None and p_modulators > 0:
        for elem, count in modulator_atoms.items():
            if elem in atom_counts:
                atom_counts[elem] += p_modulators * count

    # From OH⁻ compensators
    if n_compensator_oh > 0:
        atom_counts["O"] += n_compensator_oh * 1
        atom_counts["H"] += n_compensator_oh * 1

    # From adsorbed water
    if n_water > 0:
        atom_counts["O"] += n_water * 1
        atom_counts["H"] += n_water * 2

    # --- Compute predicted wt% ---
    predictions = {}
    for elem in ["C", "H", "N", "O"]:
        mass_elem = atom_counts[elem] * _ATOMIC_MASS[elem]
        predictions[elem] = (mass_elem / formula_mass) * 100.0 if formula_mass > 0 else 0.0

    # --- Compare with experimental ---
    experimental = {"C": C_exp, "H": H_exp, "N": N_exp, "O": O_exp}

    details = {}
    comparisons = []
    all_pass = True

    for elem in ["C", "H", "N", "O"]:
        details[f"{elem}_predicted_pct"] = predictions[elem]

        exp_val = experimental[elem]
        if exp_val is not None:
            details[f"{elem}_experimental_pct"] = exp_val
            dev = abs(predictions[elem] - exp_val)
            details[f"{elem}_deviation_pct"] = dev
            elem_pass = dev <= tolerance_pct
            if not elem_pass:
                all_pass = False
            comparisons.append(
                f"{elem}: pred={predictions[elem]:.2f}%, "
                f"exp={exp_val:.2f}%, "
                f"Δ={dev:.2f}% {'✅' if elem_pass else '❌'}"
            )

    if not comparisons:
        msg = "No experimental CHNO data provided for comparison."
        all_pass = True  # Nothing to fail
    else:
        msg = " | ".join(comparisons)

    return ValidationResult(
        check_name="CHNO Elemental Analysis",
        passed=all_pass,
        details=details,
        message=msg,
    )


# ------------------------------------------------------------------
# Check 3: ICP Metal Content
# ------------------------------------------------------------------

def validate_icp(
    formula_mass: float,
    n_metals: int,
    M_metal: float,
    metal_wt_pct_exp: float,
    metal_symbol: str = "M",
    tolerance_pct: float = 2.0,
) -> ValidationResult:
    """
    Compare predicted vs experimental metal content from ICP-OES/MS.

    This catches:
    - Wrong formula mass (e.g., hidden Cl⁻/NO₃⁻ adding mass)
    - Wrong number of metals per FU
    - Incomplete digestion in ICP sample prep

    Parameters
    ----------
    formula_mass : float
        Total formula mass of the MOF (g/mol).
        Should include node + linkers + modulators + compensators
        + adsorbed water — the full experimental formula.
    n_metals : int
        Number of metal atoms per FU.
    M_metal : float
        Atomic mass of the metal (g/mol).
    metal_wt_pct_exp : float
        Experimental metal wt% from ICP-OES or ICP-MS.
    metal_symbol : str
        Element symbol for reporting (e.g., 'Zr').
    tolerance_pct : float
        Acceptable deviation in wt% (default 2.0).

    Returns
    -------
    ValidationResult
    """
    if formula_mass <= 0:
        return ValidationResult(
            check_name="ICP Metal Content",
            passed=False,
            details={},
            message="Formula mass must be > 0 for ICP validation.",
        )

    predicted_pct = (n_metals * M_metal / formula_mass) * 100.0
    deviation = abs(predicted_pct - metal_wt_pct_exp)
    passed = deviation <= tolerance_pct

    details = {
        f"{metal_symbol}_predicted_pct": predicted_pct,
        f"{metal_symbol}_experimental_pct": metal_wt_pct_exp,
        f"{metal_symbol}_deviation_pct": deviation,
        "formula_mass_used": formula_mass,
    }

    status = "✅" if passed else "❌"
    msg = (
        f"{metal_symbol}: pred={predicted_pct:.2f}%, "
        f"exp={metal_wt_pct_exp:.2f}%, "
        f"Δ={deviation:.2f}% {status}"
    )

    if not passed:
        msg += (
            f"  (Deviation exceeds ±{tolerance_pct}%.  "
            f"If predicted > experimental: formula mass may be too high "
            f"(hidden inorganic species like Cl⁻?).  "
            f"If predicted < experimental: formula mass may be too low "
            f"(missing guest/modulator in formula?).)"
        )

    return ValidationResult(
        check_name="ICP Metal Content",
        passed=passed,
        details=details,
        message=msg,
    )