"""
Charge Balance Checker
=======================
Validates electrical charge neutrality of a proposed MOF formula.

For any MOF formula, total positive charge (from the metal SBU) must
equal total negative charge (from linkers + modulators + compensating
species like OH⁻ or Cl⁻).

If the residual charge is non-zero beyond tolerance, the formula is
physically impossible.

References
----------
- Abánades Lázaro, I. Eur. J. Inorg. Chem. 2020, 4284-4294.
  DOI: 10.1002/ejic.202000656
- Shearer, G.C. et al. Chem. Mater. 2016, 28, 3749-3761.
  DOI: 10.1021/acs.chemmater.6b00602
"""

from dataclasses import dataclass


@dataclass
class ChargeBalanceResult:
    """Result of a charge balance check.

    Attributes
    ----------
    total_positive : float
        Total positive charge (from the SBU).
    total_negative : float
        Total negative charge magnitude (from linkers, modulators,
        compensating species).
    residual : float
        total_positive − total_negative.  Should be ~0 for valid formula.
    balanced : bool
        True if |residual| < tolerance.
    """
    total_positive: float
    total_negative: float
    residual: float
    balanced: bool

    def __repr__(self) -> str:
        status = "✅ BALANCED" if self.balanced else "❌ IMBALANCED"
        return (
            f"ChargeBalance({status}, "
            f"(+){self.total_positive:.2f}, "
            f"(−){self.total_negative:.2f}, "
            f"residual={self.residual:+.2f})"
        )


def check_charge_balance(
    sbu_charge: float,
    q: float,
    linker_charge: float,
    p: float = 0.0,
    mod_charge: float = 0.0,
    n_compensator: float = 0.0,
    comp_charge: float = 0.0,
    tolerance: float = 0.5,
) -> ChargeBalanceResult:
    """
    Check whether a MOF formula is charge-balanced.

    Equation:
        total_positive = |sbu_charge|
        total_negative = |linker_charge| × q
                       + |mod_charge| × p
                       + |comp_charge| × n_compensator
        residual = total_positive − total_negative

    Parameters
    ----------
    sbu_charge : float
        Net positive charge of the SBU (e.g., +12 for Zr6O4(OH)4).
    q : float
        Number of linkers per FU.
    linker_charge : float
        Charge per linker (e.g., −2 for dicarboxylates).
    p : float
        Number of modulators per FU (default 0).
    mod_charge : float
        Charge per modulator (e.g., −1 for monocarboxylates).
    n_compensator : float
        Number of additional compensating species per FU (default 0).
    comp_charge : float
        Charge per compensating species (e.g., −1 for OH⁻ or Cl⁻).
    tolerance : float
        Maximum allowed |residual| for "balanced" status (default 0.5).

    Returns
    -------
    ChargeBalanceResult
    """
    total_positive = abs(sbu_charge)
    total_negative = (
        abs(linker_charge) * q
        + abs(mod_charge) * p
        + abs(comp_charge) * n_compensator
    )
    residual = total_positive - total_negative
    balanced = abs(residual) < tolerance

    return ChargeBalanceResult(
        total_positive=total_positive,
        total_negative=total_negative,
        residual=residual,
        balanced=balanced,
    )


def compute_compensator_needed(
    sbu_charge: float,
    q: float,
    linker_charge: float,
    p: float = 0.0,
    mod_charge: float = 0.0,
    comp_charge: float = -1.0,
) -> float:
    """
    Compute how many compensating species are needed for charge balance.

    n_comp = (|sbu_charge| − |linker_charge|×q − |mod_charge|×p) / |comp_charge|

    Parameters
    ----------
    sbu_charge : float
        Net positive charge of the SBU.
    q : float
        Linkers per FU.
    linker_charge : float
        Charge per linker.
    p : float
        Modulators per FU.
    mod_charge : float
        Charge per modulator.
    comp_charge : float
        Charge per compensator (default −1).

    Returns
    -------
    float
        Number of compensating species needed.
        Negative value means the formula is already over-compensated.
    """
    charge_deficit = (
        abs(sbu_charge)
        - abs(linker_charge) * q
        - abs(mod_charge) * p
    )
    if abs(comp_charge) < 1e-10:
        raise ValueError("Compensator charge cannot be zero.")
    return charge_deficit / abs(comp_charge)