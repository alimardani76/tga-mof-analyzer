"""
Combinatorial Guest Solver
===========================
Given a known framework mass and an observed mass-loss fraction, enumerate
all stoichiometric combinations of candidate guest molecules that reproduce
the observed loss within tolerance.

The key equation
----------------
    f_calc = sum(n_i × MW_i) / (M_F + sum(n_i × MW_i))

Note: the denominator is the TOTAL initial mass (framework + guests),
not just the framework mass.  TGA reports loss as a fraction of the total
initial sample mass.  Getting this wrong introduces systematic overestimation
that grows with guest loading.

Ranking
-------
A penalty function encodes chemical plausibility:
  Term 1: |mass error|  — must be small
  Term 2: pore volume violation — penalizes overstuffing
  Term 3: boiling point violation — penalizes assigning a high-bp
          solvent to a low-temperature mass-loss event
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from itertools import product


@dataclass
class GuestCandidate:
    """A candidate guest molecule.

    Attributes
    ----------
    name : str
        Identifier (e.g., 'DMF', 'H2O', 'EtOH').
    mw : float
        Molar mass in g/mol.
    boiling_point : float or None
        Normal boiling point in °C.  Used for penalty scoring.
    molecular_volume : float or None
        Approximate molar volume in cm³/mol.  Used for pore-volume check.
    """
    name: str
    mw: float
    boiling_point: Optional[float] = None
    molecular_volume: Optional[float] = None


@dataclass
class GuestSolution:
    """One candidate guest-mixture solution.

    Attributes
    ----------
    counts : dict
        Mapping of guest name → stoichiometric count per FU.
    calculated_loss_pct : float
        Calculated mass loss in wt%.
    mass_error : float
        |calculated − observed| in wt%.
    penalty_score : float
        Combined penalty (lower is better).
    """
    counts: Dict[str, float]
    calculated_loss_pct: float
    mass_error: float
    penalty_score: float

    def formula_string(self) -> str:
        """Human-readable guest formula, e.g. '2.0 DMF · 1.5 H2O'."""
        parts = []
        for name, n in self.counts.items():
            if n > 0:
                parts.append(f"{n:.1f} {name}")
        return " · ".join(parts) if parts else "(empty)"


# ------------------------------------------------------------------
# Common guest library
# ------------------------------------------------------------------

COMMON_GUESTS = [
    GuestCandidate("H2O",    18.015, boiling_point=100.0,  molecular_volume=18.0),
    GuestCandidate("DMF",    73.094, boiling_point=153.0,  molecular_volume=77.4),
    GuestCandidate("MeOH",   32.042, boiling_point=64.7,   molecular_volume=40.7),
    GuestCandidate("EtOH",   46.068, boiling_point=78.4,   molecular_volume=58.7),
    GuestCandidate("DEF",   101.147, boiling_point=177.0,  molecular_volume=130.0),
    GuestCandidate("acetone", 58.079, boiling_point=56.1,  molecular_volume=74.0),
    GuestCandidate("THF",    72.106, boiling_point=66.0,   molecular_volume=81.7),
    GuestCandidate("formic_acid", 46.025, boiling_point=100.8, molecular_volume=37.7),
    GuestCandidate("acetic_acid", 60.052, boiling_point=117.9, molecular_volume=57.5),
]


# ------------------------------------------------------------------
# Guest library with boiling points
# ------------------------------------------------------------------

GUEST_LIBRARY: Dict[str, Dict] = {
    "H2O":   {"formula": "H2O",     "mw": 18.015,  "bp_C": 100.0, "name": "Water"},
    "DMF":   {"formula": "C3H7NO",  "mw": 73.094,  "bp_C": 153.0, "name": "N,N-Dimethylformamide"},
    "DMA":   {"formula": "C4H9NO",  "mw": 87.120,  "bp_C": 165.0, "name": "N,N-Dimethylacetamide"},
    "DEF":   {"formula": "C5H11NO", "mw": 101.147, "bp_C": 176.0, "name": "N,N-Diethylformamide"},
    "DMSO":  {"formula": "C2H6OS",  "mw": 78.133,  "bp_C": 189.0, "name": "Dimethyl sulfoxide"},
    "MeOH":  {"formula": "CH4O",    "mw": 32.042,  "bp_C":  64.7, "name": "Methanol"},
    "EtOH":  {"formula": "C2H6O",   "mw": 46.068,  "bp_C":  78.4, "name": "Ethanol"},
    "THF":   {"formula": "C4H8O",   "mw": 72.106,  "bp_C":  66.0, "name": "Tetrahydrofuran"},
    "DCM":   {"formula": "CH2Cl2",  "mw": 84.933,  "bp_C":  39.6, "name": "Dichloromethane"},
    "CHCl3": {"formula": "CHCl3",   "mw": 119.378, "bp_C":  61.2, "name": "Chloroform"},
    "NMP":   {"formula": "C5H9NO",  "mw": 99.131,  "bp_C": 202.0, "name": "N-Methyl-2-pyrrolidone"},
    "ACN":   {"formula": "C2H3N",   "mw": 41.053,  "bp_C":  82.0, "name": "Acetonitrile"},
    "acetone":{"formula":"C3H6O",   "mw": 58.079,  "bp_C":  56.0, "name": "Acetone"},
    "toluene":{"formula":"C7H8",    "mw": 92.138,  "bp_C": 110.6, "name": "Toluene"},
    "hexane": {"formula":"C6H14",   "mw": 86.175,  "bp_C":  69.0, "name": "Hexane"},
}



# ------------------------------------------------------------------
# Solver
# ------------------------------------------------------------------

def enumerate_guest_combinations(
    M_framework: float,
    loss_fraction: float,
    candidates: Optional[List[GuestCandidate]] = None,
    step: float = 0.5,
    max_count: float = 10.0,
    tolerance: float = 0.005,
    event_temp: Optional[float] = None,
    pore_volume_cm3_per_mol: Optional[float] = None,
    max_results: int = 20,
    w_mass: float = 1.0,
    w_pore: float = 0.3,
    w_bp: float = 0.2,
) -> List[GuestSolution]:
    """
    Enumerate guest stoichiometries matching an observed mass loss.

    Parameters
    ----------
    M_framework : float
        Molar mass of the guest-free MOF formula unit (g/mol).
    loss_fraction : float
        Observed fractional mass loss (0–1), e.g. 0.12 for 12%.
    candidates : list of GuestCandidate or None
        Guest molecules to try.  If None, uses COMMON_GUESTS.
    step : float
        Step size for stoichiometric coefficients (default 0.5).
    max_count : float
        Maximum count per guest species (default 10).
    tolerance : float
        Acceptance tolerance on fractional mass loss (default 0.005 = ±0.5%).
    event_temp : float or None
        Temperature of the mass-loss event (°C).  Used for bp penalty.
    pore_volume_cm3_per_mol : float or None
        Pore volume per FU in cm³/mol.  Used for volume penalty.
    max_results : int
        Maximum number of solutions to return (default 20).
    w_mass, w_pore, w_bp : float
        Penalty weights.

    Returns
    -------
    list of GuestSolution, sorted by penalty_score ascending.
    """
    if candidates is None:
        candidates = COMMON_GUESTS

    if not candidates:
        return []

    if loss_fraction <= 0 or loss_fraction >= 1:
        return []

    n_guests = len(candidates)
    counts_range = np.arange(0, max_count + step / 2, step)

    # Limit combinatorial explosion: max 4 candidates at a time
    if n_guests > 4:
        target_mass = loss_fraction * M_framework / (1 - loss_fraction)
        candidates_sorted = sorted(
            candidates,
            key=lambda g: abs(g.mw - target_mass / 3),
        )
        candidates = candidates_sorted[:4]
        n_guests = 4

    solutions = []
    mws = [g.mw for g in candidates]

    for combo in product(counts_range, repeat=n_guests):
        guest_mass = sum(n * mw for n, mw in zip(combo, mws))
        if guest_mass <= 0:
            continue

        total_mass = M_framework + guest_mass
        f_calc = guest_mass / total_mass
        error = abs(f_calc - loss_fraction)

        if error > tolerance:
            continue

        # --- Penalty scoring ---
        penalty = w_mass * error

        # Pore volume penalty
        if pore_volume_cm3_per_mol is not None:
            vol_used = sum(
                n * (g.molecular_volume or 0)
                for n, g in zip(combo, candidates)
            )
            if vol_used > pore_volume_cm3_per_mol:
                excess = (vol_used - pore_volume_cm3_per_mol) / pore_volume_cm3_per_mol
                penalty += w_pore * excess

        # Boiling point penalty
        if event_temp is not None:
            for n, g in zip(combo, candidates):
                if n > 0 and g.boiling_point is not None:
                    if event_temp < g.boiling_point - 50:
                        bp_penalty = n * (g.boiling_point - event_temp) / 200.0
                        penalty += w_bp * bp_penalty

        counts_dict = {
            g.name: float(n)
            for g, n in zip(candidates, combo)
            if n > 0
        }
        solutions.append(GuestSolution(
            counts=counts_dict,
            calculated_loss_pct=f_calc * 100,
            mass_error=error * 100,
            penalty_score=penalty,
        ))

    solutions.sort(key=lambda s: s.penalty_score)
    return solutions[:max_results]



# ------------------------------------------------------------------
# Plausibility scoring (Stage 6)
# ------------------------------------------------------------------

def score_guest_assignment(
    assignment: Dict[str, float],
    calculated_loss_pct: float,
    observed_loss_pct: float,
    window_end_C: Optional[float] = None,
    guest_library: Optional[Dict] = None,
) -> Tuple[float, List[str]]:
    """
    Score a guest assignment by chemical plausibility.

    Lower score = better assignment.

    Scoring components:
      1. Mass error:     10 × |calc - obs|
      2. Occam penalty:  0.03 × Σ(n_i)  (prefer fewer total molecules)
      3. Species penalty: 2.0 per species beyond 3
      4. BP penalty:     3.0 per guest whose bp > window_end
      5. High-count:     1.0 per guest with n > 8

    Parameters
    ----------
    assignment : dict
        {guest_name: stoichiometric_count}
    calculated_loss_pct : float
        Theoretical mass loss from this assignment (wt%).
    observed_loss_pct : float
        Measured mass loss (wt%).
    window_end_C : float or None
        End temperature of the mass-loss window (°C).
    guest_library : dict or None
        Guest info dict. If None, uses GUEST_LIBRARY.

    Returns
    -------
    (score, penalties)
        score: float (lower is better)
        penalties: list of str describing each penalty applied
    """
    if guest_library is None:
        guest_library = GUEST_LIBRARY

    penalties = []

    # 1. Mass error
    error = abs(calculated_loss_pct - observed_loss_pct)
    score = 10.0 * error
    if error > 0.01:
        penalties.append(f"mass_error: {error:.3f}% (score +{10*error:.2f})")

    # 2. Total molecule count (Occam)
    total_n = sum(assignment.values())
    occam = 0.03 * total_n
    score += occam
    if total_n > 4:
        penalties.append(f"high_total_count: Σn={total_n:.1f} (score +{occam:.2f})")

    # 3. Species count
    n_species = sum(1 for v in assignment.values() if v > 0)
    if n_species > 3:
        extra = (n_species - 3) * 2.0
        score += extra
        penalties.append(f"many_species: {n_species} types (score +{extra:.1f})")

    # 4. Boiling point check
    if window_end_C is not None:
        for guest, count in assignment.items():
            if count <= 0:
                continue
            info = guest_library.get(guest)
            if info and info.get("bp_C") is not None:
                if info["bp_C"] > window_end_C + 20:
                    bp_penalty = 3.0
                    score += bp_penalty
                    penalties.append(
                        f"bp_mismatch: {guest} (bp={info['bp_C']:.0f}°C) "
                        f"assigned in window ending at {window_end_C:.0f}°C "
                        f"(score +{bp_penalty:.1f})"
                    )

    # 5. High individual count
    for guest, count in assignment.items():
        if count > 8:
            high_pen = 1.0
            score += high_pen
            penalties.append(
                f"high_count: {guest} n={count:.1f} (score +{high_pen:.1f})"
            )

    return score, penalties