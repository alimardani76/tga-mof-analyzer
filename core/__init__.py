"""
TGA-MOF-Analyzer v0 — Core Engine
"""

from .tga_parser import TGAData
from .tga_quality import QualityReport, QualityCheck, run_quality_checks
from .dtg import DTGResult, TGAEvent, EventDetectionResult, compute_dtg, detect_events
from .rexp import (
    RexpResult, compute_rexp, compute_linkers, compute_linkers_with_modulator,
    compute_formula_mass, compute_theoretical_rexp,
    compute_rexp_curve, find_dh_plateau,
)
from .guest_solver import GuestCandidate, GuestSolution, enumerate_guest_combinations, COMMON_GUESTS
from .charge_balance import ChargeBalanceResult, check_charge_balance, compute_compensator_needed