"""
TGA-MOF Analyzer v0 — Validation
==================================
Cross-validation checks for TGA-derived composition using
independent analytical data (CHNO, ICP, mass balance).
"""

from .cross_check import (
    ValidationResult,
    validate_mass_balance,
    validate_chno,
    validate_icp,
)