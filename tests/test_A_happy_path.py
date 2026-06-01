"""
Test Group A: Happy Path
=========================
Verifies that normal inputs produce expected results.
These tests should ALWAYS pass. If they don't, the core
engine is broken.

Run with: python -m pytest tests/ -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.formula_parser import parse_formula, predict_residue, formula_mass
from core.tga_quality import run_quality_checks
from core.uncertainty import (
    guest_count_uncertainty,
    composition_uncertainty,
    composite_loading_uncertainty,
)
from core.validator import validate_run_config


# ------------------------------------------------------------------
# Formula parser
# ------------------------------------------------------------------

class TestFormulaParser:
    def test_water(self):
        assert abs(formula_mass("H2O") - 18.015) < 0.01

    def test_zro2(self):
        assert abs(formula_mass("ZrO2") - 123.222) < 0.01

    def test_nh2bdc(self):
        assert abs(formula_mass("C8H5NO4") - 179.131) < 0.01

    def test_uio66_nh2_full(self):
        f = parse_formula("Zr6O4(OH)4(NH2BDC)6")
        assert abs(f.molar_mass - 1754.154) < 0.01

    def test_alias_expansion(self):
        f = parse_formula("Zr6O4(OH)4(BDC)6")
        assert "Zr" in f.element_counts
        assert f.element_counts["Zr"] == 6.0
        assert len(f.aliases_expanded) > 0

    def test_fractional_subscript(self):
        f = parse_formula("Zr6O6(NH2BDC)5.54")
        assert abs(f.element_counts["Zr"] - 6.0) < 0.01

    def test_simple_elements(self):
        f = parse_formula("NaCl")
        assert abs(f.element_counts["Na"] - 1.0) < 0.01
        assert abs(f.element_counts["Cl"] - 1.0) < 0.01

    def test_nested_parens(self):
        f = parse_formula("Ca(NO3)2")
        assert abs(f.element_counts["Ca"] - 1.0) < 0.01
        assert abs(f.element_counts["N"] - 2.0) < 0.01
        assert abs(f.element_counts["O"] - 6.0) < 0.01

    def test_invalid_element_raises(self):
        with pytest.raises(ValueError):
            parse_formula("Xx2O3")


# ------------------------------------------------------------------
# Residue prediction
# ------------------------------------------------------------------

class TestResiduePrediction:
    def test_uio66_nh2_residue(self):
        r = predict_residue("Zr6O4(OH)4(NH2BDC)6", atmosphere="air")
        assert abs(r.residue_pct_guest_free - 42.15) < 0.1

    def test_oxide_formula(self):
        r = predict_residue("Zr6O4(OH)4(BDC)6", atmosphere="air")
        assert "ZrO2" in r.oxide_formula

    def test_inert_atmosphere_warning(self):
        r = predict_residue("Zr6O4(OH)4(BDC)6", atmosphere="N2")
        assert len(r.warnings) > 0
        assert any("inert" in w.lower() or "non-oxidative" in w.lower()
                    for w in r.warnings)

    def test_no_metals_raises(self):
        with pytest.raises(ValueError):
            predict_residue("C6H12O6")


# ------------------------------------------------------------------
# Uncertainty propagation
# ------------------------------------------------------------------

class TestUncertainty:
    def test_guest_count_positive_sigma(self):
        u = guest_count_uncertainty(
            mass_loss_pct=10.0,
            framework_mass=1754.0,
            guest_mass=18.015,
        )
        assert u.value > 0
        assert u.sigma > 0
        assert u.sigma < u.value  # sigma should be much smaller than value

    def test_composition_positive_sigma(self):
        u = composition_uncertainty(
            mass_at_plateau_pct=75.0,
            residue_pct=33.0,
            M_residue=739.31,
            M_node=643.34,
            M_linker=179.13,
        )
        assert u.sigma > 0

    def test_composite_positive_sigma(self):
        u = composite_loading_uncertainty(
            r_composite=0.22,
            r_mof=0.33,
            r_additive=0.09,
        )
        assert 0 < u.value < 1
        assert u.sigma > 0

    def test_zero_denominator_composite(self):
        u = composite_loading_uncertainty(
            r_composite=0.22,
            r_mof=0.09,
            r_additive=0.09,
        )
        assert u.sigma == float("inf")


# ------------------------------------------------------------------
# Input validation
# ------------------------------------------------------------------

class TestValidation:
    def test_valid_config(self):
        config = {
            "tga_csv": __file__,  # use this file as a dummy path
            "formula": "Zr6O4(OH)4(BDC)6",
            "atmosphere": "air",
            "heating_rate": 10,
        }
        result = validate_run_config(config)
        assert result.is_valid

    def test_missing_csv_path(self):
        config = {}
        result = validate_run_config(config)
        assert not result.is_valid

    def test_reversed_window(self):
        config = {
            "tga_csv": __file__,
            "window_start": 400,
            "window_end": 100,
        }
        result = validate_run_config(config)
        assert not result.is_valid
        assert any("below" in e.message for e in result.errors)

    def test_invalid_formula(self):
        config = {
            "tga_csv": __file__,
            "formula": "NotAFormula123",
        }
        result = validate_run_config(config)
        assert not result.is_valid

    def test_window_crosses_decomposition(self):
        config = {
            "tga_csv": __file__,
            "window_start": 30,
            "window_end": 400,
            "decomp_start": 342,
        }
        result = validate_run_config(config)
        assert any("decomposition" in w.message.lower()
                    for w in result.warnings)