"""
Test Group D: Regression
==========================
Verifies that reference values remain stable after code changes.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.formula_parser import parse_formula, predict_residue, formula_mass


REFERENCE = {
    "uio66_nh2_framework_mass": 1754.154,
    "uio66_nh2_predicted_residue_pct": 42.15,
    "h2o_mw": 18.015,
    "dmf_mw": 73.094,
    "etoh_mw": 46.068,
    "nh2bdc_mw": 179.131,
    "zro2_mw": 123.222,
}


class TestRegressionValues:
    def test_framework_mass(self):
        f = parse_formula("Zr6O4(OH)4(NH2BDC)6")
        assert abs(f.molar_mass - REFERENCE["uio66_nh2_framework_mass"]) < 0.01

    def test_predicted_residue(self):
        r = predict_residue("Zr6O4(OH)4(NH2BDC)6", atmosphere="air")
        assert abs(r.residue_pct_guest_free - REFERENCE["uio66_nh2_predicted_residue_pct"]) < 0.1

    def test_h2o_mass(self):
        assert abs(formula_mass("H2O") - REFERENCE["h2o_mw"]) < 0.01

    def test_dmf_mass(self):
        assert abs(formula_mass("DMF") - REFERENCE["dmf_mw"]) < 0.01

    def test_etoh_mass(self):
        assert abs(formula_mass("EtOH") - REFERENCE["etoh_mw"]) < 0.01

    def test_nh2bdc_mass(self):
        assert abs(formula_mass("C8H5NO4") - REFERENCE["nh2bdc_mw"]) < 0.01

    def test_zro2_mass(self):
        assert abs(formula_mass("ZrO2") - REFERENCE["zro2_mw"]) < 0.01

    def test_element_counts_zr(self):
        f = parse_formula("Zr6O4(OH)4(NH2BDC)6")
        assert abs(f.element_counts["Zr"] - 6.0) < 0.01

    def test_element_counts_c(self):
        f = parse_formula("Zr6O4(OH)4(NH2BDC)6")
        assert abs(f.element_counts["C"] - 48.0) < 0.01

    def test_element_counts_n(self):
        f = parse_formula("Zr6O4(OH)4(NH2BDC)6")
        assert abs(f.element_counts["N"] - 6.0) < 0.01