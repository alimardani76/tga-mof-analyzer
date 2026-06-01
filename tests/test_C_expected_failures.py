"""
Test Group C: Expected Failures
=================================
Tests deliberately invalid inputs.
These SHOULD fail validation cleanly.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.validator import validate_run_config
from core.formula_parser import parse_formula


class TestExpectedFailures:
    def test_reversed_window_rejected(self):
        config = {
            "tga_csv": __file__,
            "window_start": 500,
            "window_end": 100,
        }
        result = validate_run_config(config)
        assert not result.is_valid

    def test_unknown_guest_rejected(self):
        config = {
            "tga_csv": __file__,
            "guests": "H2O,MyFavouriteSolvent",
        }
        result = validate_run_config(config)
        assert not result.is_valid

    def test_invalid_formula_rejected(self):
        config = {
            "tga_csv": __file__,
            "formula": "XxYy123Zz",
        }
        result = validate_run_config(config)
        assert not result.is_valid

    def test_missing_csv_rejected(self):
        config = {
            "tga_csv": "/nonexistent/path/data.csv",
        }
        result = validate_run_config(config)
        assert not result.is_valid

    def test_strongly_negative_residue_rejected(self):
        config = {
            "tga_csv": __file__,
            "observed_final_mass": -50,
        }
        result = validate_run_config(config)
        assert not result.is_valid

    def test_parse_formula_invalid_element(self):
        with pytest.raises(ValueError):
            parse_formula("Qq2O3")

    def test_parse_formula_unmatched_paren(self):
        with pytest.raises(ValueError):
            parse_formula("Zr6O4(OH4")