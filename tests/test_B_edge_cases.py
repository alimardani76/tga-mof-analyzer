"""
Test Group B: Edge Cases
=========================
Tests inputs that are incomplete or borderline.
The code should NOT crash — it should warn or skip.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.validator import validate_run_config


class TestEdgeCases:
    def test_atmosphere_unknown(self):
        config = {
            "tga_csv": __file__,
            "formula": "Zr6O4(OH)4(BDC)6",
            "atmosphere": "unknown",
        }
        result = validate_run_config(config)
        assert result.is_valid  # should still be valid
        assert len(result.warnings) > 0
        assert any("atmosphere" in w.message.lower() for w in result.warnings)

    def test_heating_rate_missing(self):
        config = {
            "tga_csv": __file__,
            "formula": "Zr6O4(OH)4(BDC)6",
            "atmosphere": "air",
        }
        result = validate_run_config(config)
        assert result.is_valid
        # Should have info about missing heating rate
        assert any("heating" in m.message.lower()
                    for m in result.messages)

    def test_observed_final_mass_missing(self):
        config = {
            "tga_csv": __file__,
            "formula": "Zr6O4(OH)4(BDC)6",
            "atmosphere": "air",
        }
        result = validate_run_config(config)
        assert result.is_valid
        assert any("observed" in m.message.lower() or "residue" in m.message.lower()
                    for m in result.messages)

    def test_no_formula_still_valid(self):
        config = {
            "tga_csv": __file__,
            "atmosphere": "air",
        }
        result = validate_run_config(config)
        assert result.is_valid
        assert any("formula" in m.message.lower() for m in result.messages)

    def test_negative_heating_rate(self):
        config = {
            "tga_csv": __file__,
            "heating_rate": -10,
        }
        result = validate_run_config(config)
        assert not result.is_valid

    def test_very_high_heating_rate_warning(self):
        config = {
            "tga_csv": __file__,
            "heating_rate": 100,
        }
        result = validate_run_config(config)
        assert any("high" in w.message.lower() for w in result.warnings)