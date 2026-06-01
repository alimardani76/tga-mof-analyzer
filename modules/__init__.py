"""
TGA-MOF Analyzer v0 — Modules
===============================
Five deterministic modules for extracting quantitative information
from TGA data of Metal-Organic Frameworks.

Module 1: Thermal Stability Profiling
Module 2: Guest / Solvent Content & Activation Quality
Module 3: Framework Composition / Molecular Formula
Module 4: Missing Linker Defect Quantification
Module 5: Composite Loading Analysis
"""

from . import m1_thermal_stability
from . import m2_guest_content
from . import m3_composition
from . import m4_defect_quantification
from . import m5_composite_loading