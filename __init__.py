"""
TGA-MOF-Analyzer v0
=====================
Deterministic, physics-based TGA analysis toolkit for Metal-Organic Frameworks.

Five modules, every equation traceable to peer-reviewed literature.
Works with TGA data alone.  Optional inputs (NMR, CHNO, ICP) improve accuracy.

Modules
-------
1. Thermal Stability Profiling
   T_onset, T1/T2/T5/T10, stability window, DTG event detection.

2. Guest / Solvent Content & Activation Quality
   Solvent wt%, mol/FU, combinatorial guest solver, activation score.

3. Composition / Molecular Formula Determination
   R_exp methodology, linker count, modulator resolution, charge balance.

4. Missing Linker Defect Quantification
   Multi-compensator engine, M-400 sanity check, missing-cluster caveat.

5. Composite Loading Analysis
   MOF wt% in MOF@polymer, MOF@silica, mixed-matrix membranes.

Key References
--------------
- Abánades Lázaro, I. Eur. J. Inorg. Chem. 2020, 4284-4294.
  DOI: 10.1002/ejic.202000656
- Shearer, G.C. et al. Chem. Mater. 2016, 28, 3749-3761.
  DOI: 10.1021/acs.chemmater.6b00602
- Sannes, D.K. et al. Chem. Mater. 2023, 35, 3793-3800.
  DOI: 10.1021/acs.chemmater.2c03744
- Pulparayil Mathew, J. et al. Adv. Sci. 2025, 12, e04713.
  DOI: 10.1002/advs.202504713

Author
------
TGA-MOF-Analyzer v0 — open-source, literature-grounded.
"""

__version__ = "0.1.0"

# Core engine
from .core.tga_parser import TGAData
from .core.dtg import DTGResult, TGAEvent, EventDetectionResult, compute_dtg, detect_events
from .core.rexp import RexpResult, compute_rexp, compute_linkers, compute_linkers_with_modulator
from .core.guest_solver import GuestCandidate, GuestSolution, enumerate_guest_combinations, COMMON_GUESTS
from .core.charge_balance import ChargeBalanceResult, check_charge_balance, compute_compensator_needed

# Modules
from .modules.m1_thermal_stability import StabilityResult, analyze_stability
from .modules.m2_guest_content import GuestContentResult, analyze_guest_content, analyze_guest_content_mixed
from .modules.m3_composition import (
    MOFComponents, CompositionResult, analyze_composition,
    PRESET_MOFS,
    UIO_66_ZR, UIO_67_ZR, MOF_808_ZR, MOF_5_ZN,
    HKUST_1_CU, MIL_125_TI, ZIF_8_ZN, MIL_53_AL,
)
from .modules.m4_defect_quantification import (
    CompensatorModel, CompensatorResult, DefectResult,
    analyze_defects, DEFAULT_COMPENSATORS,
)
from .modules.m5_composite_loading import CompositeResult, analyze_composite

# Validation
from .validation.cross_check import (
    ValidationResult,
    validate_mass_balance,
    validate_chno,
    validate_icp,
)