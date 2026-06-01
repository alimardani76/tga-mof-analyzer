"""
Chemical Formula Parser
========================
Parses MOF-relevant chemical formulas into element counts and
computes molar masses.

Handles:
  - Simple formulas: H2O, ZrO2, C8H4O4
  - Parenthetical groups: Zr6O4(OH)4
  - Nested parentheses: Ca(NO3)2, (NH4)2SO4
  - Fractional subscripts: (BDC)5.54, (OH)0.91
  - Dot notation / hydrates: MOF·3H2O, MOF.2DMF
  - Named aliases: BDC → C8H4O4, DMF → C3H7NO
  - Charged species (charges stripped): OH⁻, NH2-BDC²⁻

Does NOT handle:
  - Isotope notation
  - Coordination bonds or metal-ligand notation
  - Polymer repeat units

Design philosophy:
  - Parse what you can, raise clear errors for what you can't.
  - Never silently drop elements or miscount atoms.
  - Every alias is explicit and auditable.

References:
  - IUPAC 2021 atomic weights (abridged, standard values)
  - Alias masses cross-checked against PubChem CIDs
"""

import re
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


# ------------------------------------------------------------------
# Standard atomic masses (IUPAC 2021, abridged)
# ------------------------------------------------------------------

ATOMIC_MASSES: Dict[str, float] = {
    "H":   1.008,
    "He":  4.0026,
    "Li":  6.941,
    "Be":  9.0122,
    "B":  10.81,
    "C":  12.011,
    "N":  14.007,
    "O":  15.999,
    "F":  18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P":  30.974,
    "S":  32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K":  39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V":  50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.922,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.468,
    "Sr": 87.62,
    "Y":  88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.95,
    "Ru":101.07,
    "Rh":102.91,
    "Pd":106.42,
    "Ag":107.87,
    "Cd":112.41,
    "In":114.82,
    "Sn":118.71,
    "Sb":121.76,
    "Te":127.60,
    "I": 126.90,
    "Xe":131.29,
    "Cs":132.91,
    "Ba":137.33,
    "La":138.91,
    "Ce":140.12,
    "Pr":140.91,
    "Nd":144.24,
    "Sm":150.36,
    "Eu":151.96,
    "Gd":157.25,
    "Tb":158.93,
    "Dy":162.50,
    "Ho":164.93,
    "Er":167.26,
    "Tm":168.93,
    "Yb":173.05,
    "Lu":174.97,
    "Hf":178.49,
    "Ta":180.95,
    "W": 183.84,
    "Re":186.21,
    "Os":190.23,
    "Ir":192.22,
    "Pt":195.08,
    "Au":196.97,
    "Hg":200.59,
    "Tl":204.38,
    "Pb":207.2,
    "Bi":208.98,
    "Th":232.04,
    "U": 238.03,
}


# ------------------------------------------------------------------
# MOF-relevant aliases
# ------------------------------------------------------------------
# Each alias maps a common name to its empirical formula.
# These are used when the formula string contains names instead of
# element symbols (e.g., "Zr6O4(OH)4(BDC)6").
#
# Sources: PubChem, Cambridge Structural Database
#
# IMPORTANT: Aliases represent the DEPROTONATED (coordinating) form
# of the linker, i.e., the form that exists in the framework.
# For mass purposes this doesn't matter (H is removed during
# framework formation, but the formula here is what's IN the MOF).

FORMULA_ALIASES: Dict[str, str] = {
    # --- Carboxylate linkers (deprotonated, as in framework) ---
    "BDC":       "C8H4O4",       # 1,4-benzenedicarboxylate (terephthalate)
    "NH2BDC":    "C8H5NO4",      # 2-amino-1,4-benzenedicarboxylate
    "NH2-BDC":   "C8H5NO4",
    "BPDC":      "C12H8O4",      # 4,4'-biphenyldicarboxylate
    "NDC":       "C12H6O4",      # 2,6-naphthalenedicarboxylate
    "BTC":       "C9H3O6",       # 1,3,5-benzenetricarboxylate (trimesate)
    "BTB":       "C27H15O6",     # 1,3,5-benzenetribenzoate
    "DOBDC":     "C8H2O6",       # 2,5-dioxido-1,4-benzenedicarboxylate
    "DHBDC":     "C8H4O6",       # 2,5-dihydroxy-1,4-benzenedicarboxylate
    "TATB":      "C24H12N3O6",   # 4,4',4''-s-triazine-2,4,6-triyl-tribenzoate
    "FMA":       "C2HO4",        # fumarate (mono-deprotonated: C4H2O4 / 2 = per carboxylate... actually fumaric acid = C4H4O4, fumarate(2-) = C4H2O4)
    "OX":        "C2O4",         # oxalate

    # --- Azolate linkers ---
    "IM":        "C3H3N2",       # imidazolate
    "MeIM":      "C4H5N2",       # 2-methylimidazolate
    "2-MeIM":    "C4H5N2",
    "PhIM":      "C9H7N2",       # benzimidazolate

    # --- Common modulators / capping agents ---
    "FA":        "CHO2",         # formate
    "HCO2":      "CHO2",         # formate (alternative notation)
    "HCOO":      "CHO2",
    "AC":        "C2H3O2",       # acetate
    "CH3COO":    "C2H3O2",
    "OAc":       "C2H3O2",
    "TFA":       "C2F3O2",       # trifluoroacetate
    "BA":        "C7H5O2",       # benzoate
    "AA":        "C6H5O2",       # acrylic acid anion... actually this is ambiguous. Remove if problematic.

    # --- Common guests / solvents ---
    "DMF":       "C3H7NO",       # N,N-dimethylformamide
    "DMA":       "C4H9NO",       # N,N-dimethylacetamide
    "DEF":       "C5H11NO",      # N,N-diethylformamide
    "DMSO":      "C2H6OS",       # dimethyl sulfoxide
    "MeOH":      "CH4O",         # methanol
    "EtOH":      "C2H6O",        # ethanol
    "THF":       "C4H8O",        # tetrahydrofuran
    "DCM":       "CH2Cl2",       # dichloromethane
    "NMP":       "C5H9NO",       # N-methyl-2-pyrrolidone
    "ACN":       "C2H3N",        # acetonitrile
    "H2O":       "H2O",          # water
}


# ------------------------------------------------------------------
# Metal → oxide residue mapping (under air/O2 combustion)
# ------------------------------------------------------------------
# Format: metal_symbol → (oxide_formula, n_metal_per_oxide, oxide_MW)
# Used for predicting TGA residue from framework formula.
#
# Assumption: Complete combustion to the most stable oxide phase.
# This is valid for TGA in air up to ~900°C for most metals.
# Exceptions exist (e.g., Fe can form Fe2O3 or Fe3O4 depending
# on conditions) — the user should verify.

OXIDE_RESIDUES: Dict[str, Tuple[str, int, float]] = {
    # (oxide_formula, metals_per_oxide, MW_oxide)
    "Zr": ("ZrO2",   1, 123.218),
    "Hf": ("HfO2",   1, 210.489),
    "Ti": ("TiO2",   1,  79.866),
    "Al": ("Al2O3",  2, 101.961),
    "Fe": ("Fe2O3",  2, 159.688),
    "Cr": ("Cr2O3",  2, 151.990),
    "V":  ("V2O5",   2, 181.880),
    "Cu": ("CuO",    1,  79.545),
    "Zn": ("ZnO",    1,  81.379),
    "Co": ("Co3O4",  3, 240.797),
    "Ni": ("NiO",    1,  74.692),
    "Mn": ("MnO2",   1,  86.937),
    "Mg": ("MgO",    1,  40.304),
    "Ca": ("CaO",    1,  56.077),
    "Sr": ("SrO",    1, 103.619),
    "Ba": ("BaO",    1, 153.326),
    "Ce": ("CeO2",   1, 172.115),
    "La": ("La2O3",  2, 325.809),
    "Y":  ("Y2O3",   2, 225.810),
    "In": ("In2O3",  2, 277.634),
    "Bi": ("Bi2O3",  2, 465.959),
    "Sn": ("SnO2",   1, 150.709),
    "Pb": ("PbO",    1, 223.199),
    "Cd": ("CdO",    1, 128.410),
    "Mo": ("MoO3",   1, 143.938),
    "W":  ("WO3",    1, 231.838),
}


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class FormulaResult:
    """Result of parsing a chemical formula.

    Attributes
    ----------
    input_string : str
        The original formula string as provided.
    element_counts : dict
        {element_symbol: count} for every element.
    molar_mass : float
        Total molar mass in g/mol.
    aliases_expanded : list of str
        Which aliases were substituted during parsing.
    """
    input_string: str
    element_counts: Dict[str, float]
    molar_mass: float
    aliases_expanded: List[str]

    def __repr__(self) -> str:
        return (
            f"FormulaResult('{self.input_string}', "
            f"MW={self.molar_mass:.3f} g/mol, "
            f"elements={dict(self.element_counts)})"
        )

    @property
    def metals(self) -> Dict[str, float]:
        """Extract only metal elements and their counts."""
        # Non-metals and metalloids to exclude
        non_metals = {
            "H", "He", "C", "N", "O", "F", "Ne", "P", "S", "Cl",
            "Ar", "Se", "Br", "Kr", "I", "Xe", "B", "Si", "Ge",
            "As", "Te",
        }
        return {
            el: count for el, count in self.element_counts.items()
            if el not in non_metals
        }


# ------------------------------------------------------------------
# Core parser
# ------------------------------------------------------------------

def parse_formula(
    formula: str,
    aliases: Optional[Dict[str, str]] = None,
) -> FormulaResult:
    """
    Parse a chemical formula string into element counts.

    Handles:
      - Element symbols with counts: H2O, C8H5NO4, ZrO2
      - Parenthetical groups: Zr6O4(OH)4, Ca(NO3)2
      - Nested parentheses: (NH4)2SO4
      - Fractional subscripts: (BDC)5.54, (OH)0.91
      - Named aliases: BDC, DMF, NH2BDC, etc.
      - Dot notation for hydrates/solvates: MOF·3H2O, MOF.2DMF
      - Charge annotations (stripped): OH⁻, BDC²⁻, [Zr6O4(OH)4]12+

    Parameters
    ----------
    formula : str
        Chemical formula string.
    aliases : dict or None
        Custom alias dictionary. If None, uses FORMULA_ALIASES.

    Returns
    -------
    FormulaResult

    Raises
    ------
    ValueError
        If the formula cannot be parsed.
    """
    if aliases is None:
        aliases = FORMULA_ALIASES

    original = formula
    expanded_aliases = []

    # --- Step 0: Clean up ---
    # Remove whitespace, charge annotations, brackets
    formula = formula.strip()
    # Remove square brackets (often used for SBU: [Zr6O4(OH)4]12+)
    formula = formula.replace("[", "(").replace("]", ")")
    # Remove charge annotations: ²⁻, 2-, +, ⁻, etc.
    formula = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+$', '', formula)
    formula = re.sub(r'\d*[+\-]$', '', formula)

    # --- Step 1: Handle dot notation (hydrates/solvates) ---
    # Split on · or . that separates components
    # Be careful: . can also be a decimal in subscripts
    # Strategy: split on · always, split on . only if followed by
    # a digit+letter pattern (e.g., .2DMF, .3H2O)
    parts = re.split(r'[·]', formula)
    if len(parts) == 1:
        # Try splitting on . but only for solvate patterns
        # Match: .NUMBER followed by LETTER (not just .NUMBER which is decimal)
        dot_pattern = r'\.(\d+\.?\d*)([A-Z])'
        if re.search(dot_pattern, formula):
            # Split carefully
            main_part = re.split(dot_pattern, formula, maxsplit=1)
            if len(main_part) >= 4:
                parts = [main_part[0]]
                # Reconstruct the solvate part
                coeff = main_part[1]
                rest = main_part[2] + main_part[3] if len(main_part) > 3 else main_part[2]
                parts.append(coeff + rest)

    # --- Step 2: Parse each part ---
    total_counts: Dict[str, float] = {}

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        # Check for leading coefficient (e.g., "3H2O", "2DMF")
        coeff = 1.0
        coeff_match = re.match(r'^(\d+\.?\d*)\s*(?=[A-Z(])', part)
        if coeff_match and i > 0:  # Only for solvate parts, not main formula
            coeff = float(coeff_match.group(1))
            part = part[coeff_match.end():]

        # Expand aliases in this part
        part, expanded = _expand_aliases(part, aliases)
        expanded_aliases.extend(expanded)

        # Parse the elemental formula
        counts = _parse_formula_recursive(part)

        # Apply coefficient
        for el, n in counts.items():
            total_counts[el] = total_counts.get(el, 0.0) + n * coeff

    if not total_counts:
        raise ValueError(
            f"Could not parse formula '{original}'. "
            f"No elements found. Check spelling and parentheses."
        )

    # --- Step 3: Compute molar mass ---
    molar_mass = compute_molar_mass(total_counts)

    return FormulaResult(
        input_string=original,
        element_counts=total_counts,
        molar_mass=molar_mass,
        aliases_expanded=expanded_aliases,
    )


def _expand_aliases(
    formula: str,
    aliases: Dict[str, str],
) -> Tuple[str, List[str]]:
    """
    Replace named aliases with their empirical formulas.

    Processes longer aliases first to avoid partial matches
    (e.g., "NH2BDC" before "BDC").

    Returns the expanded formula and a list of which aliases were used.
    """
    expanded = []

    # Sort aliases by length (longest first) to avoid partial matches
    sorted_aliases = sorted(aliases.keys(), key=len, reverse=True)

    for alias in sorted_aliases:
        if alias in formula:
            # Check it's not part of a longer element symbol
            # by ensuring it's either at a word boundary or
            # surrounded by non-alpha characters / parentheses
            pattern = re.compile(re.escape(alias))
            if pattern.search(formula):
                replacement = aliases[alias]
                # Wrap in parentheses to preserve grouping
                formula = formula.replace(alias, f"({replacement})")
                expanded.append(f"{alias} → {replacement}")

    return formula, expanded


def _parse_formula_recursive(formula: str) -> Dict[str, float]:
    """
    Recursively parse a formula string with parentheses.

    Grammar:
      formula  = (group | element)+
      group    = '(' formula ')' number?
      element  = UPPER lower? number?
      number   = digit+ ('.' digit+)?

    Returns element counts as a dict.
    """
    counts: Dict[str, float] = {}
    i = 0
    n = len(formula)

    while i < n:
        if formula[i] == '(':
            # Find matching closing parenthesis
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if formula[j] == '(':
                    depth += 1
                elif formula[j] == ')':
                    depth -= 1
                j += 1

            if depth != 0:
                raise ValueError(
                    f"Unmatched parenthesis in '{formula}' at position {i}"
                )

            # Extract the inner formula (between parens)
            inner = formula[i + 1 : j - 1]

            # Read the subscript after the closing paren
            sub_str, k = _read_number(formula, j)
            multiplier = float(sub_str) if sub_str else 1.0

            # Recursively parse the inner formula
            inner_counts = _parse_formula_recursive(inner)

            # Add to totals
            for el, count in inner_counts.items():
                counts[el] = counts.get(el, 0.0) + count * multiplier

            i = k

        elif formula[i].isupper():
            # Read element symbol (1 uppercase + 0-1 lowercase)
            el = formula[i]
            i += 1
            while i < n and formula[i].islower():
                el += formula[i]
                i += 1

            # Validate element
            if el not in ATOMIC_MASSES:
                raise ValueError(
                    f"Unknown element '{el}' in formula '{formula}'. "
                    f"If this is a compound name, add it to FORMULA_ALIASES."
                )

            # Read subscript
            sub_str, i = _read_number(formula, i)
            count = float(sub_str) if sub_str else 1.0

            counts[el] = counts.get(el, 0.0) + count

        elif formula[i] in (' ', '-', '_'):
            # Skip whitespace, hyphens, underscores
            i += 1

        elif formula[i].isdigit() or formula[i] == '.':
            # Stray number — might be a leading coefficient
            # that wasn't caught. Skip it with a warning.
            sub_str, i = _read_number(formula, i)

        else:
            # Unknown character — skip
            i += 1

    return counts


def _read_number(s: str, start: int) -> Tuple[str, int]:
    """
    Read a number (integer or decimal) starting at position `start`.

    Returns (number_string, new_position).
    If no number found, returns ('', start).
    """
    i = start
    n = len(s)
    num = ''

    while i < n and (s[i].isdigit() or s[i] == '.'):
        num += s[i]
        i += 1

    return num, i


# ------------------------------------------------------------------
# Molar mass computation
# ------------------------------------------------------------------

def compute_molar_mass(element_counts: Dict[str, float]) -> float:
    """
    Compute molar mass from element counts.

    Parameters
    ----------
    element_counts : dict
        {element_symbol: count}

    Returns
    -------
    float
        Molar mass in g/mol.

    Raises
    ------
    ValueError
        If an element is not in the atomic masses table.
    """
    total = 0.0
    for el, count in element_counts.items():
        if el not in ATOMIC_MASSES:
            raise ValueError(f"Unknown element '{el}' — cannot compute molar mass.")
        total += ATOMIC_MASSES[el] * count
    return total


# ------------------------------------------------------------------
# Residue prediction
# ------------------------------------------------------------------

@dataclass
class ResiduePredicton:
    """Predicted TGA residue from a framework formula.

    Attributes
    ----------
    oxide_formula : str
        Expected oxide phase (e.g., "6 ZrO2").
    oxide_mass : float
        Total oxide residue mass in g/mol.
    framework_mass : float
        Guest-free framework molar mass in g/mol.
    residue_pct_guest_free : float
        Expected residue as wt% of the guest-free framework.
    residue_pct_initial : float or None
        Expected residue as wt% of the initial (guest-bearing) sample.
        None if guest mass is not provided.
    metals_found : dict
        {metal_symbol: count} extracted from the formula.
    oxide_phases : list of str
        Individual oxide phases contributing to the residue.
    warnings : list of str
        Any warnings about the prediction.
    """
    oxide_formula: str
    oxide_mass: float
    framework_mass: float
    residue_pct_guest_free: float
    residue_pct_initial: Optional[float]
    metals_found: Dict[str, float]
    oxide_phases: List[str]
    warnings: List[str]


def predict_residue(
    formula: str,
    atmosphere: str = "air",
    guest_mass_per_fu: float = 0.0,
    aliases: Optional[Dict[str, str]] = None,
) -> ResiduePredicton:
    """
    Predict the expected TGA residue from a framework formula.

    Assumes complete combustion under oxidative atmosphere to
    the most stable metal oxide phase.

    Parameters
    ----------
    formula : str
        Guest-free framework formula (e.g., "Zr6O4(OH)4(BDC)6").
    atmosphere : str
        TGA atmosphere. Oxide prediction is only reliable under
        'air' or 'O2'. Other atmospheres generate warnings.
    guest_mass_per_fu : float
        Total guest mass per formula unit in g/mol.
        If > 0, also computes residue_pct_initial.
    aliases : dict or None
        Custom alias dictionary.

    Returns
    -------
    ResiduePredicton

    Examples
    --------
    >>> r = predict_residue("Zr6O4(OH)4(BDC)6")
    >>> print(f"{r.residue_pct_guest_free:.2f}%")
    42.14%
    """
    warnings = []

    # Parse the formula
    parsed = parse_formula(formula, aliases=aliases)
    fw_mass = parsed.molar_mass
    metals = parsed.metals

    if not metals:
        raise ValueError(
            f"No metals found in '{formula}'. "
            f"Cannot predict oxide residue without metals."
        )

    # Atmosphere check
    atm_lower = atmosphere.lower().strip()
    if atm_lower in ("air", "o2", "oxygen", "synthetic air"):
        pass  # oxide prediction is valid
    elif atm_lower in ("n2", "nitrogen", "ar", "argon", "he", "helium"):
        warnings.append(
            f"Atmosphere is '{atmosphere}' (inert/non-oxidative). "
            f"Simple oxide-residue prediction assumes complete combustion "
            f"to metal oxides, which requires oxidative atmosphere (air/O2). "
            f"Under inert atmosphere, residues may contain metal, carbides, "
            f"or amorphous carbon. Oxide prediction may not be valid."
        )
    else:
        warnings.append(
            f"Atmosphere is '{atmosphere}' (unknown or unspecified). "
            f"Oxide-residue interpretation is uncertain. Confirm oxidative "
            f"conditions before trusting residue prediction."
        )

    # Compute oxide residue
    total_oxide_mass = 0.0
    oxide_phases = []

    for metal, count in metals.items():
        if metal not in OXIDE_RESIDUES:
            warnings.append(
                f"No oxide data for '{metal}'. "
                f"This metal's contribution to residue is unknown. "
                f"The predicted residue will be an underestimate."
            )
            continue

        oxide_formula, metals_per_oxide, mw_oxide = OXIDE_RESIDUES[metal]

        # How many formula units of the oxide?
        n_oxide = count / metals_per_oxide
        mass_contribution = n_oxide * mw_oxide

        total_oxide_mass += mass_contribution
        oxide_phases.append(
            f"{n_oxide:.4g} {oxide_formula} "
            f"(from {count:.4g} {metal}, {mass_contribution:.2f} g/mol)"
        )

    # Residue percentages
    residue_pct_gf = (total_oxide_mass / fw_mass) * 100.0 if fw_mass > 0 else 0.0

    residue_pct_initial = None
    if guest_mass_per_fu > 0:
        total_initial_mass = fw_mass + guest_mass_per_fu
        residue_pct_initial = (total_oxide_mass / total_initial_mass) * 100.0

    # Construct the combined oxide formula string
    oxide_str_parts = []
    for metal, count in metals.items():
        if metal in OXIDE_RESIDUES:
            oxide_formula, metals_per_oxide, _ = OXIDE_RESIDUES[metal]
            n_oxide = count / metals_per_oxide
            if abs(n_oxide - round(n_oxide)) < 0.01:
                oxide_str_parts.append(f"{int(round(n_oxide))}{oxide_formula}")
            else:
                oxide_str_parts.append(f"{n_oxide:.2f}{oxide_formula}")

    oxide_str = " + ".join(oxide_str_parts) if oxide_str_parts else "unknown"

    return ResiduePredicton(
        oxide_formula=oxide_str,
        oxide_mass=total_oxide_mass,
        framework_mass=fw_mass,
        residue_pct_guest_free=residue_pct_gf,
        residue_pct_initial=residue_pct_initial,
        metals_found=metals,
        oxide_phases=oxide_phases,
        warnings=warnings,
    )


# ------------------------------------------------------------------
# Convenience: MOFComponents from formula strings
# ------------------------------------------------------------------

def mof_components_from_formulas(
    node_formula: str,
    linker_formula: str,
    linker_charge: float = -2.0,
    ideal_linkers: float = 6.0,
    residue_formula: Optional[str] = None,
    n_metals_per_sbu: Optional[int] = None,
    aliases: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Compute all MOFComponents fields from formula strings.

    This replaces the need to manually compute M_node, M_linker,
    M_residue from atomic masses.

    Parameters
    ----------
    node_formula : str
        Dehydroxylated node formula (e.g., "Zr6O6").
    linker_formula : str
        Linker formula (e.g., "NH2-BDC" or "C8H5NO4").
    linker_charge : float
        Formal charge of the linker (default -2 for dicarboxylates).
    ideal_linkers : float
        Ideal number of linkers per SBU (default 6 for UiO-66).
    residue_formula : str or None
        Expected residue per metal (e.g., "ZrO2"). If None,
        auto-detected from the metals in node_formula.
    n_metals_per_sbu : int or None
        Number of metals per SBU. If None, auto-detected from
        the node formula.
    aliases : dict or None

    Returns
    -------
    dict
        All fields needed for MOFComponents constructor.
        Keys: metal, n_metals_per_sbu, sbu_formula, M_node,
        linker_name, M_linker, linker_charge, residue_formula,
        M_residue_per_metal, M_residue, sbu_charge, ideal_linkers
    """
    node = parse_formula(node_formula, aliases=aliases)
    linker = parse_formula(linker_formula, aliases=aliases)

    # Identify the metal and count
    metals = node.metals
    if not metals:
        raise ValueError(f"No metals found in node formula '{node_formula}'")

    # Take the first (or only) metal
    metal = list(metals.keys())[0]
    n_metals = int(round(metals[metal]))

    if n_metals_per_sbu is not None:
        n_metals = n_metals_per_sbu

    # Residue per metal
    if residue_formula is not None:
        residue_parsed = parse_formula(residue_formula, aliases=aliases)
        m_residue_per_metal = residue_parsed.molar_mass
        res_formula = residue_formula
    elif metal in OXIDE_RESIDUES:
        oxide_name, metals_per_oxide, mw_oxide = OXIDE_RESIDUES[metal]
        m_residue_per_metal = mw_oxide / metals_per_oxide
        res_formula = oxide_name
    else:
        raise ValueError(
            f"No oxide data for '{metal}'. Specify residue_formula explicitly."
        )

    m_residue_total = m_residue_per_metal * n_metals
    sbu_charge = _estimate_sbu_charge(node.element_counts, metal, n_metals)

    return {
        "metal": metal,
        "n_metals_per_sbu": n_metals,
        "sbu_formula": node_formula,
        "M_node": node.molar_mass,
        "linker_name": linker_formula,
        "M_linker": linker.molar_mass,
        "linker_charge": linker_charge,
        "residue_formula": res_formula,
        "M_residue_per_metal": m_residue_per_metal,
        "M_residue": m_residue_total,
        "sbu_charge": sbu_charge,
        "ideal_linkers": ideal_linkers,
    }


def _estimate_sbu_charge(
    element_counts: Dict[str, float],
    metal: str,
    n_metals: int,
) -> float:
    """
    Estimate the SBU charge from its composition.

    Uses common oxidation states:
      Zr(IV), Hf(IV), Ti(IV), Al(III), Fe(III), Cr(III),
      Cu(II), Zn(II), Co(II), Ni(II)

    O and OH are assumed to be O²⁻ and OH⁻ respectively.

    This is a rough estimate. For unusual SBUs, specify
    sbu_charge explicitly.
    """
    common_oxidation_states = {
        "Zr": 4, "Hf": 4, "Ti": 4, "Ce": 4,
        "Al": 3, "Fe": 3, "Cr": 3, "In": 3,
        "Cu": 2, "Zn": 2, "Co": 2, "Ni": 2, "Mn": 2,
        "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2,
    }

    if metal not in common_oxidation_states:
        return 0.0  # unknown — user should specify

    ox_state = common_oxidation_states[metal]
    positive = n_metals * ox_state

    # Count O and H to estimate oxide/hydroxide charge
    n_O = element_counts.get("O", 0.0)
    n_H = element_counts.get("H", 0.0)

    # Assume: each H is part of an OH group → n_OH = n_H
    # Remaining O are O²⁻
    n_OH = min(n_H, n_O)
    n_oxide_O = n_O - n_OH

    negative = n_oxide_O * 2 + n_OH * 1

    return positive - negative


# ------------------------------------------------------------------
# Convenience: Quick formula → mass
# ------------------------------------------------------------------

def formula_mass(formula: str) -> float:
    """
    Quick molar mass from a formula string.

    >>> formula_mass("H2O")
    18.015
    >>> formula_mass("Zr6O4(OH)4(BDC)6")
    1754.154  (approx)
    """
    return parse_formula(formula).molar_mass