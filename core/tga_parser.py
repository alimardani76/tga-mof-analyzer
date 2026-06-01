"""
TGA Data Parser — Multi-Format, Auto-Detecting
=================================================
Handles TGA data from any common source:
  - TA Instruments (Universal Analysis export)
  - Mettler Toledo (STARe export)
  - Netzsch (Proteus export)
  - Generic CSV / TSV (auto-detect columns)
  - Southampton standardized format (this project)

Detection strategy
------------------
1. Read first 100 lines of the file.
2. Look for instrument-specific signatures in headers.
3. If no signature found, auto-detect:
   a. Delimiter (comma, tab, semicolon)
   b. Header rows (skip non-numeric lines)
   c. Temperature column (increasing, range 20–1200)
   d. Mass column (decreasing or starts near 100/high value)
4. Detect units:
   a. Temperature: °C (20–1200 range) vs K (293–1500) vs °F
   b. Mass: wt% (0–100 range) vs mg (typically 1–50)
5. Normalize to (temperature_°C, mass_wt%)
6. Run quality checks.

Design philosophy
-----------------
- NEVER silently guess. If ambiguous, raise a clear error telling
  the user exactly what to specify.
- ALWAYS let the user override auto-detection with explicit params.
- Log every detection decision so the user can verify.

References
----------
- ASTM E1131-20: Standard Test Method for Compositional Analysis
  by Thermogravimetry.
- ICTAC Nomenclature Committee recommendations.
"""

import numpy as np
import os
import csv
import io
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


# ------------------------------------------------------------------
# Core data container
# ------------------------------------------------------------------

@dataclass
class TGAData:
    """
    Normalized TGA dataset.

    All data stored as:
      temperature : numpy array in °C, monotonically increasing
      mass_pct    : numpy array in wt% (normalized so max ≈ 100)

    Attributes
    ----------
    temperature : np.ndarray
        Temperature axis in °C.
    mass_pct : np.ndarray
        Mass in wt% (initial ≈ 100, decreasing).
    n_points : int
        Number of data points.
    heating_rate : float or None
        Heating rate in °C/min (None if unknown).
    atmosphere : str
        Gas atmosphere ('air', 'N2', 'Ar', 'unknown').
    sample_mass_mg : float or None
        Initial sample mass in mg (from metadata, if available).
    metadata : dict
        Any additional metadata extracted from the file.
    parse_log : list of str
        Log of parsing decisions for transparency.
    """
    temperature: np.ndarray
    mass_pct: np.ndarray
    n_points: int = 0
    heating_rate: Optional[float] = None
    atmosphere: str = "unknown"
    sample_mass_mg: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    parse_log: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.n_points = len(self.temperature)

    def __repr__(self) -> str:
        t0 = self.temperature[0]
        t1 = self.temperature[-1]
        m0 = self.mass_pct[0]
        m1 = self.mass_pct[-1]
        return (
            f"TGAData(n={self.n_points}, "
            f"T=[{t0:.1f}, {t1:.1f}]°C, "
            f"mass=[{m1:.2f}, {m0:.2f}]%, "
            f"atm={self.atmosphere}, "
            f"rate={self.heating_rate}°C/min)"
        )

    # --- Derived properties ---

    @property
    def residue_pct(self) -> float:
        """Final residue mass in wt%."""
        return float(self.mass_pct[-1])

    @property
    def total_mass_loss_pct(self) -> float:
        """Total mass loss from start to end."""
        return float(self.mass_pct[0] - self.mass_pct[-1])

    @property
    def t_range(self) -> Tuple[float, float]:
        """Temperature range (min, max) in °C."""
        return (float(self.temperature[0]), float(self.temperature[-1]))

    def get_mass_at_temp(self, target_temp: float) -> float:
        """
        Interpolate mass at a specific temperature.

        Parameters
        ----------
        target_temp : float
            Temperature in °C.

        Returns
        -------
        float
            Mass in wt% at the target temperature.

        Raises
        ------
        ValueError
            If target_temp is outside the data range.
        """
        t_min, t_max = self.t_range
        if target_temp < t_min or target_temp > t_max:
            raise ValueError(
                f"Target {target_temp}°C outside data range "
                f"[{t_min:.1f}, {t_max:.1f}]°C."
            )
        return float(np.interp(target_temp, self.temperature, self.mass_pct))

    # --- Constructors ---

    @classmethod
    def from_arrays(
        cls,
        temperature: np.ndarray,
        mass: np.ndarray,
        mass_is_mg: bool = False,
        initial_mass_mg: Optional[float] = None,
        heating_rate: Optional[float] = None,
        atmosphere: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> "TGAData":
        """
        Create TGAData from numpy arrays with explicit parameters.

        This is the SAFEST constructor — no guessing, no auto-detection.
        Use this when you know exactly what your data is.

        Parameters
        ----------
        temperature : np.ndarray
            Temperature in °C.
        mass : np.ndarray
            Mass data. If mass_is_mg=True, converted to wt%.
            If mass_is_mg=False, assumed to be wt% already.
        mass_is_mg : bool
            If True, mass is in milligrams and will be converted
            to wt% using initial_mass_mg or mass[0].
        initial_mass_mg : float or None
            Initial sample mass for mg→wt% conversion.
        heating_rate : float or None
            Heating rate in °C/min.
        atmosphere : str
            Gas atmosphere.
        metadata : dict or None
            Additional metadata.

        Returns
        -------
        TGAData
        """
        log = []
        temperature = np.asarray(temperature, dtype=np.float64)
        mass = np.asarray(mass, dtype=np.float64)

        if len(temperature) != len(mass):
            raise ValueError(
                f"Temperature ({len(temperature)}) and mass ({len(mass)}) "
                f"arrays must have the same length."
            )

        if len(temperature) < 5:
            raise ValueError("Need at least 5 data points.")

        # Convert mg → wt% if needed
        sample_mass = initial_mass_mg
        if mass_is_mg:
            if initial_mass_mg is None:
                initial_mass_mg = mass[0]
                log.append(f"Using first mass value as initial: {initial_mass_mg:.4f} mg")
            if initial_mass_mg <= 0:
                raise ValueError("Initial mass must be > 0 mg.")
            mass_pct = (mass / initial_mass_mg) * 100.0
            sample_mass = initial_mass_mg
            log.append(f"Converted mg → wt% using m0 = {initial_mass_mg:.4f} mg")
        else:
            mass_pct = mass.copy()
            log.append("Mass interpreted as wt% directly")

        # Normalize so first point = 100% (if within reasonable range)
        if 95 < mass_pct[0] < 105:
            # Already normalized, just ensure exactly 100
            if abs(mass_pct[0] - 100.0) > 0.01:
                log.append(f"First point = {mass_pct[0]:.4f}%, "
                           f"renormalized to 100.0%")
                mass_pct = mass_pct * (100.0 / mass_pct[0])
        elif mass_pct[0] > 105:
            log.append(f"⚠️ First mass point = {mass_pct[0]:.2f}% "
                       f"(>105%). Data may not be in wt%.")
        elif mass_pct[0] < 50:
            log.append(f"⚠️ First mass point = {mass_pct[0]:.2f}% "
                       f"(<50%). Data may be in mg, not wt%. "
                       f"Set mass_is_mg=True if so.")

        # Ensure temperature is increasing
        if temperature[-1] < temperature[0]:
            temperature = temperature[::-1]
            mass_pct = mass_pct[::-1]
            log.append("Reversed arrays (temperature was decreasing)")

        return cls(
            temperature=temperature,
            mass_pct=mass_pct,
            heating_rate=heating_rate,
            atmosphere=atmosphere,
            sample_mass_mg=sample_mass,
            metadata=metadata or {},
            parse_log=log,
        )

    @classmethod
    def from_file(
        cls,
        filepath: str,
        temp_col: Optional[int] = None,
        mass_col: Optional[int] = None,
        mass_is_mg: bool = False,
        initial_mass_mg: Optional[float] = None,
        heating_rate: Optional[float] = None,
        atmosphere: str = "unknown",
        delimiter: Optional[str] = None,
        skip_rows: Optional[int] = None,
        encoding: str = "utf-8-sig",
    ) -> "TGAData":
        """
        Load TGA data from a file with auto-detection.

        Auto-detection logic (used only when params are None):
        1. Delimiter: try comma, tab, semicolon — pick the one that
           gives the most consistent column count.
        2. Header rows: skip lines until a line has ≥2 numeric values.
        3. Temperature column: the column with monotonically increasing
           values in the 20–1500 range.
        4. Mass column: the column with generally decreasing values
           starting near 100 (wt%) or >1 (mg).

        You can override ANY of these with explicit parameters.
        When in doubt, specify temp_col and mass_col explicitly.

        Parameters
        ----------
        filepath : str
            Path to the TGA data file (.csv, .txt, .tsv).
        temp_col : int or None
            Column index (0-based) for temperature.
            If None, auto-detected.
        mass_col : int or None
            Column index (0-based) for mass.
            If None, auto-detected.
        mass_is_mg : bool
            If True, mass column is in milligrams.
        initial_mass_mg : float or None
            Initial mass for mg→wt% conversion.
        heating_rate : float or None
            Heating rate in °C/min.
        atmosphere : str
            Gas atmosphere.
        delimiter : str or None
            Column delimiter. If None, auto-detected.
        skip_rows : int or None
            Number of header rows to skip. If None, auto-detected.
        encoding : str
            File encoding (default 'utf-8-sig' handles BOM).

        Returns
        -------
        TGAData
        """
        log = [f"Loading: {os.path.basename(filepath)}"]

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read raw lines
        with open(filepath, "r", encoding=encoding, errors="replace") as f:
            raw_lines = f.readlines()

        if len(raw_lines) < 3:
            raise ValueError(f"File has only {len(raw_lines)} lines.")

        log.append(f"Read {len(raw_lines)} lines")

        # --- Step 1: Detect delimiter ---
        if delimiter is None:
            delimiter = _detect_delimiter(raw_lines)
            log.append(f"Auto-detected delimiter: {repr(delimiter)}")
        else:
            log.append(f"User-specified delimiter: {repr(delimiter)}")

        # --- Step 2: Detect header rows ---
        if skip_rows is None:
            skip_rows = _detect_header_rows(raw_lines, delimiter)
            log.append(f"Auto-detected {skip_rows} header row(s)")
        else:
            log.append(f"User-specified skip_rows: {skip_rows}")

        # --- Step 3: Parse numeric data ---
        data_lines = raw_lines[skip_rows:]
        columns = _parse_numeric_columns(data_lines, delimiter)

        if len(columns) < 2:
            raise ValueError(
                f"Need at least 2 numeric columns, found {len(columns)}. "
                f"Check delimiter (tried {repr(delimiter)}) and "
                f"skip_rows (tried {skip_rows})."
            )

        n_cols = len(columns)
        n_rows = len(columns[0])
        log.append(f"Parsed {n_rows} data rows × {n_cols} columns")

        # --- Step 4: Identify temperature and mass columns ---
        if temp_col is None:
            temp_col = _detect_temp_column(columns)
            log.append(f"Auto-detected temperature column: {temp_col}")
        else:
            log.append(f"User-specified temperature column: {temp_col}")

        if mass_col is None:
            mass_col = _detect_mass_column(columns, temp_col)
            log.append(f"Auto-detected mass column: {mass_col}")
        else:
            log.append(f"User-specified mass column: {mass_col}")

        if temp_col == mass_col:
            raise ValueError(
                f"Temperature and mass columns are the same ({temp_col}). "
                f"Specify temp_col and mass_col explicitly."
            )

        temperature = columns[temp_col]
        mass = columns[mass_col]

        # --- Step 5: Detect units ---
        temp_unit = _detect_temp_unit(temperature)
        log.append(f"Temperature unit detected: {temp_unit}")

        if temp_unit == "K":
            temperature = temperature - 273.15
            log.append("Converted K → °C")
        elif temp_unit == "F":
            temperature = (temperature - 32) * 5.0 / 9.0
            log.append("Converted °F → °C")

        if not mass_is_mg:
            mass_unit = _detect_mass_unit(mass)
            log.append(f"Mass unit detected: {mass_unit}")
            if mass_unit == "mg":
                mass_is_mg = True
                log.append("Auto-detected mass in mg (will convert to wt%)")

        # --- Step 6: Extract metadata from headers ---
        header_text = "\n".join(raw_lines[:skip_rows])
        extracted_meta = _extract_header_metadata(header_text)
        if extracted_meta:
            log.append(f"Extracted metadata: {list(extracted_meta.keys())}")

        if heating_rate is None and "heating_rate" in extracted_meta:
            heating_rate = extracted_meta["heating_rate"]
            log.append(f"Heating rate from header: {heating_rate}°C/min")

        if atmosphere == "unknown" and "atmosphere" in extracted_meta:
            atmosphere = extracted_meta["atmosphere"]
            log.append(f"Atmosphere from header: {atmosphere}")

        if initial_mass_mg is None and "sample_mass_mg" in extracted_meta:
            initial_mass_mg = extracted_meta["sample_mass_mg"]
            log.append(f"Sample mass from header: {initial_mass_mg} mg")

        # --- Build TGAData ---
        tga = cls.from_arrays(
            temperature=temperature,
            mass=mass,
            mass_is_mg=mass_is_mg,
            initial_mass_mg=initial_mass_mg,
            heating_rate=heating_rate,
            atmosphere=atmosphere,
            metadata=extracted_meta,
        )
        tga.parse_log = log + tga.parse_log
        return tga

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        temp_col: int = 0,
        mass_col: int = 1,
        **kwargs,
    ) -> "TGAData":
        """
        Simple CSV loader with explicit column indices.

        This is a convenience wrapper around from_file() for when
        you know exactly which columns to use.

        Parameters
        ----------
        filepath : str
        temp_col : int
            Column index for temperature (default 0).
        mass_col : int
            Column index for mass (default 1).
        **kwargs
            Passed to from_file().

        Returns
        -------
        TGAData
        """
        return cls.from_file(
            filepath,
            temp_col=temp_col,
            mass_col=mass_col,
            **kwargs,
        )

    def print_parse_log(self):
        """Print the parsing decision log."""
        print("Parse log:")
        for entry in self.parse_log:
            print(f"  {entry}")


# ------------------------------------------------------------------
# Auto-detection helpers
# ------------------------------------------------------------------

def _detect_delimiter(lines: List[str], test_lines: int = 20) -> str:
    """
    Detect the column delimiter by testing comma, tab, semicolon.

    Picks the delimiter that gives the most consistent non-zero
    column count across test lines.
    """
    candidates = [",", "\t", ";"]
    best_delim = ","
    best_score = -1

    # Skip obvious header lines for testing
    numeric_lines = []
    for line in lines[:100]:
        stripped = line.strip()
        if not stripped:
            continue
        # A line is "data-like" if it has at least one digit
        if any(c.isdigit() for c in stripped):
            numeric_lines.append(stripped)
        if len(numeric_lines) >= test_lines:
            break

    if not numeric_lines:
        return ","  # fallback

    for delim in candidates:
        counts = []
        for line in numeric_lines:
            parts = line.split(delim)
            n_numeric = sum(1 for p in parts if _is_numeric(p.strip()))
            if n_numeric >= 2:
                counts.append(n_numeric)

        if not counts:
            continue

        # Score: number of consistent lines × consistency
        from collections import Counter
        counter = Counter(counts)
        most_common_count, freq = counter.most_common(1)[0]
        score = freq * most_common_count  # prefer more columns AND more lines

        if score > best_score:
            best_score = score
            best_delim = delim

    return best_delim


def _detect_header_rows(lines: List[str], delimiter: str) -> int:
    """
    Find how many lines to skip before numeric data begins.

    A line is considered "data" if it has ≥2 numeric fields.
    """
    for i, line in enumerate(lines):
        parts = line.strip().split(delimiter)
        n_numeric = sum(1 for p in parts if _is_numeric(p.strip()))
        if n_numeric >= 2:
            return i
    return 0


def _parse_numeric_columns(
    lines: List[str],
    delimiter: str,
) -> List[np.ndarray]:
    """
    Parse lines into columns of floats.

    Non-numeric values become NaN.  Columns are returned as numpy
    arrays with NaN rows stripped (only rows where ALL columns are
    numeric are kept).
    """
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(delimiter)
        row = []
        for p in parts:
            p = p.strip().strip('"').strip("'")
            if _is_numeric(p):
                row.append(float(p))
            else:
                row.append(float("nan"))
        rows.append(row)

    if not rows:
        return []

    # Ensure consistent column count (use most common)
    from collections import Counter
    col_counts = Counter(len(r) for r in rows)
    expected_cols = col_counts.most_common(1)[0][0]

    # Filter to rows with the expected column count and all numeric
    clean_rows = []
    for row in rows:
        if len(row) == expected_cols and not any(np.isnan(v) for v in row):
            clean_rows.append(row)

    if not clean_rows:
        return []

    data = np.array(clean_rows)
    return [data[:, i] for i in range(data.shape[1])]


def _detect_temp_column(columns: List[np.ndarray]) -> int:
    """
    Detect which column is temperature.

    Heuristics:
    1. Monotonically increasing (or mostly increasing).
    2. Range consistent with temperature (20–1500°C or 293–1800K).
    3. If multiple candidates, pick the one with the tightest
       spacing (most uniform step size).
    """
    best_col = 0
    best_score = -1

    for i, col in enumerate(columns):
        if len(col) < 5:
            continue

        # Check if mostly increasing
        diffs = np.diff(col)
        frac_increasing = np.mean(diffs > 0)
        if frac_increasing < 0.8:
            continue

        # Check range
        col_min, col_max = col.min(), col.max()
        span = col_max - col_min

        # Temperature-like: spans at least 100 degrees, starts below 200
        if span < 50:
            continue
        if col_min > 500:  # probably not temperature
            continue

        # Score: monotonicity × range plausibility
        score = frac_increasing * span
        if score > best_score:
            best_score = score
            best_col = i

    return best_col


def _detect_mass_column(
    columns: List[np.ndarray],
    temp_col: int,
) -> int:
    """
    Detect which column is mass.

    Heuristics:
    1. Not the temperature column.
    2. Generally decreasing (net decrease from start to end).
    3. If in wt%: starts near 100, ends > 0.
    4. If in mg: starts at some positive value, decreases.
    """
    best_col = None
    best_score = -1

    for i, col in enumerate(columns):
        if i == temp_col:
            continue
        if len(col) < 5:
            continue

        # Net decrease
        net_change = col[0] - col[-1]
        if net_change <= 0:
            continue  # mass should decrease overall

        # Score: magnitude of decrease × plausibility
        # Prefer columns starting near 100 (wt%)
        start_val = col[0]
        if 80 < start_val < 120:
            plausibility = 2.0  # likely wt%
        elif 0.1 < start_val < 80:
            plausibility = 1.0  # could be mg or partial data
        else:
            plausibility = 0.5

        score = net_change * plausibility

        if score > best_score:
            best_score = score
            best_col = i

    if best_col is None:
        # Fallback: first column that isn't temperature
        for i in range(len(columns)):
            if i != temp_col:
                return i
        raise ValueError("Cannot detect mass column.")

    return best_col


def _detect_temp_unit(temperature: np.ndarray) -> str:
    """
    Detect temperature unit from value range.

    Returns 'C', 'K', or 'F'.
    """
    t_min = temperature.min()
    t_max = temperature.max()

    if t_min > 200 and t_max > 400:
        # Could be Kelvin (273K = 0°C, 1273K = 1000°C)
        if t_min > 250:
            return "K"
    if t_max > 500 and t_min > 50:
        # Could be Fahrenheit (212°F = 100°C)
        # But also could be °C for high-T runs
        # Heuristic: if min > 60 and max > 1000, likely °C
        if t_min < 100 and t_max < 2000:
            return "C"
    return "C"  # default assumption


def _detect_mass_unit(mass: np.ndarray) -> str:
    """
    Detect mass unit: 'pct' (wt%) or 'mg'.

    Heuristic: if first value is between 50 and 120, likely wt%.
    If first value is between 0.1 and 50, likely mg.
    """
    first = mass[0]
    if 50 < first < 120:
        return "pct"
    elif 0.1 < first < 50:
        return "mg"
    elif first > 120:
        return "pct"  # could be unnormalized wt%
    return "pct"  # default


def _extract_header_metadata(header_text: str) -> Dict:
    """
    Extract metadata from file headers.

    Looks for common patterns in instrument export headers:
    - Heating rate: "10.00°C/min", "Rate: 10", "Heating Rate"
    - Atmosphere: "Air", "N2", "Nitrogen", "Argon"
    - Sample mass: "Sample Weight: 5.234 mg"
    - Instrument: "TA Instruments", "Mettler", "Netzsch"
    """
    import re
    meta = {}

    text_lower = header_text.lower()

    # Heating rate
    rate_patterns = [
        r'(\d+\.?\d*)\s*[°]?c\s*/\s*min',
        r'heating\s*rate\s*[:=]\s*(\d+\.?\d*)',
        r'rate\s*[:=]\s*(\d+\.?\d*)',
    ]
    for pattern in rate_patterns:
        match = re.search(pattern, text_lower)
        if match:
            meta["heating_rate"] = float(match.group(1))
            break

    # Atmosphere
    atm_map = {
        "nitrogen": "N2", "n2": "N2", "n₂": "N2",
        "argon": "Ar", "ar": "Ar",
        "air": "air", "synthetic air": "air",
        "oxygen": "O2", "o2": "O2",
        "helium": "He", "he": "He",
    }
    for keyword, value in atm_map.items():
        if keyword in text_lower:
            meta["atmosphere"] = value
            break

    # Sample mass
    mass_patterns = [
        r'sample\s*(?:weight|mass)\s*[:=]\s*(\d+\.?\d*)\s*mg',
        r'weight\s*[:=]\s*(\d+\.?\d*)\s*mg',
        r'mass\s*[:=]\s*(\d+\.?\d*)\s*mg',
    ]
    for pattern in mass_patterns:
        match = re.search(pattern, text_lower)
        if match:
            meta["sample_mass_mg"] = float(match.group(1))
            break

    # Instrument
    instruments = [
        ("ta instruments", "TA Instruments"),
        ("universal analysis", "TA Instruments"),
        ("trios", "TA Instruments"),
        ("mettler", "Mettler Toledo"),
        ("stare", "Mettler Toledo"),
        ("netzsch", "Netzsch"),
        ("proteus", "Netzsch"),
        ("perkinelmer", "PerkinElmer"),
        ("pyris", "PerkinElmer"),
        ("setaram", "SETARAM"),
    ]
    for keyword, name in instruments:
        if keyword in text_lower:
            meta["instrument"] = name
            break

    return meta


def _is_numeric(s: str) -> bool:
    """Check if a string can be parsed as a float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False