# TGA-MOF-Analyzer

Automated thermogravimetric analysis toolkit for metal–organic frameworks.

Validated on 14 TGA curves from open-access data (Crickmore & Bradshaw, DOI: 10.5258/SOTON/D2004).
Equations traceable to Abánades Lázaro (EJIC 2020), Shearer (Chem. Mater. 2016), Sannes (Chem. Mater. 2023).

---

## What It Computes

| Output | Method | Required Input |
|--------|--------|----------------|
| Decomposition temperature (T_onset) | Tangent intersection (ASTM E2550), threshold, DTG max | TGA data only |
| T_1, T_2, T_5, T_10 stability metrics | Linear interpolation from activated mass | TGA data only |
| Mass-loss events with 100% mass balance | Valley-to-valley DTG segmentation | TGA data only |
| Mass loss by temperature windows | Interpolated mass at window boundaries | TGA data only |
| Predicted oxide residue | Formula → metal → oxide lookup | TGA data + formula |
| Observed vs predicted residue comparison | Δ = observed − predicted | TGA data + formula |
| Molecular formula (linkers per FU) | R_exp = m(T_DH)/m_residue, q equation | TGA data + formula + T_DH |
| Composition sensitivity (q vs T_DH) | Sweep T_DH from 150–490°C | TGA data + formula |
| Missing linker defects | 5 compensator models (Shearer/Sannes) | Formula + Module 3 output |
| Composite MOF loading (wt%) | Residue ratio equation | TGA data + reference residues |
| Guest stoichiometry candidates | Brute-force enumeration with scoring | TGA data + framework mass |
| Uncertainty (σ) on all quantities | Analytical linear propagation (GUM) | TGA noise estimate |
| Publication-quality plots | TGA+DTG, event map, R_exp, sensitivity | TGA data |

---

## Installation

```

pip install numpy scipy matplotlib

```

For development and testing:

```

pip install numpy scipy matplotlib pytest streamlit

```

---

## Usage

### Option 1: Streamlit web app (no Python knowledge needed)

```

pip install streamlit
streamlit run app.py

```

Opens a browser interface. Upload CSV, select MOF type, click Run.

Deployed version: https://tga-mof-analyzer.streamlit.app

### Option 2: Command-line tool

```

python tga_analyze.py --wizard

```

Interactive wizard asks questions and runs the analysis.

Or create a JSON config and run directly:

```

python tga_analyze.py my\_config.json
python tga_analyze.py my\_config.json --output results/
python tga_analyze.py my\_config.json --no-plots

```

Generate a config template:

```

python tga_analyze.py --example-simple > config.json

````

### Option 3: Python API (for notebooks or scripts)

```python
from core.tga_parser import TGAData
from core.tga_quality import run_quality_checks
from core.dtg import compute_dtg, detect_events
from core.formula_parser import parse_formula, predict_residue
from modules.m1_thermal_stability import analyze_stability
from modules.m2_guest_content import analyze_guest_content_windows

tga = TGAData.from_file("my_data.csv", atmosphere="air", heating_rate=10)
qr = run_quality_checks(tga)
stab = analyze_stability(tga)
win = analyze_guest_content_windows(tga)

print(f"T_onset = {stab.t_onset:.1f}°C")
print(f"Residue = {tga.residue_pct:.2f}%")
print(f"Events = {stab.events.n_events}")
````

### Option 4: Full 14-case validation run

```
python run_analysis.py
```

Requires the case data files in `cases/` and `blank_controls/`.

***

## Supported TGA Data Formats

Any CSV or text file with temperature and mass columns.

The parser auto-detects:

* Delimiter: comma, tab, semicolon
* Header rows: skipped automatically
* Temperature column: monotonically increasing, range 0–1500
* Mass column: generally decreasing, starts near 100 (wt%) or <50 (mg)
* Temperature units: °C or K (converted automatically)
* Mass units: wt% or mg (normalized to 100% automatically)

Example formats that work:

```
Temperature (°C),Mass (%)
25.0,100.00
26.0,99.95
```

```
Temp	Weight%
25.4	100.000
26.4	99.967
```

```
°C;mg
25.0;5.234
26.0;5.231
```

***

## JSON Configuration Reference

### Minimal config (only file path required)

```json
{
  "case_id": "my_sample",
  "tga_csv": "data/sample.csv"
}
```

### Standard config

```json
{
  "case_id": "UiO66_NH2_batch1",
  "tga_csv": "data/sample.csv",
  "formula": "Zr6O4(OH)4(NH2BDC)6",
  "atmosphere": "air",
  "heating_rate": 10
}
```

### Full config

```json
{
  "case_id": "UiO66_detailed",
  "tga_csv": "data/sample.csv",
  "formula": "Zr6O4(OH)4(NH2BDC)6",
  "atmosphere": "air",
  "heating_rate": 10,
  "decomp_start": 342,
  "guests": "H2O,DMF,EtOH",
  "window_start": 30,
  "window_end": 100,
  "tolerance": 1.0
}
```

### Batch config (multiple samples)

```json
{
  "runs": [
    {
      "case_id": "sample_A",
      "tga_csv": "data/A.csv",
      "formula": "Zr6O4(OH)4(BDC)6",
      "atmosphere": "air"
    },
    {
      "case_id": "sample_B",
      "tga_csv": "data/B.csv",
      "atmosphere": "N2"
    }
  ]
}
```

### Field reference

| Field                 | Type   | Required | Default      | Description                                                       |
| --------------------- | ------ | -------- | ------------ | ----------------------------------------------------------------- |
| `case_id`             | string | no       | `"analysis"` | Names output files                                                |
| `tga_csv`             | string | yes      | —            | Path to TGA data file                                             |
| `formula`             | string | no       | none         | Guest-free framework formula. Accepts aliases (BDC, NH2BDC, etc.) |
| `atmosphere`          | string | no       | `"air"`      | TGA purge gas: air, N2, Ar, O2                                    |
| `heating_rate`        | float  | no       | none         | °C/min                                                            |
| `observed_final_mass` | float  | no       | auto         | Override residue (wt%)                                            |
| `decomp_start`        | float  | no       | auto         | Framework decomposition onset (°C)                                |
| `guests`              | string | no       | none         | Comma-separated guest candidates for enumeration                  |
| `window_start`        | float  | no       | 30           | Start of guest analysis window (°C)                               |
| `window_end`          | float  | no       | 100          | End of guest analysis window (°C)                                 |
| `tolerance`           | float  | no       | 1.0          | Mass-loss matching tolerance (wt%)                                |

***

## Formula Syntax

Standard chemical notation with parentheses and subscripts:

```
H2O
ZrO2
Zr6O4(OH)4(BDC)6
Cu3(BTC)2
Zn(MeIM)2
Al(OH)(BDC)
```

### Available aliases

These expand automatically when used in formulas:

| Alias  | Expands to | Name                                        |
| ------ | ---------- | ------------------------------------------- |
| BDC    | C8H4O4     | Terephthalate                               |
| NH2BDC | C8H5NO4    | 2-aminoterephthalate                        |
| BPDC   | C12H8O4    | Biphenyl-4,4'-dicarboxylate                 |
| NDC    | C12H6O4    | 2,6-naphthalenedicarboxylate                |
| BTC    | C9H3O6     | Trimesate                                   |
| BTB    | C27H15O6   | 1,3,5-tri(4-carboxyphenyl)benzene           |
| DOBDC  | C8H2O6     | 2,5-dioxidoterephthalate                    |
| TATB   | C24H12N3O6 | 4,4',4''-s-triazine-2,4,6-triyl-tribenzoate |
| IM     | C3H3N2     | Imidazolate                                 |
| MeIM   | C4H5N2     | 2-methylimidazolate                         |
| PhIM   | C9H7N2     | Phenylimidazolate                           |
| FA     | CHO2       | Formate                                     |
| AC     | C2H3O2     | Acetate                                     |
| TFA    | C2F3O2     | Trifluoroacetate                            |
| BA     | C7H5O2     | Benzoate                                    |
| DMF    | C3H7NO     | N,N-dimethylformamide                       |
| DMA    | C4H9NO     | N,N-dimethylacetamide                       |
| DEF    | C5H11NO    | N,N-diethylformamide                        |
| DMSO   | C2H6OS     | Dimethyl sulfoxide                          |
| MeOH   | CH4O       | Methanol                                    |
| EtOH   | C2H6O      | Ethanol                                     |
| THF    | C4H8O      | Tetrahydrofuran                             |
| DCM    | CH2Cl2     | Dichloromethane                             |
| NMP    | C5H9NO     | N-methyl-2-pyrrolidone                      |
| ACN    | C2H3N      | Acetonitrile                                |
| H2O    | H2O        | Water                                       |

If your formula is unknown, omit the `formula` field. Modules 1, 2, and 5 still work without it.

***

## Output Files

For each analysis run, the tool produces:

| File                        | Contents                                                      |
| --------------------------- | ------------------------------------------------------------- |
| `{case_id}_result.json`     | All computed values in machine-readable JSON                  |
| `{case_id}_report.txt`      | Human-readable report with suggested publication wording      |
| `{case_id}_tga_dtg.png`     | TGA + DTG overlay plot                                        |
| `{case_id}_events.png`      | Color-coded mass-loss event map                               |
| `{case_id}_rexp.png`        | R\_exp(T) curve with plateau annotation (if formula provided) |
| `{case_id}_sensitivity.png` | q vs T\_DH sensitivity plot (if formula provided)             |

***

## Testing

```
python -m pytest tests/ -v
```

45 tests in 4 groups:

| Group | File                          | Tests | Purpose                                                       |
| ----- | ----------------------------- | ----- | ------------------------------------------------------------- |
| A     | `test_A_happy_path.py`        | 22    | Normal operation: parsing, residue, uncertainty, validation   |
| B     | `test_B_edge_cases.py`        | 6     | Incomplete inputs: missing atmosphere, heating rate, formula  |
| C     | `test_C_expected_failures.py` | 7     | Invalid inputs: reversed windows, bad formulas, missing files |
| D     | `test_D_regression.py`        | 10    | Reference values: framework mass, residue %, element counts   |

***

## Repository Structure

```
tga-mof-analyzer/
├── app.py
├── tga_analyze.py
├── run_analysis.py
├── README.md
├── TECHNICAL_GUIDE.md
├── LICENSE
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── .gitignore
├── .streamlit/
│   └── config.toml
├── core/
│   ├── __init__.py
│   ├── tga_parser.py
│   ├── tga_quality.py
│   ├── dtg.py
│   ├── rexp.py
│   ├── guest_solver.py
│   ├── charge_balance.py
│   ├── formula_parser.py
│   ├── residue_analysis.py
│   ├── uncertainty.py
│   ├── validator.py
│   ├── json_runner.py
│   └── report.py
├── modules/
│   ├── __init__.py
│   ├── m1_thermal_stability.py
│   ├── m2_guest_content.py
│   ├── m3_composition.py
│   ├── m4_defect_quantification.py
│   └── m5_composite_loading.py
├── viz/
│   ├── __init__.py
│   └── plotting.py
├── tests/
│   ├── __init__.py
│   ├── test_A_happy_path.py
│   ├── test_B_edge_cases.py
│   ├── test_C_expected_failures.py
│   ├── test_D_regression.py
│   └── reference_values.json
├── examples/
│   ├── sample_tga.csv
│   └── sample_config.json
├── cases/
│   └── (14 TGA data directories — not in public repo)
├── blank_controls/
│   └── (2 blank TGA directories — not in public repo)
├── validation/
│   └── cross_check.py
└── output/
    └── (generated files — in .gitignore)
```

***

## File Descriptions

### Root files

**`app.py`** — Streamlit web interface. Handles file upload, sidebar inputs, analysis execution, plot rendering, and download buttons. All computation is delegated to `core/` and `modules/`. No analysis logic lives here.

**`tga_analyze.py`** — Command-line interface. Parses arguments (`--wizard`, `--example`, config path). Loads JSON config, validates inputs, calls analysis functions, saves JSON result and text report. Supports single and batch runs.

**`run_analysis.py`** — Development/validation script. Hardcoded analysis of all 14 cases from Crickmore & Bradshaw data. Runs all 5 modules, generates all plots, prints the full summary table. Not for end users.

**`README.md`** — This file.

**`TECHNICAL_GUIDE.md`** — Complete technical documentation. Physics, equations, architecture, assumptions, limitations, design choices, modification guide.

**`LICENSE`** — MIT License.

**`requirements.txt`** — Production dependencies. Used by Streamlit Cloud for deployment. Contains: numpy, scipy, matplotlib.

**`requirements-dev.txt`** — Development dependencies. Adds: pytest, streamlit.

**`setup.py`** — Package installer. Enables `pip install .` and `pip install -e .` for editable development. Defines the `tga-analyze` console script entry point.

**`.gitignore`** — Excludes `__pycache__/`, `output/`, `*.png`, `venv/`, `.pytest_cache/`, IDE files, and user-specific config files from version control.

### `.streamlit/`

**`config.toml`** — Streamlit theme (colors, font) and server settings (max upload size 50 MB).

### `core/` — Computation engine

All math lives here. No file I/O except reading the TGA CSV. No plotting. No user interaction.

**`__init__.py`** — Empty. Makes `core/` a Python package.

**`tga_parser.py`** — Loads TGA data from CSV/TXT files into a `TGAData` object. Auto-detects delimiter (comma/tab/semicolon), header rows, temperature column (monotonically increasing), mass column (generally decreasing), temperature units (°C/K), and mass units (wt%/mg). Normalizes mass to 100% at first data point. Logs every auto-detection decision to `parse_log`. Three constructors: `from_arrays` (explicit), `from_file` (auto-detect), `from_csv` (simple known layout). Properties: `residue_pct`, `total_mass_loss_pct`, `t_range`, `get_mass_at_temp(T)`.

**`tga_quality.py`** — Runs 7 automated checks on parsed TGA data before analysis. Checks: sufficient data points (≥50), temperature range (≥200°C span), temperature monotonicity, mass range (start ≈100%, end >−5%), noise level (median local σ from sliding window), residue stability (range and σ of last 5% of data), initial artifact detection (sharp drop in first 10°C). Returns a `QualityReport` with `noise_estimate_pct` (used by uncertainty propagation) and `has_initial_artifact` flag.

**`dtg.py`** — Computes the DTG curve and detects mass-loss events. `compute_dtg` uses `scipy.signal.savgol_filter` with `deriv=1` for simultaneous smoothing and differentiation. The Savitzky-Golay window auto-scales from data density and heating rate. Does not clip negative values (preserves mass-gain information). `detect_events` finds DTG peaks via `scipy.signal.find_peaks` with prominence threshold, then finds valleys (DTG minima) between adjacent peaks. Events span valley-to-valley: first event starts at data start, last event ends at data end. This guarantees that the sum of all event losses plus the residue equals 100.00%. Small events below `min_mass_loss` are merged into adjacent events. Initial artifacts are flagged but not removed.

**`rexp.py`** — Implements the R\_exp composition method from Abánades Lázaro (2020). `compute_rexp` calculates R\_exp(T) = m(T)/m\_residue at every temperature. `find_dh_plateau` slides a window across R\_exp(T) and picks the flattest region (minimum std of dR/dT). `compute_linkers` solves q = (R\_exp\_DH × M\_residue − M\_node) / M\_linker. Also provides `compute_formula_mass`, `compute_theoretical_rexp`, and `compute_linkers_with_modulator` for extended calculations.

**`guest_solver.py`** — Enumerates all stoichiometric guest combinations that reproduce an observed mass loss within tolerance. Uses brute-force grid search over coefficients (step 0.5, max 10 per guest, max 4 guests simultaneously). The key equation: f\_calc = Σ(n\_i × MW\_i) / (M\_F + Σ(n\_i × MW\_i)), where the denominator is total mass (framework + guests), not just framework mass. Surviving combinations are ranked by a penalty function: mass error + pore volume violation + boiling point violation. Contains `COMMON_GUESTS` (list of `GuestCandidate` objects) and `GUEST_LIBRARY` (dict with formula, MW, boiling point, name for 15 solvents). Also contains `score_guest_assignment` which scores assignments by: 10×|error| + 0.03×Σn (Occam) + 2.0 per species beyond 3 + 3.0 per bp mismatch + 1.0 per high-count guest.

**`charge_balance.py`** — Checks if a MOF formula is charge-balanced. Computes positive charge (from SBU), negative charge (from linkers × charge), and residual. Reports balanced if |residual| < 0.5. Also computes how many compensating ions are needed to balance a defective formula.

**`formula_parser.py`** — Parses chemical formula strings into element counts and molar masses. Contains `ATOMIC_MASSES` (90 elements, IUPAC 2021), `FORMULA_ALIASES` (30+ MOF-relevant aliases like BDC→C8H4O4), and `OXIDE_RESIDUES` (26 metals → oxide formula, metals per oxide, MW). The parser handles parentheses (nested), subscripts (integer and decimal), dot notation for hydrates, leading coefficients, and alias expansion. `parse_formula` returns a `FormulaResult` with `element_counts`, `molar_mass`, and `aliases_expanded`. `predict_residue` takes a formula, finds metals, looks up their oxides, and computes predicted residue percentage. `formula_mass` is a convenience function: formula string → float. Verified: Zr6O4(OH)4(NH2BDC)6 = 1754.154 g/mol, predicted residue = 42.15%.

**`residue_analysis.py`** — Compares observed TGA residue to predicted oxide residue. Computes Δ = observed − predicted. Interprets the difference: |Δ|<2pp = excellent match; Δ<0 = excess organic (guests or defects); Δ>0 = inorganic impurity. Includes atmosphere-dependent warnings (inert atmosphere makes oxide prediction unreliable).

**`uncertainty.py`** — Analytical (linear) uncertainty propagation following JCGM 100:2008 (GUM). Three functions: `guest_count_uncertainty` (σ for n guests/FU), `composition_uncertainty` (σ for q linkers/FU), `composite_loading_uncertainty` (σ for w\_MOF). Each computes partial derivatives and combines via σ² = Σ(∂f/∂xi)²σ\_xi². Also provides `sigma_from_quality` which extracts σ\_mass and σ\_residue from a `QualityReport`. The `UncertainValue` dataclass holds value ± σ with unit string and relative percentage.

**`validator.py`** — Checks all inputs before computation. Returns `ValidationResult` with lists of errors and warnings, never Python tracebacks. Checks: file exists, window\_start < window\_end, window doesn't cross decomp\_start, formula parseable, guests known, atmosphere specified, observed mass reasonable, heating rate positive and not extreme, numeric fields non-negative.

**`json_runner.py`** — Reads a JSON config file, validates it, routes to appropriate modules, and returns structured JSON output. Supports single run or batch (`"runs"` array). Calls: validator → tga\_parser → tga\_quality → Module 1 → Module 2 → residue analysis.

**`report.py`** — Generates output files. `save_report_json` writes results as JSON with numpy serialization (converts ndarray to list, nan to null). `generate_methods_wording` produces a suggested methods paragraph for publications. `generate_results_wording` produces a suggested results paragraph. `generate_full_report_text` combines all sections into a complete human-readable report.

### `modules/` — Analysis modules

Each module answers one specific question. Each depends on `core/` for computation.

**`__init__.py`** — Empty.

**`m1_thermal_stability.py`** — "At what temperature does my MOF decompose?" Calls `compute_dtg` and `detect_events`. Identifies the decomposition event (largest DTG peak above 250°C). Computes T\_onset by tangent intersection (line through DTG peak projected to baseline mass). Computes T\_onset by threshold method (first T where DTG exceeds a dynamic threshold). Computes T\_DTG\_max (temperature of maximum DTG in decomposition region). Computes T\_x metrics (T\_1, T\_2, T\_5, T\_10) by linear interpolation from activated mass. Computes stability window (end of activation to start of decomposition). Returns `StabilityResult`.

**`m2_guest_content.py`** — "How much guest/solvent is in my MOF?" Two modes. Event-based mode (`analyze_guest_content`) uses detected events. Window-based mode (`analyze_guest_content_windows`) uses user-specified or default temperature windows: RT–120°C, 120–250°C, 250–400°C, 400–600°C, 600°C–end. For each window, loss = m(T\_start) − m(T\_end) via linear interpolation. Sum of all windows + residue = 100%. Returns `GuestContentResult`.

**`m3_composition.py`** — "What is the molecular formula of my MOF?" Takes `TGAData`, `MOFComponents` (M\_node, M\_linker, M\_residue, q\_ideal), and a user-specified T\_DH. Interpolates mass at T\_DH. Computes R\_exp\_DH = m(T\_DH) / m\_residue. Computes q = (R\_exp\_DH × M\_residue − M\_node) / M\_linker. Checks charge balance. Generates formula string. Warns if q > ideal, charge imbalanced, or compensator count negative. Returns `CompositionResult`.

**`m4_defect_quantification.py`** — "How many linkers are missing?" Takes `CompositionResult` from Module 3. Simple estimate: x = q\_ideal − q\_experimental. Then iterates over 5 compensator models (vacancy, OH/H2O, formate, acetate, chloride), solving x = (M\_ideal − M\_obs) / (M\_linker − n\_cap × M\_cap) for each. Checks charge balance for each model. Reports coordination number. Warns that TGA cannot distinguish missing-linker from missing-cluster defects. Returns `DefectResult`.

**`m5_composite_loading.py`** — "What fraction of my composite is MOF?" Takes composite TGA data plus reference residue fractions for pure MOF and pure additive. Computes r\_composite from composite TGA residue. Applies w\_MOF = (r\_composite − r\_additive) / (r\_MOF − r\_additive). Warns if r\_MOF ≈ r\_additive (denominator → 0, method unreliable). Warns if w\_MOF outside 0–100%. Returns `CompositeResult`.

### `viz/` — Visualization

**`__init__.py`** — Empty.

**`plotting.py`** — Five plot types, all using matplotlib. All save as PNG at 150+ dpi. Consistent styling: 10×5 figure, 12pt fonts, alpha=0.3 grid.

1. `plot_tga_dtg(tga, dtg, save_path)` — Dual-axis plot. Left axis (blue): TGA mass %. Right axis (red): DTG %/°C. Combined legend.

2. `plot_event_map(tga, events, save_path)` — TGA curve with each event filled in a different color (Set3 colormap). Legend shows event number and mass loss. Up to 12 events labeled.

3. `plot_rexp_curve(tga, rexp, save_path)` — Upper panel: R\_exp(T) with horizontal line at plateau value and vertical line at suggested T\_DH. Lower panel: dR/dT showing where the plateau is flattest.

4. `plot_composition_sensitivity(sweep_data, save_path)` — q vs T\_DH curve. Horizontal line at q\_ideal. Shaded region for "plausible" range (|deficiency| < 15%). Equation annotation. Arrow at auto-plateau suggestion.

5. `plot_composite_series(data, save_path)` — Stacked bar chart showing MOF% and additive% for a series of samples. Equation annotation.

### `tests/` — Test suite

**`__init__.py`** — Empty.

**`test_A_happy_path.py`** — 22 tests. Normal operation. Tests formula parsing (H2O, ZrO2, NH2BDC, UiO-66-NH2, aliases, fractional subscripts, nested parentheses, invalid element raises ValueError). Tests residue prediction (42.15% for UiO-66-NH2, oxide formula, inert atmosphere warning, no-metals raises error). Tests uncertainty (positive σ for guests, composition, composites; zero denominator handling). Tests validation (valid config passes, missing CSV fails, reversed window fails, invalid formula fails, window crossing decomp warns).

**`test_B_edge_cases.py`** — 6 tests. Incomplete inputs. Tests: unknown atmosphere → warning not error; missing heating rate → info not error; no observed\_final\_mass → info not error; no formula → valid (skip M3/M4); negative heating rate → error; very high heating rate → warning.

**`test_C_expected_failures.py`** — 7 tests. Invalid inputs that must be rejected. Tests: reversed window → error; unknown guest → error; invalid formula → error; missing CSV → error; strongly negative residue → error; invalid element in formula → ValueError; unmatched parenthesis → ValueError.

**`test_D_regression.py`** — 10 tests. Reference values that must not change. Tests: framework mass = 1754.154 ± 0.01; predicted residue = 42.15 ± 0.1%; H2O MW = 18.015; DMF MW = 73.094; EtOH MW = 46.068; NH2BDC MW = 179.131; ZrO2 MW = 123.222; element counts Zr = 6, C = 48, N = 6.

**`reference_values.json`** — Ground truth values from 14-case validation and collaborator cross-check. Used by Group D tests. Do not modify without scientific justification.

### `examples/`

**`sample_tga.csv`** — Synthetic TGA data (67 points, 25–900°C). Mimics a UiO-66-type curve: guest loss at 30–150°C, decomposition at 350–450°C, residue \~32%. Safe to redistribute (synthetic, no IP).

**`sample_config.json`** — Example JSON config pointing to `sample_tga.csv` with UiO-66 formula.

### `cases/` — Validation data (not in public repo)

14 directories, each containing one TGA CSV file from Crickmore & Bradshaw (DOI: 10.5258/SOTON/D2004). Cases 01–03 and 09: pure UiO-66-NH2 powders from different synthesis batches. Cases 04–08: UiO-66-NH2\@FFP composites at 5/10/15/20 synthesis cycles plus NH3+ variant. Cases 10–11: UiO-66-NH2\@alginate composites. Cases 12–14: Zn-based MOFs with unknown formulas.

### `blank_controls/` — Blank data (not in public repo)

Two directories: FFP blank (residue = −0.79%, treated as 0%) and Ca-alginate blank (residue = 8.89%). Used as reference for composite loading calculations.

### `validation/`

**`cross_check.py`** — Compares results across cases. Checks reproducibility of pure MOF batches (Cases 01 vs 03 vs 09). Checks monotonic increase of FFP loading with synthesis cycles. Checks alginate composite results.

### `output/` — Generated files (in .gitignore)

Contains PNG plots and any generated reports from `run_analysis.py`. Not tracked by git.

***

## Equations

### Composition

```
q = (R_exp_DH × M_residue − M_node) / M_linker
R_exp_DH = m(T_DH) / m_residue
```

### Guest count (single guest)

```
n = (f / (1 − f)) × (M_F / M_G)
f = mass_loss_fraction (0 to 1)
```

### Guest count (multiple guests, forward calculation)

```
f_calc = Σ(n_i × M_i) / (M_F + Σ(n_i × M_i))
```

### Missing linkers

```
x = (M_ideal_DH − R_exp_DH × M_residue) / (M_linker − n_cap × M_cap)
```

### Composite loading

```
w_MOF = (r_composite − r_additive) / (r_MOF − r_additive)
r = residue_mass / initial_mass
```

### Residue prediction

```
predicted_residue_% = (M_oxide_total / M_framework) × 100
```

### Uncertainty (guest count example)

```
σ_n² = (∂n/∂f × σ_f)² + (∂n/∂M_F × σ_MF)² + (∂n/∂M_G × σ_MG)²
```

***

## References

* Abánades Lázaro et al., Eur. J. Inorg. Chem. 2020, 4284–4294
* Shearer et al., Chem. Mater. 2016, 28, 3749–3761
* Sannes et al., Chem. Mater. 2023, 35, 3793–3803
* ASTM E2550: Standard Test Method for Thermal Stability
* ISO 11358: Plastics — Thermogravimetry
* JCGM 100:2008: Guide to the Expression of Uncertainty in Measurement
* Crickmore & Bradshaw, University of Southampton, DOI: 10.5258/SOTON/D2004

***


