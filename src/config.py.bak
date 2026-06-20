# src/config.py
# All tunable parameters in one place.
# Change thresholds here, nowhere else.

OUTLIER_PERCENTILE = 0.98

# Scoring weights — justified by audit priority:
# Integrity anomalies (DPR) weighted highest because they indicate
# process non-compliance, which is harder to detect and higher risk.
WEIGHT_DPR = 0.50
WEIGHT_PNA = 0.35
WEIGHT_ACTIVITY = 0.15

# Census fallback: estimated population for pincodes with no district match
CENSUS_FALLBACK_POP = 30_000

# Population uncertainty band: Census 2011 data, adjusted for ~15 years growth
POP_GROWTH_FACTOR_MIN = 1.0   # Conservative (no growth assumed)
POP_GROWTH_FACTOR_MAX = 1.20  # Upper bound (20% urban growth estimate)

# Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "outputs/maps"

CENSUS_PATH = "data/raw/Census_2011.csv"
POPULATION_COLUMN = "Population"

LATLON_PATH = "data/raw/india_district_latlon.csv"

# ───────────────────────────────────────────────────── ─────────────────────────────────────────────────────
# V2 ADDITIONS — paste these at the BOTTOM of your existing src/config.py
# Do not delete or modify any existing constants.


# ── RGI Population Projection Pipeline ──
# Preferred CSV schema (district-level, most precise):
#     state,district,annual_growth_rate
#     ANDHRA PRADESH,VISAKHAPATNAM,0.0142
#
# Acceptable schema (state-level only, triggers Tier 2 fallback):
#     state,annual_growth_rate
#     ANDHRA PRADESH,0.0138
#
# See docs/rgi_extraction_guide.md for PDF extraction instructions.
RGI_PATH = "data/raw/RGI_growth_rates.csv"

# ── Projection parameters ──
BASE_YEAR = 2011
TARGET_YEAR = 2026

# Derived column names — computed from TARGET_YEAR so they stay consistent
# if you later project to 2031.
PROJECTED_POP_COLUMN = f"Projected_{TARGET_YEAR}"          # "Projected_2026"
PROJECTED_POP_LOWER_COLUMN = f"Projected_{TARGET_YEAR}_lower"
PROJECTED_POP_UPPER_COLUMN = f"Projected_{TARGET_YEAR}_upper"

# Uncertainty band applied to the point projection (±5%).
# Represents model uncertainty from state→district rate disaggregation.
# Documented in Paper B §3.2 and Paper A §4.
PROJECTION_UNCERTAINTY = 0.05

# National fallback annual growth rate.
# Used when neither district nor state RGI rates can be matched.
# Source: UN World Population Prospects 2022 + RGI 2011-2036 trend.
# India's mean annual growth rate 2011-2026 ≈ 1.1-1.3%.
NATIONAL_FALLBACK_RATE = 0.012

# ── k-Anonymity privacy control (Paper A §4) ──
# Districts where Projected_2026 < this threshold are flagged.
# At PINCODE level (district_pop / n_pincodes), actual populations are
# even lower — sparse rural PINcodes risk geographic re-identification.
# Flagged rows are masked before the IndiaID-Bench Zenodo release.
K_ANONYMITY_THRESHOLD = 500