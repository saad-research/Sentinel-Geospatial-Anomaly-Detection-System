"""
src/projections.py
══════════════════════════════════════════════════════════════════════════════
RGI Population Projection Pipeline — Aadhaar Sentinel V2

DUAL PURPOSE:
  1. METHODOLOGY FIX (Paper B):
       Replaces stale Census 2011 denominator in PNA with RGI-projected 2026
       district populations. Fixes systematic undercount in high-growth cities.

  2. DATASET CONTRIBUTION (Paper A — IndiaID-Bench):
       The 'Projected_2026' column is the novel element that makes IndiaID-Bench
       distinct from the raw UIDAI + Census 2011 data that already exists.

THREE-TIER GROWTH RATE FALLBACK:
  Tier 1 — District-level RGI rate   (most precise; requires district column in CSV)
  Tier 2 — State-level mean rate      (if district not in RGI data)
  Tier 3 — National fallback 1.2%    (last resort; always logs a warning)

  Every district is tagged with 'growth_source' for full audit trail in Paper A.

INPUT CSV (data/raw/RGI_growth_rates.csv):
  Preferred (district-level):
      state,district,annual_growth_rate
      ANDHRA PRADESH,VISAKHAPATNAM,0.0142

  Acceptable (state-level only):
      state,annual_growth_rate
      ANDHRA PRADESH,0.0138

  annual_growth_rate must be a decimal fraction (0.014 = 1.4%/yr, NOT 1.4).

VALIDATION:
  National projected growth 2011→2026 is checked against a plausible 10–25%
  range (India ~17%). Implausible per-district factors trigger warnings.
  Neither check halts the pipeline — they are diagnostic, not blocking.

k-ANONYMITY (Paper A §4):
  Districts with Projected_2026 < K_ANONYMITY_THRESHOLD are flagged via
  k_anonymity_flag = True. These rows should be masked/dropped before
  the IndiaID-Bench Zenodo release. They are NOT dropped here; that decision
  belongs to the export step so the analysis pipeline runs on all data.

REFERENCES:
  RGI (2020). Population Projections for India and States 2011-2036.
  Office of the Registrar General, India.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# ── Validation constants ──────────────────────────────────────────────────────
# Plausible 15-year growth factor range for any Indian district
_MIN_GROWTH_FACTOR = 0.85   # ≈ -1.1% annual
_MAX_GROWTH_FACTOR = 1.40   # ≈ +2.3% annual

# Expected national aggregate growth 2011→2026 (India ~17%)
_NATIONAL_GROWTH_LOW = 10.0   # percent
_NATIONAL_GROWTH_HIGH = 25.0  # percent


# ── Public API ────────────────────────────────────────────────────────────────

def load_rgi_csv(rgi_path: str) -> Optional[pd.DataFrame]:
    """
    Load RGI growth-rate CSV safely.

    Returns None (not an exception) if the file is absent so that the pipeline
    degrades gracefully to the national fallback rate — pipeline.py should not
    crash just because the RGI CSV hasn't been extracted yet.

    Expected columns:
      Required: state, annual_growth_rate
      Optional: district  (enables Tier 1 district-level matching)

    annual_growth_rate must be a decimal fraction.
    Raises ValueError if the file exists but is missing required columns.
    """
    if not os.path.exists(rgi_path):
        logger.warning(
            "RGI growth-rate file not found at '%s'. "
            "Pipeline will use national fallback rate %.4f for all districts. "
            "To fix: extract Table 4/5 from RGI 2011-2036 PDF and save as "
            "[state, district, annual_growth_rate] CSV.",
            rgi_path,
            config.NATIONAL_FALLBACK_RATE,
        )
        return None

    df = pd.read_csv(rgi_path)
    df.columns = df.columns.str.strip().str.lower()

    required = {"state", "annual_growth_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"RGI CSV at '{rgi_path}' is missing required columns: {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    # Normalize text keys to UPPERCASE to match Census/UIDAI conventions
    df["state"] = df["state"].astype(str).str.upper().str.strip()
    if "district" in df.columns:
        df["district"] = df["district"].astype(str).str.upper().str.strip()

    df["annual_growth_rate"] = pd.to_numeric(df["annual_growth_rate"], errors="coerce")

    # Catch common mistake: rates entered as percentages (1.4) not decimals (0.014)
    extreme = df["annual_growth_rate"].abs() > 0.05
    known_high_growth = extreme & df["state"].isin(config.RGI_HIGH_GROWTH_UTS)
    unexplained_extreme = extreme & ~known_high_growth

    if unexplained_extreme.any():
        logger.warning(
            "%d RGI rows have |annual_growth_rate| > 5%%/yr. "
            "Verify values are decimal fractions (0.014), not percentages (1.4). "
            "Affected states: %s",
            int(unexplained_extreme.sum()),
            df.loc[unexplained_extreme, "state"].unique().tolist(),
        )
    if known_high_growth.any():
        logger.info(
            "%d high-growth UTs using published RGI rates outside standard bounds "
            "(expected, see config.RGI_HIGH_GROWTH_UTS): %s",
            df.loc[known_high_growth, "state"].nunique(),
            sorted(df.loc[known_high_growth, "state"].unique().tolist()),
        )

    logger.info("RGI data loaded: %d rows from '%s'", len(df), rgi_path)
    return df


def build_projected_population(
    census_df: pd.DataFrame,
    rgi_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Apply RGI growth rates to Census 2011 district populations to produce
    RGI-projected populations at config.TARGET_YEAR.

    Parameters
    ----------
    census_df : DataFrame
        Must contain: ['district', 'state', config.POPULATION_COLUMN]
        Text columns should be clean strings; this function normalizes internally.
    rgi_df : DataFrame or None
        From load_rgi_csv(). None triggers national-fallback-only mode.

    Returns
    -------
    DataFrame (same rows as census_df) with additional columns:
        annual_growth_rate  — rate used (decimal fraction)
        growth_source       — 'district_rgi' | 'state_rgi_fallback' | 'national_fallback'
        Projected_{YEAR}    — point projection (integer)
        Projected_{YEAR}_lower — lower uncertainty bound (point × 0.95)
        Projected_{YEAR}_upper — upper uncertainty bound (point × 1.05)
        k_anonymity_flag    — True if Projected_{YEAR} < K_ANONYMITY_THRESHOLD

    All original census_df columns are preserved.
    """
    pop_col = config.POPULATION_COLUMN
    years = config.TARGET_YEAR - config.BASE_YEAR

    if pop_col not in census_df.columns:
        raise ValueError(
            f"census_df missing population column '{pop_col}'. "
            f"Available columns: {census_df.columns.tolist()}"
        )

    df = census_df.copy()

    # Normalize text keys to UPPERCASE (matches loader.py and Census conventions)
    df["district"] = df["district"].astype(str).str.upper().str.strip()
    df["state"] = df["state"].astype(str).str.upper().str.strip()

    # Initialize growth rate columns
    df["annual_growth_rate"] = np.nan
    df["growth_source"] = None

    # ── Three-tier growth rate assignment ─────────────────────────────────
    if rgi_df is not None:
        df, coverage = _apply_rgi_rates(df, rgi_df)
    else:
        logger.warning(
            "No RGI data — applying national fallback %.4f to all %d districts.",
            config.NATIONAL_FALLBACK_RATE,
            len(df),
        )
        df["annual_growth_rate"] = config.NATIONAL_FALLBACK_RATE
        df["growth_source"] = "national_fallback"
        coverage = {"district_rgi": 0, "state_rgi_fallback": 0, "national_fallback": len(df)}

    _log_coverage_report(coverage, len(df))

    # ── Compound growth projection ─────────────────────────────────────────
    growth_factor = (1.0 + df["annual_growth_rate"]) ** years

    proj_col = config.PROJECTED_POP_COLUMN
    df[proj_col] = (df[pop_col] * growth_factor).round().astype(int)
    df[config.PROJECTED_POP_LOWER_COLUMN] = (
        df[proj_col] * (1.0 - config.PROJECTION_UNCERTAINTY)
    ).round().astype(int)
    df[config.PROJECTED_POP_UPPER_COLUMN] = (
        df[proj_col] * (1.0 + config.PROJECTION_UNCERTAINTY)
    ).round().astype(int)

    # ── k-Anonymity flag (Paper A §4) ─────────────────────────────────────
    # Flag, do NOT drop. Dropping is the export layer's responsibility.
    df["k_anonymity_flag"] = df[proj_col] < config.K_ANONYMITY_THRESHOLD
    n_flagged = int(df["k_anonymity_flag"].sum())
    if n_flagged > 0:
        logger.info(
            "k-Anonymity: %d districts flagged (Projected_%d < %d). "
            "These should be masked before Zenodo release (Paper A §4).",
            n_flagged,
            config.TARGET_YEAR,
            config.K_ANONYMITY_THRESHOLD,
        )

    # ── Validation ─────────────────────────────────────────────────────────
    _validate_projections(df, proj_col, years)

    return df


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_rgi_rates(
    df: pd.DataFrame, rgi_df: pd.DataFrame
) -> tuple:
    """
    Apply RGI rates via three-tier fallback. Returns (updated_df, coverage_dict).
    Both df and rgi_df must already have UPPERCASE district/state columns.
    """
    rgi = rgi_df.copy()
    tier1_count = 0

    # ── Tier 1: District-level match ──────────────────────────────────────
    if "district" in rgi.columns:
        district_rates = rgi[["state", "district", "annual_growth_rate"]].copy()
        # Merge introduces annual_growth_rate_rgi; original stays as NaN placeholder
        df = df.merge(
            district_rates,
            on=["state", "district"],
            how="left",
            suffixes=("", "_rgi"),
        )
        matched = df["annual_growth_rate_rgi"].notna()
        df.loc[matched, "annual_growth_rate"] = df.loc[matched, "annual_growth_rate_rgi"]
        df.loc[matched, "growth_source"] = "district_rgi"
        df = df.drop(columns=["annual_growth_rate_rgi"])
        tier1_count = int(matched.sum())
        logger.info(
            "Tier 1 (district RGI): %d / %d districts matched",
            tier1_count,
            len(df),
        )

    # ── Tier 2: State-level mean fallback ─────────────────────────────────
    # Compute state means from ALL available district rates in the RGI data
    state_means = (
        rgi.groupby("state", as_index=False)["annual_growth_rate"]
        .mean()
        .rename(columns={"annual_growth_rate": "_state_rate"})
    )
    unmatched = df["annual_growth_rate"].isna()
    tier2_count = 0
    if unmatched.any():
        df = df.merge(state_means, on="state", how="left")
        fill_mask = unmatched & df["_state_rate"].notna()
        df.loc[fill_mask, "annual_growth_rate"] = df.loc[fill_mask, "_state_rate"]
        df.loc[fill_mask, "growth_source"] = "state_rgi_fallback"
        tier2_count = int(fill_mask.sum())
        df = df.drop(columns=["_state_rate"])
        logger.info("Tier 2 (state mean fallback): %d districts filled", tier2_count)

    # ── Tier 3: National fallback ─────────────────────────────────────────
    still_missing = df["annual_growth_rate"].isna()
    tier3_count = int(still_missing.sum())
    if tier3_count > 0:
        df.loc[still_missing, "annual_growth_rate"] = config.NATIONAL_FALLBACK_RATE
        df.loc[still_missing, "growth_source"] = "national_fallback"
        logger.warning(
            "Tier 3 (national fallback %.4f): %d districts had no RGI match. "
            "Common cause: spelling differences between Census and RGI district names. "
            "Affected states: %s",
            config.NATIONAL_FALLBACK_RATE,
            tier3_count,
            df.loc[still_missing, "state"].unique().tolist(),
        )

    coverage = {
        "district_rgi": tier1_count,
        "state_rgi_fallback": tier2_count,
        "national_fallback": tier3_count,
    }
    return df, coverage


def _validate_projections(df: pd.DataFrame, proj_col: str, years: int) -> None:
    """
    Sanity checks on projected outputs. Logs warnings; never raises.
    Provides the numbers needed for Paper B §6 (Limitations) and Paper A §4.
    """
    # Check for non-positive populations
    bad = df[df[proj_col] <= 0]
    if len(bad):
        logger.warning(
            "VALIDATION: %d districts have projected population ≤ 0. "
            "States affected: %s",
            len(bad),
            bad["state"].unique().tolist(),
        )

    # Check growth factors are in a plausible range for Indian districts
    growth_factor = df[proj_col] / df[config.POPULATION_COLUMN]
    out_of_range = df[(growth_factor < _MIN_GROWTH_FACTOR) | (growth_factor > _MAX_GROWTH_FACTOR)]

    is_known_high_growth = out_of_range["state"].isin(config.RGI_HIGH_GROWTH_UTS)
    known_high_growth = out_of_range[is_known_high_growth]
    unexplained = out_of_range[~is_known_high_growth]

    if len(unexplained):
        logger.warning(
            "VALIDATION: %d districts have implausible %d-year growth factor "
            "(outside %.2f–%.2f). Top cases:\n%s",
            len(unexplained),
            years,
            _MIN_GROWTH_FACTOR,
            _MAX_GROWTH_FACTOR,
            unexplained[["district", "state", proj_col]].head(5).to_string(index=False),
        )
    if len(known_high_growth):
        logger.info(
            "%d high-growth UTs using published RGI rates outside standard bounds "
            "(expected, see config.RGI_HIGH_GROWTH_UTS): %s",
            known_high_growth["state"].nunique(),
            sorted(known_high_growth["state"].unique().tolist()),
        )

    # National aggregate growth check
    total_2011 = df[config.POPULATION_COLUMN].sum()
    total_proj = df[proj_col].sum()
    national_pct = ((total_proj - total_2011) / total_2011) * 100
    target = int(proj_col.split("_")[1])

    logger.info(
        "VALIDATION: National population 2011 → %d: %s → %s (Δ = +%.1f%%)",
        target,
        f"{total_2011:,.0f}",
        f"{total_proj:,.0f}",
        national_pct,
    )

    if not (_NATIONAL_GROWTH_LOW <= national_pct <= _NATIONAL_GROWTH_HIGH):
        logger.warning(
            "VALIDATION: National growth %.1f%% is outside expected band [%.0f%%, %.0f%%]. "
            "Check RGI rates for percentage-vs-decimal errors (1.4 vs 0.014).",
            national_pct,
            _NATIONAL_GROWTH_LOW,
            _NATIONAL_GROWTH_HIGH,
        )


def _log_coverage_report(coverage: dict, total: int) -> None:
    d = coverage.get("district_rgi", 0)
    s = coverage.get("state_rgi_fallback", 0)
    n = coverage.get("national_fallback", 0)
    logger.info(
        "Growth-rate coverage (%d districts total): "
        "district_rgi=%d (%.1f%%) | state_fallback=%d (%.1f%%) | "
        "national_fallback=%d (%.1f%%)",
        total,
        d, (d / total * 100) if total else 0,
        s, (s / total * 100) if total else 0,
        n, (n / total * 100) if total else 0,
    )