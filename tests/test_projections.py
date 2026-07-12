"""
tests/test_projections.py
══════════════════════════════════════════════════════════════════════════════
Integration tests for src/projections.py.

All tests use synthetic data — no Census or RGI files required.
The synthetic census uses UPPERCASE district/state (matching loader.py and
the normalization inside build_projected_population).

Run:
    pytest tests/test_projections.py -v
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.projections import build_projected_population, load_rgi_csv
from src.config import K_ANONYMITY_THRESHOLD, RGI_HIGH_GROWTH_UTS

# ── Fixtures ──────────────────────────────────────────────────────────────────
# NOTE: projections.py uppercases district/state internally, so fixtures can
# use any case — normalization is tested explicitly below.

@pytest.fixture
def census():
    """Five Maharashtra districts. 'SPARSE' is below k-anonymity threshold."""
    return pd.DataFrame({
        "district":   ["MUMBAI SUBURBAN", "PUNE", "NAGPUR", "SOLAPUR", "SPARSE"],
        "state":      ["MAHARASHTRA"] * 5,
        "Population": [12_478_447,        5_057_709, 2_405_665, 951_118, 200],
    })


@pytest.fixture
def rgi_district():
    """District-level RGI rates for three of the five districts."""
    return pd.DataFrame({
        "state":              ["MAHARASHTRA",    "MAHARASHTRA", "MAHARASHTRA"],
        "district":           ["MUMBAI SUBURBAN", "PUNE",        "NAGPUR"],
        "annual_growth_rate": [0.0095,            0.0210,        0.0155],
    })


@pytest.fixture
def rgi_state():
    """State-level only — no district column."""
    return pd.DataFrame({
        "state":              ["MAHARASHTRA"],
        "annual_growth_rate": [0.014],
    })


# ── Tests: output schema ──────────────────────────────────────────────────────

class TestOutputSchema:

    def test_required_columns_present(self, census, rgi_district):
        result = build_projected_population(census, rgi_district)
        required = [
            "district", "state", "Population",
            "annual_growth_rate", "growth_source",
            "Projected_2026", "Projected_2026_lower", "Projected_2026_upper",
            "k_anonymity_flag",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_unchanged(self, census, rgi_district):
        result = build_projected_population(census, rgi_district)
        assert len(result) == len(census)

    def test_no_nan_in_annual_growth_rate(self, census, rgi_district):
        """Every row must have a rate — three-tier fallback guarantees this."""
        result = build_projected_population(census, rgi_district)
        assert result["annual_growth_rate"].isna().sum() == 0

    def test_no_nan_in_growth_source(self, census, rgi_district):
        result = build_projected_population(census, rgi_district)
        assert result["growth_source"].isna().sum() == 0

    def test_no_nan_in_projected_column(self, census, rgi_district):
        result = build_projected_population(census, rgi_district)
        assert result["Projected_2026"].isna().sum() == 0


# ── Tests: projection math ────────────────────────────────────────────────────

class TestProjectionMath:

    def test_projected_gt_2011_for_positive_growth(self, census, rgi_district):
        result = build_projected_population(census, rgi_district)
        large = result[result["Population"] > 1000]
        assert (large["Projected_2026"] > large["Population"]).all()

    def test_uncertainty_bounds_ordering(self, census, rgi_district):
        result = build_projected_population(census, rgi_district)
        assert (result["Projected_2026_lower"] <= result["Projected_2026"]).all()
        assert (result["Projected_2026"] <= result["Projected_2026_upper"]).all()

    def test_compound_growth_formula(self, census):
        """Verify P_2026 = P_2011 × (1 + r)^15 with national fallback."""
        result = build_projected_population(census, rgi_df=None)
        # National fallback rate is 0.012, 15 years
        from src import config
        rate = config.NATIONAL_FALLBACK_RATE
        years = config.TARGET_YEAR - config.BASE_YEAR
        expected = (census["Population"] * (1 + rate) ** years).round().astype(int)
        diff = (result["Projected_2026"] - expected.values).abs()
        assert (diff <= 1).all(), f"Compound growth formula mismatch. Max diff: {diff.max()}"

    def test_uncertainty_band_is_5_percent(self, census, rgi_district):
        result = build_projected_population(census, rgi_district)
        lower_ratio = result["Projected_2026_lower"] / result["Projected_2026"]
        upper_ratio = result["Projected_2026_upper"] / result["Projected_2026"]
        # Allow ±1 for integer rounding
        assert ((lower_ratio - 0.95).abs() < 0.01).all()
        assert ((upper_ratio - 1.05).abs() < 0.01).all()


# ── Tests: three-tier fallback logic ─────────────────────────────────────────

class TestTierFallback:

    def test_tier1_district_rate_used_when_matched(self, census, rgi_district):
        result = build_projected_population(census, rgi_district)
        mumbai = result[result["district"] == "MUMBAI SUBURBAN"].iloc[0]
        assert mumbai["growth_source"] == "district_rgi"
        assert abs(mumbai["annual_growth_rate"] - 0.0095) < 1e-9

    def test_tier2_state_fallback_for_unmatched_district(self, census, rgi_district):
        """SOLAPUR and SPARSE have no district rate → get state mean as fallback."""
        result = build_projected_population(census, rgi_district)
        # State mean = mean(0.0095, 0.021, 0.0155) ≈ 0.015
        state_mean = np.mean([0.0095, 0.021, 0.0155])
        for district in ["SOLAPUR", "SPARSE"]:
            row = result[result["district"] == district].iloc[0]
            assert row["growth_source"] == "state_rgi_fallback", \
                f"{district}: expected 'state_rgi_fallback', got '{row['growth_source']}'"
            assert abs(row["annual_growth_rate"] - state_mean) < 1e-9

    def test_state_level_rgi_triggers_tier2_for_all(self, census, rgi_state):
        """State-level only RGI → all districts get state_rgi_fallback."""
        result = build_projected_population(census, rgi_state)
        assert (result["growth_source"] == "state_rgi_fallback").all()
        assert (result["annual_growth_rate"] == 0.014).all()

    def test_tier3_national_fallback_when_no_rgi(self, census):
        result = build_projected_population(census, rgi_df=None)
        assert (result["growth_source"] == "national_fallback").all()
        from src import config
        assert (result["annual_growth_rate"] == config.NATIONAL_FALLBACK_RATE).all()

    def test_mixed_coverage_counts(self, census, rgi_district):
        """3 districts matched at district level; 2 fall back to state mean."""
        result = build_projected_population(census, rgi_district)
        sources = result["growth_source"].value_counts()
        assert sources.get("district_rgi", 0) == 3
        assert sources.get("state_rgi_fallback", 0) == 2
        assert sources.get("national_fallback", 0) == 0

    def test_case_normalization_input_ignored(self, rgi_district):
        """Input district/state case should not affect join outcome."""
        mixed_case_census = pd.DataFrame({
            "district":   ["Mumbai Suburban", "pune", "NAGPUR", "Solapur", "sparse"],
            "state":      ["Maharashtra", "maharashtra", "MAHARASHTRA", "maharashtra", "MAHARASHTRA"],
            "Population": [12_478_447, 5_057_709, 2_405_665, 951_118, 200],
        })
        result = build_projected_population(mixed_case_census, rgi_district)
        # Mumbai, Pune, Nagpur should still match at district level
        sources = result["growth_source"].value_counts()
        assert sources.get("district_rgi", 0) == 3


# ── Tests: k-anonymity ────────────────────────────────────────────────────────

class TestKAnonymity:

    def test_sparse_district_flagged(self, census, rgi_district):
        result = build_projected_population(census, rgi_district)
        sparse = result[result["district"] == "SPARSE"].iloc[0]
        # Population_2011 = 200, projected also small → below K_ANONYMITY_THRESHOLD
        assert sparse["k_anonymity_flag"] is True or sparse["k_anonymity_flag"] == True

    def test_large_district_not_flagged(self, census, rgi_district):
        result = build_projected_population(census, rgi_district)
        mumbai = result[result["district"] == "MUMBAI SUBURBAN"].iloc[0]
        assert mumbai["k_anonymity_flag"] is False or mumbai["k_anonymity_flag"] == False

    def test_flag_uses_projected_not_2011_population(self):
        """A district just below K_ANONYMITY_THRESHOLD in 2011 but above after
        projection must NOT be flagged."""
        # K_ANONYMITY_THRESHOLD default is 500.
        # A district with Pop=450, growth_rate=0.15 over 15 years → ~454*1.15^15
        # Actually with 0.12 rate: 450 * (1.012)^15 ≈ 539 → above threshold
        borderline_census = pd.DataFrame({
            "district":   ["BORDERLINE"],
            "state":      ["TEST STATE"],
            "Population": [450],
        })
        result = build_projected_population(borderline_census, rgi_df=None)
        # National fallback 0.012 over 15 years: 450 * 1.012^15 ≈ 539 > 500
        assert result.iloc[0]["k_anonymity_flag"] == False


# ── Tests: load_rgi_csv ───────────────────────────────────────────────────────

class TestLoadRgiCsv:

    def test_returns_none_for_missing_file(self, tmp_path):
        result = load_rgi_csv(str(tmp_path / "nonexistent.csv"))
        assert result is None

    def test_loads_state_level_csv(self, tmp_path):
        path = tmp_path / "rgi.csv"
        pd.DataFrame({
            "state": ["Maharashtra"],
            "annual_growth_rate": [0.014],
        }).to_csv(path, index=False)
        result = load_rgi_csv(str(path))
        assert result is not None
        assert "annual_growth_rate" in result.columns
        assert result.iloc[0]["state"] == "MAHARASHTRA"  # uppercased

    def test_loads_district_level_csv(self, tmp_path):
        path = tmp_path / "rgi_district.csv"
        pd.DataFrame({
            "state": ["Maharashtra"],
            "district": ["Pune"],
            "annual_growth_rate": [0.021],
        }).to_csv(path, index=False)
        result = load_rgi_csv(str(path))
        assert "district" in result.columns
        assert result.iloc[0]["district"] == "PUNE"  # uppercased

    def test_raises_on_missing_required_columns(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"state": ["MH"], "growth": [0.014]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="annual_growth_rate"):
            load_rgi_csv(str(path))

    def test_column_headers_case_insensitive(self, tmp_path):
        """CSV with 'State' and 'Annual_Growth_Rate' headers should load correctly."""
        path = tmp_path / "mixed_case_headers.csv"
        pd.DataFrame({
            "State": ["Maharashtra"],
            "Annual_Growth_Rate": [0.014],
        }).to_csv(path, index=False)
        result = load_rgi_csv(str(path))
        assert result is not None
        assert "annual_growth_rate" in result.columns


# ── Tests: RGI_HIGH_GROWTH_UTS (known high-growth UTs, no capping) ──────────

class TestRgiHighGrowthUTs:
    """
    Daman & Diu, Dadra & Nagar Haveli, and Puducherry have genuine RGI
    Mathematical-Method rates outside the normal plausibility bounds
    (migration-driven, RGI 2011-2036 Table 8). Rates are NOT capped;
    validation reclassifies them from "warning" to "expected, see
    config.RGI_HIGH_GROWTH_UTS" instead of silently dropping the signal.
    """

    # -- load_rgi_csv: the raw >5%/yr percentage-vs-decimal sanity check -----

    def test_whitelisted_state_does_not_trigger_extreme_rate_warning(self, tmp_path, caplog):
        path = tmp_path / "rgi_whitelisted.csv"
        pd.DataFrame({
            "state": ["Daman & Diu"],
            "annual_growth_rate": [0.073591],  # published RGI rate, > 5%/yr
        }).to_csv(path, index=False)

        caplog.set_level(logging.INFO, logger="src.projections")
        result = load_rgi_csv(str(path))

        assert result is not None
        assert "> 5%/yr" not in caplog.text
        assert "high-growth UT" in caplog.text
        assert "DAMAN & DIU" in caplog.text

    def test_non_whitelisted_state_still_triggers_extreme_rate_warning(self, tmp_path, caplog):
        path = tmp_path / "rgi_not_whitelisted.csv"
        pd.DataFrame({
            "state": ["Test State"],
            "annual_growth_rate": [0.09],  # implausible for any real state -> should warn
        }).to_csv(path, index=False)

        caplog.set_level(logging.INFO, logger="src.projections")
        result = load_rgi_csv(str(path))

        assert result is not None
        assert "> 5%/yr" in caplog.text
        assert "TEST STATE" in caplog.text

    # -- _validate_projections (via build_projected_population): the -------
    # -- district-level 15-year growth-factor plausibility check -----------

    def test_whitelisted_state_does_not_trigger_growth_factor_warning(self, caplog):
        census = pd.DataFrame({
            "district":   ["DAMAN"],
            "state":      ["DAMAN & DIU"],
            "Population": [191_173],
        })
        rgi = pd.DataFrame({
            "state":              ["DAMAN & DIU"],
            "annual_growth_rate": [0.073591],
        })

        caplog.set_level(logging.INFO, logger="src.projections")
        build_projected_population(census, rgi)

        assert "implausible" not in caplog.text
        assert "high-growth UT" in caplog.text
        assert "DAMAN & DIU" in caplog.text

    def test_non_whitelisted_extreme_state_still_triggers_growth_factor_warning(self, caplog):
        census = pd.DataFrame({
            "district":   ["SOME DISTRICT"],
            "state":      ["TEST STATE"],
            "Population": [100_000],
        })
        rgi = pd.DataFrame({
            "state":              ["TEST STATE"],
            "annual_growth_rate": [0.07],  # not in RGI_HIGH_GROWTH_UTS -> must still warn
        })

        caplog.set_level(logging.INFO, logger="src.projections")
        build_projected_population(census, rgi)

        assert "implausible" in caplog.text
        assert "TEST STATE" in caplog.text

    # -- No capping side effect ----------------------------------------------

    def test_whitelisted_state_projection_uses_published_rate_uncapped(self):
        """Projected_2026 for a whitelisted state must equal the raw compound
        growth formula applied to the published rate -- no clamping to the
        0.85-1.40 plausibility band."""
        census = pd.DataFrame({
            "district":   ["DAMAN"],
            "state":      ["DAMAN & DIU"],
            "Population": [191_173],
        })
        rate = 0.073591
        rgi = pd.DataFrame({
            "state":              ["DAMAN & DIU"],
            "annual_growth_rate": [rate],
        })
        result = build_projected_population(census, rgi)

        from src import config
        years = config.TARGET_YEAR - config.BASE_YEAR
        expected = round(191_173 * (1 + rate) ** years)
        assert result.iloc[0]["Projected_2026"] == expected

        # Sanity check: this value is genuinely outside the plausible bound --
        # confirms the assertion above is exercising the uncapped path, not
        # coincidentally matching a capped value.
        growth_factor = expected / 191_173
        assert growth_factor > 1.40

    def test_whitelist_matches_config_constant(self):
        """Guards against the whitelist silently drifting from config.py."""
        assert RGI_HIGH_GROWTH_UTS == {
            "DAMAN & DIU", "DADRA & NAGAR HAVELI", "PUDUCHERRY",
        }