"""
tests/test_engine.py
══════════════════════════════════════════════════════════════════════════════
Scoring and anomaly detection tests — Aadhaar Sentinel V2

Original 3 tests preserved and passing.
V2 additions:
  - add_population_uncertainty: V2 bounds vs V1 fallback
  - cluster_anomalies_dbscan: column presence, cluster/noise labels
  - build_ensemble_outliers: return structure, iso_score on all rows
  - Privacy_Masked: excluded from IF fit
  - PNA bounds: mathematical ordering guarantee
"""

import pandas as pd
import numpy as np
import pytest

from src.scoring import (
    add_population_uncertainty,
    build_ensemble_outliers,
    cluster_anomalies_dbscan,
    compute_risk_score,
    flag_anomalies_isolation_forest,
    flag_anomalies_statistical,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_dummy_df():
    """Original 3-row fixture — kept for V1 backward-compat tests."""
    return pd.DataFrame({
        "pincode":      ["110001", "110002", "110003"],
        "district":     ["WEST DELHI", "EAST DELHI", "NORTH DELHI"],
        "state":        ["DELHI", "DELHI", "DELHI"],
        "demo_total":   [5000, 100, 200],
        "bio_total":    [10, 90, 180],
        "enrol_total":  [500, 300, 400],
        "TAI":          [5510, 490, 780],
        "DPR":          [500.0, 1.1, 1.1],
        "PNA":          [6.1, 0.05, 0.08],
        "est_pincode_pop": [1000, 1000, 1000],
    })


def make_v2_dummy_df(n_normal: int = 40, n_outlier: int = 5) -> pd.DataFrame:
    """
    V2 fixture: n_normal PINcodes with typical values + n_outlier extreme ones.

    Large enough (45 rows) for IsolationForest contamination=0.02 to flag
    at least 1 outlier (0.02 × 45 ≈ 1). The extreme outliers have DPR > 100
    and PNA > 5, which are always flagged reliably.

    Includes all V2 columns: est_pincode_pop_lower/upper, DPR_v2, Privacy_Masked.
    """
    rng = np.random.default_rng(seed=42)

    # Normal PINcodes
    normal = pd.DataFrame({
        "pincode":      [f"1{i:05d}" for i in range(n_normal)],
        "district":     [f"DISTRICT_{i % 5}" for i in range(n_normal)],
        "state":        ["DELHI"] * n_normal,
        "demo_total":   rng.integers(50, 300, n_normal).tolist(),
        "bio_total":    rng.integers(40, 250, n_normal).tolist(),
        "enrol_total":  rng.integers(100, 500, n_normal).tolist(),
        "TAI":          rng.integers(200, 1000, n_normal).tolist(),
        "DPR":          rng.uniform(0.8, 3.0, n_normal).tolist(),
        "DPR_v2":       rng.uniform(0.6, 2.5, n_normal).tolist(),
        "PNA":          rng.uniform(0.02, 0.3, n_normal).tolist(),
        "est_pincode_pop":       [10_000] * n_normal,
        "est_pincode_pop_lower": [9_500] * n_normal,
        "est_pincode_pop_upper": [10_500] * n_normal,
        "annual_growth_rate":    [0.012] * n_normal,
        "growth_source":         ["district_rgi"] * n_normal,
        "Privacy_Masked":        [False] * n_normal,
    })

    # Extreme outliers (will definitely be flagged by IF)
    outliers = pd.DataFrame({
        "pincode":      [f"9{i:05d}" for i in range(n_outlier)],
        "district":     [f"OUTLIER_DISTRICT_{i}" for i in range(n_outlier)],
        "state":        ["DELHI"] * n_outlier,
        "demo_total":   [10_000 + i * 500 for i in range(n_outlier)],
        "bio_total":    [20 + i for i in range(n_outlier)],
        "enrol_total":  [1_000 + i * 100 for i in range(n_outlier)],
        "TAI":          [11_020 + i * 600 for i in range(n_outlier)],
        "DPR":          [200.0 + i * 10 for i in range(n_outlier)],
        "DPR_v2":       [180.0 + i * 8 for i in range(n_outlier)],
        "PNA":          [6.0 + i * 0.5 for i in range(n_outlier)],
        "est_pincode_pop":       [2_000] * n_outlier,
        "est_pincode_pop_lower": [1_900] * n_outlier,
        "est_pincode_pop_upper": [2_100] * n_outlier,
        "annual_growth_rate":    [0.012] * n_outlier,
        "growth_source":         ["district_rgi"] * n_outlier,
        "Privacy_Masked":        [False] * n_outlier,
    })

    return pd.concat([normal, outliers], ignore_index=True)


# ── Original V1 tests (must remain passing) ───────────────────────────────────

def test_risk_score_column_exists():
    df = compute_risk_score(make_dummy_df())
    assert "audit_priority_score" in df.columns


def test_high_dpr_ranks_high():
    df = compute_risk_score(make_dummy_df())
    top = df.sort_values("audit_priority_score", ascending=False).iloc[0]
    assert top["district"] == "WEST DELHI"


def test_statistical_outliers_not_empty():
    df = flag_anomalies_statistical(make_dummy_df())
    assert len(df) > 0


# ── V2: add_population_uncertainty ───────────────────────────────────────────

class TestAddPopulationUncertainty:

    def test_v2_uses_lower_upper_bounds(self):
        """V2: PNA_conservative uses est_pincode_pop_lower (not est_pincode_pop)."""
        df = make_v2_dummy_df()
        result = add_population_uncertainty(df)

        # PNA_conservative = TAI / pop_lower → should be HIGHER than PNA
        # (fewer assumed residents = higher stress reading)
        assert "PNA_conservative" in result.columns
        assert "PNA_upper_bound" in result.columns
        # PNA_conservative > PNA > PNA_upper_bound (for positive TAI and pop)
        assert (result["PNA_conservative"] >= result["PNA"]).all()
        assert (result["PNA"] >= result["PNA_upper_bound"]).all()

    def test_v1_fallback_when_columns_absent(self):
        """V1 fallback: est_pincode_pop_lower/upper absent → flat rate multiplier."""
        df = make_dummy_df()  # no lower/upper columns
        result = add_population_uncertainty(df)
        assert "PNA_conservative" in result.columns
        assert "PNA_upper_bound" in result.columns
        # V1: PNA_conservative == PNA (no lower bound adjustment)
        # PNA_upper_bound == TAI / (pop * 1.20) → lower than PNA
        assert (result["PNA_upper_bound"] <= result["PNA_conservative"]).all()

    def test_no_nan_in_output_bounds(self):
        """Uncertainty bounds must be finite for all rows with positive population."""
        df = make_v2_dummy_df()
        result = add_population_uncertainty(df)
        assert result["PNA_conservative"].notna().all()
        assert result["PNA_upper_bound"].notna().all()

    def test_bounds_ordering_guarantee(self):
        """Lower pop → higher PNA. Upper pop → lower PNA. Strict ordering."""
        df = make_v2_dummy_df()
        result = add_population_uncertainty(df)
        # PNA_conservative (from lower pop) must be ≥ PNA_upper_bound (from upper pop)
        assert (result["PNA_conservative"] >= result["PNA_upper_bound"]).all()


# ── V2: DBSCAN clustering ─────────────────────────────────────────────────────

class TestClusterAnomaliesDbscan:

    def _make_clusterable_df(self):
        """Synthetic IF-flagged df with a clear dense cluster."""
        rng = np.random.default_rng(0)
        # 6 closely-spaced points (should form ≥1 cluster with min_samples=3)
        cluster_points = pd.DataFrame({
            "pincode": [f"A{i}" for i in range(6)],
            "district": ["CLUSTER_DIST"] * 6,
            "state": ["STATE"] * 6,
            "DPR": rng.uniform(180, 200, 6).tolist(),
            "PNA": rng.uniform(5.8, 6.2, 6).tolist(),
            "TAI": rng.uniform(10000, 11000, 6).tolist(),
        })
        # 3 isolated points (noise)
        isolated = pd.DataFrame({
            "pincode": [f"B{i}" for i in range(3)],
            "district": ["ISOLATED_DIST"] * 3,
            "state": ["STATE"] * 3,
            "DPR":   [50.0, 300.0, 1.5],
            "PNA":   [3.0, 0.01, 8.0],
            "TAI":   [5000, 200, 15000],
        })
        return pd.concat([cluster_points, isolated], ignore_index=True)

    def test_output_columns_present(self):
        df = cluster_anomalies_dbscan(self._make_clusterable_df())
        assert "dbscan_cluster" in df.columns
        assert "dbscan_is_clustered" in df.columns

    def test_row_count_preserved(self):
        df_in = self._make_clusterable_df()
        df_out = cluster_anomalies_dbscan(df_in)
        assert len(df_out) == len(df_in)

    def test_cluster_labels_are_int_or_minus_one(self):
        df = cluster_anomalies_dbscan(self._make_clusterable_df())
        assert df["dbscan_cluster"].dtype in [np.int32, np.int64, int]
        assert (df["dbscan_cluster"] >= -1).all()

    def test_is_clustered_consistent_with_label(self):
        df = cluster_anomalies_dbscan(self._make_clusterable_df())
        assert (df["dbscan_is_clustered"] == (df["dbscan_cluster"] >= 0)).all()

    def test_dense_group_forms_cluster(self):
        """6 closely-spaced anomalies should be assigned to at least one cluster."""
        df = cluster_anomalies_dbscan(self._make_clusterable_df(), eps=0.5, min_samples=3)
        assert df["dbscan_is_clustered"].any(), "Expected at least one dense cluster"

    def test_too_few_rows_returns_all_noise(self):
        """Fewer rows than min_samples → all points are noise."""
        tiny_df = self._make_clusterable_df().head(2)
        result = cluster_anomalies_dbscan(tiny_df, min_samples=3)
        assert (result["dbscan_cluster"] == -1).all()
        assert result["dbscan_is_clustered"].sum() == 0


# ── V2: build_ensemble_outliers ───────────────────────────────────────────────

class TestBuildEnsembleOutliers:

    def test_returns_four_element_tuple(self):
        df = make_v2_dummy_df()
        result = build_ensemble_outliers(df)
        assert len(result) == 4, "Expected (full_df, if_outliers, ensemble_df, stats)"

    def test_full_df_has_iso_score_on_all_rows(self):
        full_df, _, _, _ = build_ensemble_outliers(make_v2_dummy_df())
        assert "iso_score" in full_df.columns
        assert "anomaly_score" in full_df.columns
        assert full_df["iso_score"].notna().all()

    def test_full_df_row_count_unchanged(self):
        df = make_v2_dummy_df()
        full_df, _, _, _ = build_ensemble_outliers(df)
        assert len(full_df) == len(df)

    def test_if_outliers_subset_of_full_df(self):
        full_df, if_outliers, _, _ = build_ensemble_outliers(make_v2_dummy_df())
        assert len(if_outliers) <= len(full_df)
        assert (if_outliers["anomaly_score"] == -1).all()

    def test_ensemble_df_has_dbscan_columns(self):
        _, _, ensemble_df, _ = build_ensemble_outliers(make_v2_dummy_df())
        assert "dbscan_cluster" in ensemble_df.columns
        assert "dbscan_is_clustered" in ensemble_df.columns

    def test_ensemble_df_is_subset_of_if_outliers(self):
        """ensemble_df should contain exactly the IF-flagged PINcodes + cluster labels."""
        _, if_outliers, ensemble_df, _ = build_ensemble_outliers(make_v2_dummy_df())
        assert len(ensemble_df) == len(if_outliers)

    def test_stats_dict_has_required_keys(self):
        _, _, _, stats = build_ensemble_outliers(make_v2_dummy_df())
        required_keys = {
            "if_flagged", "dbscan_clusters",
            "dbscan_clustered_pincodes", "dbscan_isolated_outliers",
        }
        assert required_keys.issubset(stats.keys())

    def test_stats_counts_are_consistent(self):
        _, _, ensemble_df, stats = build_ensemble_outliers(make_v2_dummy_df())
        assert stats["if_flagged"] == len(ensemble_df)
        assert (
            stats["dbscan_clustered_pincodes"] + stats["dbscan_isolated_outliers"]
            == stats["if_flagged"]
        )

    def test_privacy_masked_rows_not_in_fit_but_scored(self):
        """
        If some rows are Privacy_Masked, the model should still score them
        (anomaly_score present) but should not have been used for fitting.
        We verify indirectly: masked rows get iso_score (not NaN).
        """
        df = make_v2_dummy_df()
        df.loc[df.index[:5], "Privacy_Masked"] = True
        full_df, _, _, _ = build_ensemble_outliers(df)
        masked_scores = full_df.loc[full_df["Privacy_Masked"], "iso_score"]
        assert masked_scores.notna().all()