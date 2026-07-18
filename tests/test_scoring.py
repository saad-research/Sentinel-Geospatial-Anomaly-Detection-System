"""
tests/test_scoring.py
══════════════════════════════════════════════════════════════════════════════
Tests for src/scoring.py Method 3 (LOF) and Stage 2a (HDBSCAN) -- V2.2
additions previously untested (see CLAUDE.md "Known open gaps" #3).
"""

import numpy as np
import pandas as pd
import pytest

from src.scoring import cluster_anomalies_hdbscan, flag_anomalies_lof


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_lof_outlier_df(n_normal: int = 40) -> pd.DataFrame:
    """
    Normal cluster + a handful of extreme points scattered apart from each
    other (not bunched into their own tight group).

    LOF flags points whose *local* neighbourhood density is low relative to
    their neighbours'. A tight little group of extreme values looks locally
    "normal" to LOF -- each point's nearest neighbours are just as extreme,
    so the density ratio is ~1 (this is exactly the IF-vs-LOF complementarity
    documented in scoring.py's build_ensemble_outliers). To reliably trigger
    lof_flag, the outliers here are mutually distant as well as distant from
    the normal cluster.
    """
    rng = np.random.default_rng(seed=42)

    normal = pd.DataFrame({
        "pincode":        [f"1{i:05d}" for i in range(n_normal)],
        "TAI":            rng.integers(200, 1000, n_normal).tolist(),
        "DPR":            rng.uniform(0.8, 3.0, n_normal).tolist(),
        "PNA":            rng.uniform(0.02, 0.3, n_normal).tolist(),
        "Privacy_Masked": [False] * n_normal,
    })
    scattered_outliers = pd.DataFrame({
        "pincode":        ["900000", "900001", "900002", "900003", "900004"],
        "TAI":            [50_000, 30, 20_000, 15, 60_000],
        "DPR":            [500.0, 0.01, 350.0, 0.001, 900.0],
        "PNA":            [15.0, 0.0001, 9.0, 0.00005, 20.0],
        "Privacy_Masked": [False] * 5,
    })
    return pd.concat([normal, scattered_outliers], ignore_index=True)


def make_two_cluster_df() -> pd.DataFrame:
    """
    Two well-separated dense groups (>= min_cluster_size each) plus a few
    isolated noise points.

    HDBSCAN's default allow_single_cluster=False means the root cluster is
    never itself selected -- a single homogeneous blob, however tight, is
    left entirely as noise (verified directly against sklearn.cluster.HDBSCAN
    before writing this fixture). At least two branches of substructure are
    required for any non-root cluster to be selected.
    """
    rng = np.random.default_rng(0)
    n = 8
    cluster_a = pd.DataFrame({
        "pincode": [f"A{i}" for i in range(n)],
        "DPR": rng.uniform(190, 195, n).tolist(),
        "PNA": rng.uniform(5.95, 6.05, n).tolist(),
        "TAI": rng.uniform(10_400, 10_600, n).tolist(),
    })
    cluster_b = pd.DataFrame({
        "pincode": [f"C{i}" for i in range(n)],
        "DPR": rng.uniform(5, 8, n).tolist(),
        "PNA": rng.uniform(0.05, 0.15, n).tolist(),
        "TAI": rng.uniform(300, 400, n).tolist(),
    })
    isolated = pd.DataFrame({
        "pincode": [f"B{i}" for i in range(3)],
        "DPR":   [50.0, 300.0, 1.5],
        "PNA":   [3.0, 0.01, 8.0],
        "TAI":   [5000, 200, 15000],
    })
    return pd.concat([cluster_a, cluster_b, isolated], ignore_index=True)


# ── Method 3: flag_anomalies_lof ─────────────────────────────────────────────

class TestFlagAnomaliesLof:

    def test_output_columns_present(self):
        df = flag_anomalies_lof(make_lof_outlier_df())
        assert "lof_flag" in df.columns
        assert "lof_score" in df.columns

    def test_row_count_unchanged(self):
        df_in = make_lof_outlier_df()
        df_out = flag_anomalies_lof(df_in)
        assert len(df_out) == len(df_in)

    def test_lof_flag_is_boolean(self):
        df = flag_anomalies_lof(make_lof_outlier_df())
        assert df["lof_flag"].dtype == bool

    def test_scattered_extreme_point_flagged(self):
        """A point extreme AND isolated from every other point (including the
        other extreme points) should register as a locally low-density
        anomaly."""
        df = flag_anomalies_lof(make_lof_outlier_df())
        assert df["lof_flag"].any(), "Expected at least one scattered outlier flagged"

    def test_flagged_rows_have_negative_score(self):
        """lof_score mirrors iso_score's sign convention: negative = anomaly."""
        df = flag_anomalies_lof(make_lof_outlier_df())
        flagged = df[df["lof_flag"]]
        assert len(flagged) > 0
        assert (flagged["lof_score"] < 0).all()

    def test_custom_n_neighbors_accepted(self):
        df_in = make_lof_outlier_df()
        df_out = flag_anomalies_lof(df_in, n_neighbors=5)
        assert "lof_flag" in df_out.columns
        assert len(df_out) == len(df_in)

    def test_privacy_masked_rows_not_in_fit_but_scored(self):
        """Mirrors flag_anomalies_isolation_forest's masking contract: masked
        rows are excluded from lof.fit() but still receive a score via
        predict/decision_function on the full frame."""
        df = make_lof_outlier_df()
        df.loc[df.index[:5], "Privacy_Masked"] = True
        result = flag_anomalies_lof(df)
        masked_scores = result.loc[result["Privacy_Masked"], "lof_score"]
        assert masked_scores.notna().all()


# ── Stage 2a: cluster_anomalies_hdbscan ──────────────────────────────────────

class TestClusterAnomaliesHdbscan:

    def test_output_columns_present(self):
        df = cluster_anomalies_hdbscan(make_two_cluster_df())
        assert "hdbscan_cluster" in df.columns
        assert "hdbscan_is_clustered" in df.columns
        assert "hdbscan_probability" in df.columns

    def test_row_count_preserved(self):
        df_in = make_two_cluster_df()
        df_out = cluster_anomalies_hdbscan(df_in)
        assert len(df_out) == len(df_in)

    def test_cluster_labels_are_int_or_minus_one(self):
        df = cluster_anomalies_hdbscan(make_two_cluster_df())
        assert df["hdbscan_cluster"].dtype in [np.int32, np.int64, int]
        assert (df["hdbscan_cluster"] >= -1).all()

    def test_is_clustered_consistent_with_label(self):
        df = cluster_anomalies_hdbscan(make_two_cluster_df())
        assert (df["hdbscan_is_clustered"] == (df["hdbscan_cluster"] >= 0)).all()

    def test_probability_within_valid_range(self):
        df = cluster_anomalies_hdbscan(make_two_cluster_df())
        assert df["hdbscan_probability"].between(0.0, 1.0).all()

    def test_noise_points_have_zero_probability(self):
        df = cluster_anomalies_hdbscan(make_two_cluster_df())
        noise = df[df["hdbscan_cluster"] == -1]
        assert len(noise) > 0
        assert (noise["hdbscan_probability"] == 0.0).all()

    def test_two_separated_groups_form_clusters(self):
        """Two dense, well-separated groups should each be selected as a
        stable cluster, with the isolated singleton points left as noise."""
        df = cluster_anomalies_hdbscan(make_two_cluster_df())
        assert df["hdbscan_is_clustered"].any(), "Expected at least one stable cluster"
        n_clusters = len(set(df["hdbscan_cluster"].unique()) - {-1})
        assert n_clusters >= 2

    def test_too_few_rows_returns_all_noise(self):
        """Fewer rows than min_cluster_size -> all points are noise."""
        tiny_df = make_two_cluster_df().head(2)
        result = cluster_anomalies_hdbscan(tiny_df, min_cluster_size=5)
        assert (result["hdbscan_cluster"] == -1).all()
        assert result["hdbscan_is_clustered"].sum() == 0
        assert (result["hdbscan_probability"] == 0.0).all()

    def test_custom_min_cluster_size_accepted(self):
        df_in = make_two_cluster_df()
        df_out = cluster_anomalies_hdbscan(df_in, min_cluster_size=3, min_samples=2)
        assert len(df_out) == len(df_in)
        assert "hdbscan_cluster" in df_out.columns
