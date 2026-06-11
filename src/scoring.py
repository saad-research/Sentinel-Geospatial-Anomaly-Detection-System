"""
src/scoring.py
══════════════════════════════════════════════════════════════════════════════
Anomaly Detection & Risk Scoring — Aadhaar Sentinel V2

V2 CHANGES:
  1. add_population_uncertainty — FIXED double-counting bug.
       V1 multiplied projected pop by POP_GROWTH_FACTOR_MAX again.
       V2 uses est_pincode_pop_lower/upper from RGI projection bounds.
       V1 flat-rate fallback retained when V2 columns are absent.

  2. flag_anomalies_isolation_forest — FIXED Privacy_Masked exclusion.
       V1 fit the model on all rows, including sparse rural PINcodes where
       est_pincode_pop is a rough proxy. These can have extreme PNA values
       that are artefacts of the proxy, not real anomalies, and distort the
       model boundary. V2 fits on unmasked rows, scores all rows.

  3. NEW: build_ensemble_outliers — hierarchical IF → DBSCAN pipeline.
       Paper B §4: Stage 1 IsolationForest finds global outliers; Stage 2
       DBSCAN on the flagged subset identifies coordinated cluster patterns.
       Adds iso_score to ALL rows in the main dataframe (for dashboard and
       for IndiaID-Bench feature completeness).
       Returns (full_df_with_scores, if_outliers, ensemble_df, stats).

  4. NEW: cluster_anomalies_dbscan — DBSCAN on standardized (DPR, PNA, TAI).
       Clustered PINcodes (dbscan_cluster ≥ 0) represent correlated anomaly
       neighbourhoods; isolated outliers (dbscan_cluster = -1) are statistically
       extreme but uncoordinated. The distinction matters for audit prioritization.

  5. flag_anomalies_isolation_forest kept for backward-compatibility.
       For new code, use build_ensemble_outliers.

PAPER B §4 FRAMING:
  "We implement a two-stage hierarchical ensemble. Stage 1 applies
  IsolationForest to the joint (TAI, DPR, PNA) feature space to identify
  global multivariate outliers (contamination = 0.02, mirroring the 98th
  percentile baseline). Stage 2 applies DBSCAN (ε = 0.5, min_samples = 3)
  in standardized feature space to the Stage 1 output, distinguishing
  coordinated multi-PINCODE anomaly clusters from isolated outliers."
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import (
    OUTLIER_PERCENTILE,
    WEIGHT_ACTIVITY,
    WEIGHT_DPR,
    WEIGHT_PNA,
)

logger = logging.getLogger(__name__)

# ── DBSCAN parameters ─────────────────────────────────────────────────────────
# Documented here for Paper B §4 reproducibility. Move to config.py if you want
# to run a sensitivity sweep across these values.
_DBSCAN_EPS = 0.5           # Neighbourhood radius in standardized feature space
_DBSCAN_MIN_SAMPLES = 3     # Minimum PINcodes to constitute a cluster
_DBSCAN_FEATURES = ["DPR", "PNA", "TAI"]


# ── Risk scoring ──────────────────────────────────────────────────────────────

def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalized composite risk score.

    audit_priority_score = (dpr_z × 0.50) + (pna_z × 0.35) + (activity_z × 0.15)

    Weights are in config.py and justified by audit logic:
      - DPR (0.50): process non-compliance — hardest to detect, highest urgency
      - PNA (0.35): infrastructure stress — requires operational response
      - TAI (0.15): overall load context

    Score is shifted to ≥ 0 range for dashboard readability.
    Z-scores are computed on the full dataset (not per-state) so that the
    composite is nationally comparable — important for Paper B Table 2.
    """
    df = df.copy()
    df["dpr_z"] = zscore(df["DPR"].fillna(0))
    df["pna_z"] = zscore(df["PNA"].fillna(0))
    df["activity_z"] = zscore(df["TAI"].fillna(0))

    df["audit_priority_score"] = (
        df["dpr_z"] * WEIGHT_DPR
        + df["pna_z"] * WEIGHT_PNA
        + df["activity_z"] * WEIGHT_ACTIVITY
    )
    df["audit_priority_score"] -= df["audit_priority_score"].min()
    return df


def add_population_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PNA uncertainty bounds.

    V2 (when est_pincode_pop_lower/upper are present — RGI projection bounds):
      PNA_conservative = TAI / est_pincode_pop_lower  (fewer assumed residents
                                                        → higher stress reading)
      PNA_upper_bound  = TAI / est_pincode_pop_upper  (more assumed residents
                                                        → lower stress reading)

    V1 fallback (when V2 columns absent — Census 2011 era):
      PNA_conservative = TAI / est_pincode_pop         (same as PNA, no adjustment)
      PNA_upper_bound  = TAI / (est_pincode_pop × 1.20) (assumes 20% urban growth)

    V1 fallback is logged as a warning so the pipeline doesn't silently use
    stale uncertainty logic after V2 migration.
    """
    df = df.copy()

    if "est_pincode_pop_lower" in df.columns and "est_pincode_pop_upper" in df.columns:
        # V2: divide by the RGI projection bounds
        df["PNA_conservative"] = df["TAI"] / df["est_pincode_pop_lower"].replace(0, np.nan)
        df["PNA_upper_bound"] = df["TAI"] / df["est_pincode_pop_upper"].replace(0, np.nan)
    else:
        # V1 fallback — still correct for Census 2011 era, but logs intent
        from src.config import POP_GROWTH_FACTOR_MAX
        logger.warning(
            "add_population_uncertainty: est_pincode_pop_lower/upper not found. "
            "Using V1 flat-rate multiplier (POP_GROWTH_FACTOR_MAX=%.2f). "
            "Run V2 pipeline to use RGI projection bounds.",
            POP_GROWTH_FACTOR_MAX,
        )
        df["PNA_conservative"] = df["TAI"] / df["est_pincode_pop"]
        df["PNA_upper_bound"] = df["TAI"] / (df["est_pincode_pop"] * POP_GROWTH_FACTOR_MAX)

    return df


# ── Statistical baseline ──────────────────────────────────────────────────────

def flag_anomalies_statistical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile-based baseline. Flags PINcodes in the top OUTLIER_PERCENTILE
    on ANY of TAI, DPR, or PNA (union).

    Kept as Paper B comparison baseline. The 97% overlap between this method
    and IsolationForest is the key methodological validation result.
    """
    tai_t = df["TAI"].quantile(OUTLIER_PERCENTILE)
    dpr_t = df["DPR"].quantile(OUTLIER_PERCENTILE)
    pna_t = df["PNA"].quantile(OUTLIER_PERCENTILE)

    mask = (df["TAI"] >= tai_t) | (df["DPR"] >= dpr_t) | (df["PNA"] >= pna_t)
    outliers = df[mask].copy()
    outliers["detection_method"] = "statistical_percentile"
    return outliers


# ── ML anomaly detection ──────────────────────────────────────────────────────

def flag_anomalies_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standalone IsolationForest anomaly detection.
    Preserved for backward-compatibility with existing pipeline.py calls.

    V2 NOTE: For the full hierarchical ensemble (IF + DBSCAN) and to get
    iso_score on ALL rows (not just flagged), use build_ensemble_outliers().

    V2 FIX: Fits on non-Privacy_Masked rows only to prevent sparse rural
    PINcodes with proxy-distorted PNA from anchoring the anomaly boundary.
    All rows are scored regardless.
    """
    features = ["DPR", "PNA", "TAI"]

    # Fit on non-masked rows (proxy PNA distortion mitigation)
    masked_col = df.get("Privacy_Masked", pd.Series(False, index=df.index))
    fit_mask = ~masked_col
    X_fit = df.loc[fit_mask, features].fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=1 - OUTLIER_PERCENTILE,
        random_state=42,
    )
    model.fit(X_fit)

    X_all = df[features].fillna(0)
    df = df.copy()
    df["anomaly_score"] = model.predict(X_all)
    df["iso_score"] = model.decision_function(X_all)

    outliers = df[df["anomaly_score"] == -1].copy()
    outliers["detection_method"] = "isolation_forest"

    logger.info(
        "IsolationForest: %d / %d PINcodes flagged (fit on %d unmasked rows)",
        len(outliers), len(df), int(fit_mask.sum()),
    )
    return outliers


def cluster_anomalies_dbscan(
    outliers_df: pd.DataFrame,
    eps: float = _DBSCAN_EPS,
    min_samples: int = _DBSCAN_MIN_SAMPLES,
    features: Optional[list] = None,
) -> pd.DataFrame:
    """
    Stage 2 of the hierarchical ensemble — Paper B §4.

    Applies DBSCAN in standardized (DPR, PNA, TAI) feature space to the
    IsolationForest-flagged subset. Dense clusters represent groups of
    PINcodes with correlated anomaly signatures — stronger evidence of
    systematic irregularity than isolated outliers.

    Parameters
    ----------
    outliers_df : output of flag_anomalies_isolation_forest() or the
                  if_outliers from build_ensemble_outliers()
    eps         : DBSCAN neighbourhood radius in StandardScaler-normalized space.
                  0.5 = within half a standard deviation on all features jointly.
    min_samples : minimum PINcodes to constitute a cluster (not noise).

    Returns
    -------
    outliers_df with added columns:
      dbscan_cluster      — cluster ID (≥ 0) or -1 (noise / isolated outlier)
      dbscan_is_clustered — True if point belongs to a dense cluster
    """
    if features is None:
        features = _DBSCAN_FEATURES

    df = outliers_df.copy()

    if len(df) < min_samples:
        logger.warning(
            "cluster_anomalies_dbscan: only %d rows — fewer than min_samples=%d. "
            "All points will be labeled as noise (-1).",
            len(df), min_samples,
        )
        df["dbscan_cluster"] = -1
        df["dbscan_is_clustered"] = False
        return df

    X = df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    df["dbscan_cluster"] = labels
    df["dbscan_is_clustered"] = labels >= 0

    unique_clusters = set(labels) - {-1}
    n_clusters = len(unique_clusters)
    n_clustered = int((labels >= 0).sum())
    n_noise = int((labels == -1).sum())

    logger.info(
        "DBSCAN (ε=%.2f, min_samples=%d): %d clusters | "
        "%d clustered PINcodes | %d noise (isolated outliers)",
        eps, min_samples, n_clusters, n_clustered, n_noise,
    )
    return df


# ── Hierarchical ensemble (primary V2 pipeline path) ─────────────────────────

def build_ensemble_outliers(df: pd.DataFrame) -> tuple:
    """
    Hierarchical Ensemble Framework — Paper B §4.

    Stage 1: IsolationForest detects global multivariate anomalies.
    Stage 2: DBSCAN identifies coordinated cluster patterns within Stage 1 output.

    Key design decision:
      iso_score is added to ALL rows in the returned full_df so that
      sentinel_final.csv captures how close every PINCODE is to being flagged.
      This is a feature for IndiaID-Bench (Paper A) and for the dashboard
      Top-N ranking within the full distribution.

    Returns
    -------
    (full_df, if_outliers, ensemble_df, stats)

    full_df      : input df with anomaly_score and iso_score on all rows
    if_outliers  : Stage 1 flagged rows only (backward-compat with V1 outliers_ml.csv)
    ensemble_df  : Stage 1 flagged rows + dbscan_cluster and dbscan_is_clustered columns
    stats        : dict with Paper B §5 result numbers
    """
    features = ["DPR", "PNA", "TAI"]

    # ── Stage 1: IsolationForest ──────────────────────────────────────────
    masked_col = df.get("Privacy_Masked", pd.Series(False, index=df.index))
    fit_mask = ~masked_col
    X_fit = df.loc[fit_mask, features].fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=1 - OUTLIER_PERCENTILE,
        random_state=42,
    )
    model.fit(X_fit)

    X_all = df[features].fillna(0)
    full_df = df.copy()
    full_df["anomaly_score"] = model.predict(X_all)
    full_df["iso_score"] = model.decision_function(X_all)

    if_outliers = full_df[full_df["anomaly_score"] == -1].copy()
    if_outliers["detection_method"] = "isolation_forest"

    n_if = len(if_outliers)
    logger.info(
        "Stage 1 (IsolationForest): %d / %d PINcodes flagged (fit on %d unmasked)",
        n_if, len(full_df), int(fit_mask.sum()),
    )

    # ── Stage 2: DBSCAN on Stage 1 output ────────────────────────────────
    ensemble_df = cluster_anomalies_dbscan(if_outliers)

    unique_clusters = set(ensemble_df["dbscan_cluster"].unique()) - {-1}
    n_clusters = len(unique_clusters)
    n_clustered = int(ensemble_df["dbscan_is_clustered"].sum())

    stats = {
        "if_flagged": n_if,
        "dbscan_clusters": n_clusters,
        "dbscan_clustered_pincodes": n_clustered,
        "dbscan_isolated_outliers": int((ensemble_df["dbscan_cluster"] == -1).sum()),
    }

    logger.info(
        "Stage 2 (DBSCAN): %d clusters | %d clustered PINcodes | %d isolated",
        n_clusters, n_clustered, stats["dbscan_isolated_outliers"],
    )

    return full_df, if_outliers, ensemble_df, stats