"""
src/scoring.py
══════════════════════════════════════════════════════════════════════════════
Anomaly Detection & Risk Scoring — Aadhaar Sentinel V2.2

V2 CHANGES (from V1):
  1. add_population_uncertainty — FIXED double-counting bug.
  2. flag_anomalies_isolation_forest — FIXED Privacy_Masked exclusion.
  3. build_ensemble_outliers — IF → DBSCAN hierarchical pipeline.
  4. cluster_anomalies_dbscan — DBSCAN in standardized feature space.

V2.1 ADDITIONS (this file):
  5. flag_anomalies_lof — Local Outlier Factor as Method 3 (Paper B §4).
       One implementation, two callers: standalone and build_ensemble_outliers.
       Uses novelty=True to mirror IF protocol exactly: fit on non-masked rows,
       score all rows. lof_score uses decision_function (same sign convention
       as iso_score: negative = anomaly, threshold at 0).

  6. cluster_anomalies_hdbscan — HDBSCAN primary clustering (Paper B §4, Stage 2).
       Available directly from sklearn.cluster (sklearn >= 1.3). No external
       hdbscan package required. Runs on IF-flagged subset, same input as DBSCAN,
       enabling direct ablation comparison. DBSCAN retained for ablation.

  7. build_ensemble_outliers — upgraded to three-method ensemble.
       Stage 1a: IsolationForest (global multivariate outliers).
       Stage 1b: LOF (density-based, local neighbourhood anomalies).
       Stage 2a: HDBSCAN on IF-flagged subset (primary cluster method).
       Stage 2b: DBSCAN on IF-flagged subset (ablation baseline).
       Stats dict adds LOF and HDBSCAN keys while keeping all existing
       DBSCAN keys for pipeline.py backward compatibility.
       IF-LOF overlap is reported as the high-confidence set size.

PAPER B §4 FRAMING:
  "We implement a three-method ensemble. Method 1: statistical 98th-percentile
  baseline (univariate union on TAI, DPR, PNA). Method 2: IsolationForest
  (n_estimators=200, contamination=0.02) on the joint (TAI, DPR, PNA) feature
  space. Method 3: Local Outlier Factor (n_neighbors=20, contamination=0.02)
  on the same feature space. Both Methods 2 and 3 fit exclusively on non-
  Privacy_Masked PINcodes to prevent sparse rural proxy-PNA values from
  distorting model boundaries; all PINcodes are scored regardless.
  Stage 2 applies HDBSCAN (min_cluster_size=5, min_samples=3) and DBSCAN
  (eps=0.5, min_samples=3) in StandardScaler-normalized feature space to the
  IF-flagged subset, with HDBSCAN as the primary method and DBSCAN as an
  ablation baseline. The IF-LOF intersection is the high-confidence anomaly
  set and constitutes the primary Paper B §5 validation result."
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.config import (
    OUTLIER_PERCENTILE,
    WEIGHT_ACTIVITY,
    WEIGHT_DPR,
    WEIGHT_PNA,
)

logger = logging.getLogger(__name__)

# -- Shared feature list ------------------------------------------------------
_ANOMALY_FEATURES = ["DPR", "PNA", "TAI"]

# -- DBSCAN parameters (ablation baseline) ------------------------------------
_DBSCAN_EPS = 0.5
_DBSCAN_MIN_SAMPLES = 3
_DBSCAN_FEATURES = _ANOMALY_FEATURES

# -- HDBSCAN parameters (primary Stage 2) -------------------------------------
# sklearn.cluster.HDBSCAN, available since sklearn 1.3. No external package.
_HDBSCAN_MIN_CLUSTER_SIZE = 5
_HDBSCAN_MIN_SAMPLES = 3

# -- LOF parameters -----------------------------------------------------------
_LOF_N_NEIGHBORS = 20   # sensitivity sweep: 10-50 (see scripts/sensitivity.py)


# -- Risk scoring -------------------------------------------------------------

def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalized composite risk score.

    audit_priority_score = (dpr_z x 0.50) + (pna_z x 0.35) + (activity_z x 0.15)

    Weights documented in config.py. Justified by audit logic:
      - DPR (0.50): process non-compliance -- hardest to detect, highest urgency.
      - PNA (0.35): infrastructure stress -- requires operational response.
      - TAI (0.15): overall load context.

    Score shifted to >= 0 range for dashboard readability.
    Z-scores computed nationally (not per-state) for cross-district comparability.
    Paper B Table 2 reports score distribution.
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

    V2 (est_pincode_pop_lower/upper present -- RGI projection bounds):
      PNA_conservative = TAI / est_pincode_pop_lower
      PNA_upper_bound  = TAI / est_pincode_pop_upper

    V1 fallback (V2 columns absent -- Census 2011 era):
      PNA_conservative = TAI / est_pincode_pop
      PNA_upper_bound  = TAI / (est_pincode_pop x POP_GROWTH_FACTOR_MAX)
    """
    df = df.copy()

    if "est_pincode_pop_lower" in df.columns and "est_pincode_pop_upper" in df.columns:
        df["PNA_conservative"] = df["TAI"] / df["est_pincode_pop_lower"].replace(0, np.nan)
        df["PNA_upper_bound"] = df["TAI"] / df["est_pincode_pop_upper"].replace(0, np.nan)
    else:
        from src.config import POP_GROWTH_FACTOR_MAX
        logger.warning(
            "add_population_uncertainty: est_pincode_pop_lower/upper not found. "
            "Using V1 flat-rate multiplier (POP_GROWTH_FACTOR_MAX=%.2f).",
            POP_GROWTH_FACTOR_MAX,
        )
        df["PNA_conservative"] = df["TAI"] / df["est_pincode_pop"]
        df["PNA_upper_bound"] = df["TAI"] / (df["est_pincode_pop"] * POP_GROWTH_FACTOR_MAX)

    return df


# -- Statistical baseline -----------------------------------------------------

def flag_anomalies_statistical(df: pd.DataFrame) -> pd.DataFrame:
    """
    98th-percentile baseline -- Paper B comparison baseline (Method 1).

    Flags PINcodes in top OUTLIER_PERCENTILE on ANY of TAI, DPR, PNA (union).
    Univariate: treats each metric independently. Cannot capture multivariate
    structure -- that is the motivation for IF and LOF in Methods 2 and 3.
    """
    tai_t = df["TAI"].quantile(OUTLIER_PERCENTILE)
    dpr_t = df["DPR"].quantile(OUTLIER_PERCENTILE)
    pna_t = df["PNA"].quantile(OUTLIER_PERCENTILE)

    mask = (df["TAI"] >= tai_t) | (df["DPR"] >= dpr_t) | (df["PNA"] >= pna_t)
    outliers = df[mask].copy()
    outliers["detection_method"] = "statistical_percentile"
    return outliers


# -- Method 2: IsolationForest ------------------------------------------------

def flag_anomalies_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standalone IsolationForest. Backward-compatible. For the full three-method
    ensemble use build_ensemble_outliers().

    V2 fix: fits on non-Privacy_Masked rows only; scores all rows.
    """
    masked_col = df.get("Privacy_Masked", pd.Series(False, index=df.index))
    fit_mask = ~masked_col
    X_fit = df.loc[fit_mask, _ANOMALY_FEATURES].fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=1 - OUTLIER_PERCENTILE,
        random_state=42,
    )
    model.fit(X_fit)

    X_all = df[_ANOMALY_FEATURES].fillna(0)
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


# -- Method 3: Local Outlier Factor -------------------------------------------

def flag_anomalies_lof(
    df: pd.DataFrame,
    n_neighbors: int = _LOF_N_NEIGHBORS,
) -> pd.DataFrame:
    """
    Local Outlier Factor -- Paper B §4, Method 3.

    LOF measures local density deviation: a PINCODE is anomalous if its local
    neighbourhood density is significantly lower than its neighbours'.

    IF vs LOF complementarity:
      - IsolationForest detects globally isolated points (low density globally).
      - LOF detects locally anomalous points (low density relative to local
        neighbourhood). Both are distinct failure signatures in enrollment data.
    Their intersection (IF and LOF) is the highest-confidence anomaly set.

    Architecture mirrors IsolationForest exactly:
      - novelty=True: fit on non-Privacy_Masked rows, score ALL rows.
      - Same contamination as IF (1 - OUTLIER_PERCENTILE) for fair comparison.
      - lof_score uses decision_function: same sign convention as iso_score
        (negative = anomaly, threshold at 0, more negative = stronger anomaly).

    Parameters
    ----------
    n_neighbors : LOF neighbourhood size. Default 20.
                  Sensitivity sweep covers 10-50 (scripts/sensitivity.py).

    Returns
    -------
    df with added columns:
        lof_flag  -- True if flagged as LOF anomaly
        lof_score -- decision_function output (negative = anomaly, mirrors iso_score)
    """
    masked_col = df.get("Privacy_Masked", pd.Series(False, index=df.index))
    fit_mask = ~masked_col

    X_fit = df.loc[fit_mask, _ANOMALY_FEATURES].fillna(0).values
    X_all = df[_ANOMALY_FEATURES].fillna(0).values

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=1 - OUTLIER_PERCENTILE,
        novelty=True,
    )
    lof.fit(X_fit)

    df = df.copy()
    df["lof_flag"] = lof.predict(X_all) == -1
    df["lof_score"] = lof.decision_function(X_all)

    n_flagged = int(df["lof_flag"].sum())
    logger.info(
        "LOF (n_neighbors=%d): %d / %d PINcodes flagged (fit on %d unmasked rows)",
        n_neighbors, n_flagged, len(df), int(fit_mask.sum()),
    )
    return df


# -- Stage 2a: DBSCAN (ablation baseline) -------------------------------------

def cluster_anomalies_dbscan(
    outliers_df: pd.DataFrame,
    eps: float = _DBSCAN_EPS,
    min_samples: int = _DBSCAN_MIN_SAMPLES,
    features: Optional[list] = None,
) -> pd.DataFrame:
    """
    DBSCAN clustering on IF-flagged subset -- Paper B §4, ablation baseline.

    Retained for comparison against HDBSCAN. DBSCAN requires a fixed epsilon
    that cannot adapt to India's variable urban/rural density. HDBSCAN is
    the primary method; DBSCAN is the ablation baseline.

    Returns outliers_df with dbscan_cluster and dbscan_is_clustered columns.
    """
    if features is None:
        features = _DBSCAN_FEATURES

    df = outliers_df.copy()

    if len(df) < min_samples:
        logger.warning(
            "cluster_anomalies_dbscan: %d rows < min_samples=%d. All noise (-1).",
            len(df), min_samples,
        )
        df["dbscan_cluster"] = -1
        df["dbscan_is_clustered"] = False
        return df

    X = df[features].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)

    df["dbscan_cluster"] = labels
    df["dbscan_is_clustered"] = labels >= 0

    n_clusters = len(set(labels) - {-1})
    n_clustered = int((labels >= 0).sum())
    logger.info(
        "DBSCAN (eps=%.2f, min_samples=%d): %d clusters | %d clustered | %d noise",
        eps, min_samples, n_clusters, n_clustered, int((labels == -1).sum()),
    )
    return df


# -- Stage 2b: HDBSCAN (primary clustering) -----------------------------------

def cluster_anomalies_hdbscan(
    outliers_df: pd.DataFrame,
    min_cluster_size: int = _HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = _HDBSCAN_MIN_SAMPLES,
    features: Optional[list] = None,
) -> pd.DataFrame:
    """
    HDBSCAN clustering -- primary cluster method, Paper B §4 Stage 2.

    Available directly from sklearn.cluster (sklearn >= 1.3, confirmed in 1.8).
    No external hdbscan package required.

    HDBSCAN advantage over DBSCAN for India's enrollment data:
      Urban PINcodes form dense clusters; rural and border PINcodes exist at
      much lower density. DBSCAN's single epsilon cannot handle both. HDBSCAN
      selects the most stable clusters across all epsilon values, adapting to
      local density variation without manual epsilon tuning.

    hdbscan_probability provides soft cluster membership [0, 1], enabling
    ranked cluster membership for audit prioritisation beyond binary assignment.

    Runs on the same IF-flagged input as DBSCAN for direct ablation comparison.

    Returns
    -------
    outliers_df with added columns:
        hdbscan_cluster      -- cluster ID (>= 0) or -1 (noise/isolated)
        hdbscan_is_clustered -- True if point belongs to a stable cluster
        hdbscan_probability  -- soft membership confidence [0, 1]
    """
    if features is None:
        features = _ANOMALY_FEATURES

    df = outliers_df.copy()

    if len(df) < min_cluster_size:
        logger.warning(
            "cluster_anomalies_hdbscan: %d rows < min_cluster_size=%d. All noise.",
            len(df), min_cluster_size,
        )
        df["hdbscan_cluster"] = -1
        df["hdbscan_is_clustered"] = False
        df["hdbscan_probability"] = 0.0
        return df

    X = df[features].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        copy=True,
    )
    clusterer.fit(X_scaled)

    df["hdbscan_cluster"] = clusterer.labels_
    df["hdbscan_is_clustered"] = clusterer.labels_ >= 0
    df["hdbscan_probability"] = clusterer.probabilities_

    n_clusters = len(set(clusterer.labels_) - {-1})
    n_clustered = int((clusterer.labels_ >= 0).sum())
    n_noise = int((clusterer.labels_ == -1).sum())

    logger.info(
        "HDBSCAN (min_cluster_size=%d, min_samples=%d): %d clusters | "
        "%d clustered PINcodes | %d noise",
        min_cluster_size, min_samples, n_clusters, n_clustered, n_noise,
    )
    return df


# -- Three-method ensemble (primary pipeline path) ----------------------------

def build_ensemble_outliers(df: pd.DataFrame) -> tuple:
    """
    Three-Method Hierarchical Ensemble -- Paper B §4.

    Stage 1a: IsolationForest -- global multivariate outliers.
    Stage 1b: LOF -- density-based local outliers (calls flag_anomalies_lof).
    Stage 2a: HDBSCAN on IF-flagged subset -- primary cluster structure.
    Stage 2b: DBSCAN on IF-flagged subset -- ablation baseline.

    Key design decisions:
    - Both IF and LOF fit on non-Privacy_Masked rows (same boundary protocol).
    - iso_score and lof_score appear on ALL rows in full_df so that
      sentinel_final.csv captures continuous anomaly depth, not just binary flags.
    - HDBSCAN and DBSCAN both run on IF-flagged subset (same input, direct
      ablation comparison). IF-LOF overlap is reported as a separate high-confidence
      stat, not used as cluster input -- this preserves existing stats semantics.
    - All existing DBSCAN stats keys preserved for pipeline.py backward compat.

    Returns
    -------
    (full_df, if_outliers, ensemble_df, stats)

    full_df      : all PINcodes with iso_score, lof_flag, lof_score on every row
    if_outliers  : IF-flagged rows (backward compat -- outliers_ml.csv)
    ensemble_df  : IF-flagged rows + HDBSCAN + DBSCAN cluster columns
    stats        : Paper B §5 result numbers -- all methods, overlaps, clusters
    """
    # -- Stage 1a: IsolationForest --------------------------------------------
    masked_col = df.get("Privacy_Masked", pd.Series(False, index=df.index))
    fit_mask = ~masked_col
    X_fit = df.loc[fit_mask, _ANOMALY_FEATURES].fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=1 - OUTLIER_PERCENTILE,
        random_state=42,
    )
    model.fit(X_fit)

    full_df = df.copy()
    X_all = df[_ANOMALY_FEATURES].fillna(0)
    full_df["anomaly_score"] = model.predict(X_all)
    full_df["iso_score"] = model.decision_function(X_all)

    if_outliers = full_df[full_df["anomaly_score"] == -1].copy()
    if_outliers["detection_method"] = "isolation_forest"
    n_if = len(if_outliers)

    logger.info(
        "Stage 1a (IsolationForest): %d / %d PINcodes flagged (fit on %d unmasked)",
        n_if, len(full_df), int(fit_mask.sum()),
    )

    # -- Stage 1b: LOF (single implementation called here) -------------------
    full_df = flag_anomalies_lof(full_df)
    n_lof = int(full_df["lof_flag"].sum())

    if_idx = set(if_outliers.index)
    lof_idx = set(full_df[full_df["lof_flag"]].index)
    n_if_lof = len(if_idx & lof_idx)

    logger.info(
        "Stage 1b (LOF): %d flagged | IF-LOF overlap = %d (%.0f%% of IF)",
        n_lof, n_if_lof, (n_if_lof / n_if * 100) if n_if > 0 else 0,
    )

    # -- Stage 2a: HDBSCAN on IF-flagged subset (primary) --------------------
    ensemble_df = cluster_anomalies_hdbscan(if_outliers)

    # -- Stage 2b: DBSCAN on same input (ablation baseline) ------------------
    ensemble_df = cluster_anomalies_dbscan(ensemble_df)

    # -- Compute stats --------------------------------------------------------
    hdb_unique = set(ensemble_df["hdbscan_cluster"].unique()) - {-1}
    dbs_unique = set(ensemble_df["dbscan_cluster"].unique()) - {-1}

    stats = {
        # Existing keys -- backward compat with pipeline.py
        "if_flagged": n_if,
        "dbscan_clusters": len(dbs_unique),
        "dbscan_clustered_pincodes": int(ensemble_df["dbscan_is_clustered"].sum()),
        "dbscan_isolated_outliers": int((ensemble_df["dbscan_cluster"] == -1).sum()),
        # New keys -- Paper B §5 Table 2
        "lof_flagged": n_lof,
        "if_lof_overlap": n_if_lof,
        "hdbscan_clusters": len(hdb_unique),
        "hdbscan_clustered_pincodes": int(ensemble_df["hdbscan_is_clustered"].sum()),
        "hdbscan_isolated_outliers": int((ensemble_df["hdbscan_cluster"] == -1).sum()),
    }

    logger.info(
        "Stage 2 -- HDBSCAN: %d clusters | %d clustered | %d noise",
        stats["hdbscan_clusters"], stats["hdbscan_clustered_pincodes"],
        stats["hdbscan_isolated_outliers"],
    )

    return full_df, if_outliers, ensemble_df, stats
