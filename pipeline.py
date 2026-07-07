"""
pipeline.py
══════════════════════════════════════════════════════════════════════════════
End-to-End Pipeline Runner — Aadhaar Sentinel V2.1

Single entry point. Run with:
    python pipeline.py

V2 CHANGES:
  - Uses build_ensemble_outliers (IF + DBSCAN) instead of standalone IF call.
  - iso_score and anomaly_score are now saved in sentinel_final.csv (on ALL
    PINcodes), not just in outliers_ml.csv.
  - Saves four output CSVs instead of two:
      sentinel_final.csv   — all PINcodes with all metrics + iso_score
      outliers_ml.csv      — IF-flagged subset (backward-compat with app.py)
      ensemble_final.csv   — IF-flagged subset + DBSCAN/HDBSCAN cluster labels
      outliers_stat.csv    — statistical baseline outliers (previously unsaved)
  - Generates HTML maps after scoring (wrapped in try/except so a missing
    coordinates file does not abort the whole pipeline).
  - Prints a structured summary matching the Paper B §5 result numbers.

V2.1 ADDITIONS:
  - build_ensemble_outliers upgraded to three-method ensemble:
      Method 1: Statistical 98th-percentile baseline (TAI, DPR, PNA).
      Method 2: IsolationForest (global multivariate outliers).
      Method 3: Local Outlier Factor (density-based local outliers).
    Stage 2 applies HDBSCAN (primary clustering) and DBSCAN (ablation baseline)
    on the IF-flagged subset. Summary now reports LOF flags, IF–LOF overlap,
    HDBSCAN clusters, and isolated outliers, in addition to existing DBSCAN
    stats.
"""

import os
import time

import pandas as pd

from src import config
from src.engine import calculate_base_metrics
from src.loader import load_and_concat_csvs, preprocess_and_aggregate
from src.maps import generate_all_maps
from src.scoring import (
    add_population_uncertainty,
    build_ensemble_outliers,
    compute_risk_score,
    flag_anomalies_statistical,
)


def run() -> None:
    t_start = time.time()

    # ── Step 1: Load raw data ─────────────────────────────────────────────
    print("Loading raw data...")
    demo_df = load_and_concat_csvs(
        os.path.join(config.RAW_DATA_DIR, "aadhaar_demographic_updates")
    )
    bio_df = load_and_concat_csvs(
        os.path.join(config.RAW_DATA_DIR, "aadhaar_biometric_update_pincode")
    )
    enr_df = load_and_concat_csvs(
        os.path.join(config.RAW_DATA_DIR, "aadhaar_enrolment_pincode")
    )
    district_pop = pd.read_csv(config.CENSUS_PATH)

    # ── Step 2: Aggregate to PINCODE level ────────────────────────────────
    print("Aggregating to PINCODE level...")
    demo_pin, bio_pin, enr_pin = preprocess_and_aggregate(demo_df, bio_df, enr_df)

    # ── Step 3: Compute TAI / DPR / DPR_v2 / PNA with RGI projections ────
    print("Computing metrics (V2 RGI projections)...")
    base_df = calculate_base_metrics(demo_pin, bio_pin, enr_pin, district_pop)

    # ── Step 4: Risk scoring ──────────────────────────────────────────────
    print("Scoring (composite risk + uncertainty bounds)...")
    scored_df = compute_risk_score(base_df)
    scored_df = add_population_uncertainty(scored_df)

    # ── Step 5: Anomaly detection — hierarchical IF + LOF + HDBSCAN/DBSCAN ─
    print("Anomaly detection (stat + IF + LOF → HDBSCAN/DBSCAN ensemble)...")
    scored_df, if_outliers, ensemble_df, ensemble_stats = build_ensemble_outliers(scored_df)

    # Step 5b: Statistical baseline (for Paper B comparison)
    outliers_stat = flag_anomalies_statistical(scored_df)

    # ── Step 6: Save outputs ──────────────────────────────────────────────
    print("Saving results...")
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    scored_df.to_csv(
        os.path.join(config.PROCESSED_DIR, "sentinel_final.csv"), index=False
    )
    if_outliers.to_csv(
        os.path.join(config.PROCESSED_DIR, "outliers_ml.csv"), index=False
    )
    ensemble_df.to_csv(
        os.path.join(config.PROCESSED_DIR, "ensemble_final.csv"), index=False
    )
    outliers_stat.to_csv(
        os.path.join(config.PROCESSED_DIR, "outliers_stat.csv"), index=False
    )

    # ── Step 7: Generate maps ─────────────────────────────────────────────
    print("Generating geospatial maps...")
    try:
        generate_all_maps(scored_df)
        maps_status = "Generated"
    except Exception as e:
        maps_status = f"Skipped ({e})"
        print(f"  Map generation non-fatal error: {e}")

    # ── Step 8: Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t_start

    # Statistical vs IF overlap (Paper B §5 baseline vs Method 2)
    stat_pins = set(outliers_stat["pincode"].astype(str))
    ml_pins = set(if_outliers["pincode"].astype(str))
    overlap_stat_if = len(stat_pins & ml_pins)
    overlap_stat_if_pct = (
        overlap_stat_if / ensemble_stats["if_flagged"] * 100
        if ensemble_stats["if_flagged"] > 0
        else 0.0
    )

    # IF ∩ LOF high-confidence set (Paper B §5 primary validation result)
    overlap_if_lof = ensemble_stats.get("if_lof_overlap", 0)
    overlap_if_lof_pct = (
        overlap_if_lof / ensemble_stats["if_flagged"] * 100
        if ensemble_stats["if_flagged"] > 0
        else 0.0
    )

    privacy_masked = int(scored_df.get("Privacy_Masked", pd.Series(False)).sum())

    print(f"\n{'=' * 52}")
    print(f"  Aadhaar Sentinel V2.1 — Pipeline Complete ({elapsed:.1f}s)")
    print(f"{'=' * 52}")
    print(f"  Total PINcodes analysed   : {len(scored_df):>10,}")
    print(f"  Statistical flags (98th)  : {len(outliers_stat):>10,}")
    print(f"  Isolation Forest flags    : {ensemble_stats['if_flagged']:>10,}")
    print(f"  LOF flags                 : {ensemble_stats.get('lof_flagged', 0):>10,}")
    print(f"  Stat–IF overlap           : {overlap_stat_if:>10,}  ({overlap_stat_if_pct:.0f}%)")
    print(f"  IF–LOF overlap            : {overlap_if_lof:>10,}  ({overlap_if_lof_pct:.0f}%)")
    print(f"  HDBSCAN clusters          : {ensemble_stats.get('hdbscan_clusters', 0):>10,}")
    print(f"  HDBSCAN clustered PINcodes: {ensemble_stats.get('hdbscan_clustered_pincodes', 0):>10,}")
    print(f"  HDBSCAN isolated outliers : {ensemble_stats.get('hdbscan_isolated_outliers', 0):>10,}")
    print(f"  DBSCAN clusters           : {ensemble_stats['dbscan_clusters']:>10,}")
    print(f"  Clustered PINcodes        : {ensemble_stats['dbscan_clustered_pincodes']:>10,}")
    print(f"  Isolated IF outliers      : {ensemble_stats['dbscan_isolated_outliers']:>10,}")
    print(f"  Privacy_Masked (flagged)  : {privacy_masked:>10,}")
    print(f"{'=' * 52}")
    print(f"  Outputs  → {config.PROCESSED_DIR}/")
    print(f"  Maps     → {maps_status}")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    run()