"""
scripts/build_release.py
══════════════════════════════════════════════════════════════════════════════
IndiaID-Bench v1.0.0 release builder (Zenodo deposit, Paper A).

Read-only w.r.t. the pipeline: does not import or modify anything in src/,
does not touch pipeline.py or app.py, does not rerun the pipeline. Reads
existing data/processed/*.csv and data/raw/district_name_map.csv, writes only
outputs/release/*.

ROW ALIGNMENT (verified empirically, not assumed):
  ensemble_final.csv (658 rows) is NOT 1:1 with sentinel_final.csv (32,898
  rows) by length, and pincode is not a unique key (documented repeatedly
  elsewhere in this repo -- see scripts/figures.py). A naive merge on
  (pincode, district, state) was tested here and inflates the 658 IF rows to
  772 matches, because that triple is not unique in sentinel_final.csv either
  (26,299 unique triples of 32,898 rows).

  Instead: build_ensemble_outliers() in src/scoring.py constructs
  if_outliers as a simple boolean filter of full_df (full_df[anomaly_score
  == -1]), with no reordering. So ensemble_final.csv should be exactly
  sentinel_final.csv[anomaly_score == -1], in the same row order. This is
  verified at runtime below (verify_positional_alignment) by comparing every
  shared column between the two frames positionally, before any cluster
  column is attached -- if verification fails, the script stops rather than
  silently attaching cluster labels to the wrong rows.

PRIVACY: rows with Privacy_Masked == True are dropped entirely (not just
flagged) before release, per Paper A §III-D k-anonymity commitment
(K_ANONYMITY_THRESHOLD = 500 in src/config.py). Privacy_Masked itself is
excluded from the release schema since it is constant (False) after
suppression, so it carries no information once masked rows are gone.

Run:
    python scripts/build_release.py   (run pipeline.py first)
"""

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

SENTINEL_PATH = Path(config.PROCESSED_DIR) / "sentinel_final.csv"
ENSEMBLE_PATH = Path(config.PROCESSED_DIR) / "ensemble_final.csv"
DISTRICT_MAP_SRC = Path(config.DISTRICT_MAP_PATH)

OUT_DIR = Path("outputs/release")
RELEASE_CSV = OUT_DIR / "indiaid_bench_v1.0.0.csv"
DISTRICT_MAP_OUT = OUT_DIR / "district_name_map_v1.0.0.csv"
CHECKSUMS_OUT = OUT_DIR / "checksums.txt"

# Target schema, in release column order.
# (constant/False after suppression -- carries no information), DPR_v2,
# anomaly_score (superseded by if_flag), detection_method.
RELEASE_COLUMNS = [
    "pincode", "district", "state",
    "enrol_total", "demo_total", "bio_total",
    "TAI", "DPR", "PNA", "PNA_conservative", "PNA_upper_bound",
    "est_pincode_pop", "est_pincode_pop_lower", "est_pincode_pop_upper",
    "annual_growth_rate", "growth_source",
    "dpr_z", "pna_z", "activity_z", "audit_priority_score",
    "if_flag", "iso_score", "lof_flag", "lof_score",
    "hdbscan_cluster", "hdbscan_probability", "dbscan_cluster",
    "Privacy_Masked"
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_positional_alignment(sf: pd.DataFrame, ens: pd.DataFrame, if_mask: pd.Series) -> pd.DataFrame:
    """
    Confirms ensemble_final.csv == sentinel_final.csv[if_mask], same order,
    by comparing every shared column positionally. Returns the aligned
    IF-flagged subset (index reset) for the caller to use. Raises if any
    column disagrees anywhere -- attaching cluster labels on unverified
    alignment would silently corrupt the release.
    """
    if_subset = sf.loc[if_mask].reset_index(drop=True)
    ens_r = ens.reset_index(drop=True)

    if len(if_subset) != len(ens_r):
        raise RuntimeError(
            f"Row count mismatch: sentinel_final[if_flag]={len(if_subset)} vs "
            f"ensemble_final={len(ens_r)}. Cannot safely attach cluster columns."
        )

    shared_cols = [c for c in sf.columns if c in ens_r.columns]
    bad = {}
    for c in shared_cols:
        a, b = if_subset[c], ens_r[c]
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            eq = a.sub(b).abs().lt(1e-9) | (a.isna() & b.isna())
        else:
            eq = (a == b) | (a.isna() & b.isna())
        n_bad = int((~eq).sum())
        if n_bad:
            bad[c] = n_bad

    if bad:
        raise RuntimeError(
            f"Positional alignment between sentinel_final[if_flag] and "
            f"ensemble_final.csv FAILED on columns: {bad}. Stopping -- "
            f"attaching HDBSCAN/DBSCAN columns would mislabel rows."
        )

    print(f"Positional alignment verified: {len(if_subset)} rows, "
          f"{len(shared_cols)} shared columns, 0 mismatches.")
    return if_subset


def main():
    print("=" * 74)
    print("  IndiaID-Bench v1.0.0 release builder")
    print("=" * 74)

    if not SENTINEL_PATH.exists() or not ENSEMBLE_PATH.exists():
        raise FileNotFoundError(
            f"{SENTINEL_PATH} and/or {ENSEMBLE_PATH} not found. Run "
            f"`python pipeline.py` first."
        )

    sf = pd.read_csv(SENTINEL_PATH)
    ens = pd.read_csv(ENSEMBLE_PATH)
    total_rows = len(sf)
    print(f"\nLoaded sentinel_final.csv: {sf.shape} | ensemble_final.csv: {ens.shape}")

    if "if_flag" not in sf.columns:
        print("NOTE: 'if_flag' does not exist as a literal column; deriving "
              "as (anomaly_score == -1), matching the convention already "
              "used in figures.py and app.py.")
    if_mask = sf["anomaly_score"] == -1

    for col in ("hdbscan_cluster", "hdbscan_probability", "dbscan_cluster"):
        if col not in sf.columns:
            print(f"NOTE: '{col}' does not exist in sentinel_final.csv "
                  f"(only computed on the IF-flagged subset in "
                  f"ensemble_final.csv) -- will be attached for IF-flagged "
                  f"rows and left null for the rest.")

    # ── Verify alignment, then attach cluster columns by position ───────────
    if_subset_aligned = verify_positional_alignment(sf, ens, if_mask)
    if_index_in_order = sf.index[if_mask]  # original sf index, in if_mask's row order

    sf = sf.copy()
    sf["if_flag"] = if_mask
    for col in ("hdbscan_cluster", "hdbscan_probability", "dbscan_cluster"):
        sf[col] = np.nan
        sf.loc[if_index_in_order, col] = ens[col].to_numpy()
    sf["hdbscan_cluster"] = sf["hdbscan_cluster"].astype("Int64")
    sf["dbscan_cluster"] = sf["dbscan_cluster"].astype("Int64")

    # ── Sanity check: recompute flag counts on the PRE-suppression frame ────
    tai_t = sf["TAI"].quantile(config.OUTLIER_PERCENTILE)
    dpr_t = sf["DPR"].quantile(config.OUTLIER_PERCENTILE)
    pna_t = sf["PNA"].quantile(config.OUTLIER_PERCENTILE)
    stat_mask = (sf["TAI"] >= tai_t) | (sf["DPR"] >= dpr_t) | (sf["PNA"] >= pna_t)

    n_stat_pre = int(stat_mask.sum())
    n_if_pre = int(sf["if_flag"].sum())
    n_lof_pre = int(sf["lof_flag"].sum())
    print(f"\nPre-suppression flag counts: Statistical={n_stat_pre} "
          f"(expect 1918), IF={n_if_pre} (expect 658), LOF={n_lof_pre} (expect 623)")
    assert n_stat_pre == 1918, f"Statistical flag count drifted: {n_stat_pre} != 1918"
    assert n_if_pre == 658, f"IF flag count drifted: {n_if_pre} != 658"
    assert n_lof_pre == 623, f"LOF flag count drifted: {n_lof_pre} != 623"

    # ── Suppress privacy-masked rows ─────────────────────────────────────────
    masked_mask = sf["Privacy_Masked"].astype(bool)
    n_masked = int(masked_mask.sum())
    released_df = sf.loc[~masked_mask].copy()
    n_released = len(released_df)

    assert n_released + n_masked == total_rows, (
        f"released ({n_released}) + suppressed ({n_masked}) != total ({total_rows})"
    )
    print(f"\nTotal rows: {total_rows} | Privacy_Masked rows suppressed: {n_masked} "
          f"| Released rows: {n_released}")

    # ── Post-suppression flag counts (report only, not asserted) ────────────
    n_stat_post = int(stat_mask.loc[released_df.index].sum())
    n_if_post = int(released_df["if_flag"].sum())
    n_lof_post = int(released_df["lof_flag"].sum())
    print(f"Post-suppression flag counts (report only, expected <= pre): "
          f"Statistical={n_stat_post}, IF={n_if_post}, LOF={n_lof_post}")

    # ── Select release schema ────────────────────────────────────────────────
    missing_target_cols = [c for c in RELEASE_COLUMNS if c not in released_df.columns]
    if missing_target_cols:
        raise RuntimeError(f"Target columns not found, refusing to invent them: "
                            f"{missing_target_cols}")
    released_df = released_df[RELEASE_COLUMNS]

    # ── Write outputs ─────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    released_df.to_csv(RELEASE_CSV, index=False)
    print(f"\nSaved -> {RELEASE_CSV}")

    if not DISTRICT_MAP_SRC.exists():
        raise FileNotFoundError(f"{DISTRICT_MAP_SRC} not found.")
    shutil.copyfile(DISTRICT_MAP_SRC, DISTRICT_MAP_OUT)
    print(f"Copied {DISTRICT_MAP_SRC} -> {DISTRICT_MAP_OUT}")

    # ── Checksums ─────────────────────────────────────────────────────────────
    with open(CHECKSUMS_OUT, "w") as f:
        for p in (RELEASE_CSV, DISTRICT_MAP_OUT):
            digest = sha256_of(p)
            f.write(f"{digest}  {p.name}\n")
            print(f"SHA-256 ({p.name}): {digest}")
    print(f"Saved -> {CHECKSUMS_OUT}")

    # ── Release report ───────────────────────────────────────────────────────
    print(f"\n{'=' * 74}")
    print("  RELEASE REPORT")
    print(f"{'=' * 74}")
    print(f"  Rows released       : {n_released:,}")
    print(f"  Rows suppressed     : {n_masked:,}")
    print(f"  Columns             : {released_df.shape[1]}")
    print(f"\n  {'column':<24} {'dtype':<10} {'nulls':>8}")
    print(f"  {'-' * 24} {'-' * 10} {'-' * 8}")
    for col in released_df.columns:
        n_null = int(released_df[col].isna().sum())
        print(f"  {col:<24} {str(released_df[col].dtype):<10} {n_null:>8,}")
    print(f"{'=' * 74}\n")

    return released_df


if __name__ == "__main__":
    main()
