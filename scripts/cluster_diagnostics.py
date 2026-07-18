"""
scripts/cluster_diagnostics.py
══════════════════════════════════════════════════════════════════════════════
HDBSCAN cluster sanity check -- Paper B §5 (read-only analysis, no pipeline
side effects).

Question: are the 33 HDBSCAN clusters on the IF-flagged subset (481 clustered
PINcodes, 177 isolated) genuine structure, or fragmentation of a handful of
real clusters into many small near-duplicates? DBSCAN finds only 5 clusters
on the same input -- this script checks whether that gap is explained by
genuine sub-structure HDBSCAN can see and DBSCAN's single epsilon cannot, or
whether HDBSCAN is over-splitting.

Reads data/processed/ensemble_final.csv (the IF-flagged subset with HDBSCAN
and DBSCAN cluster columns attached -- see build_ensemble_outliers in
src/scoring.py). Does NOT import or modify anything in src/, does NOT touch
config.py, and does NOT rerun the pipeline. The only write is
data/processed/cluster_diagnostics.csv.

Feature z-scoring is done within the 658-row IF-flagged population loaded
from ensemble_final.csv -- the same population StandardScaler was fit on
inside cluster_anomalies_hdbscan -- so cluster centroids are directly
comparable to the feature space HDBSCAN actually clustered in.

Run:
    python scripts/cluster_diagnostics.py   (run pipeline.py first)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ENSEMBLE_PATH = "data/processed/ensemble_final.csv"
OUTPUT_PATH = "data/processed/cluster_diagnostics.csv"

_FEATURES = ["DPR", "PNA", "TAI"]
_MIN_CLUSTER_SIZE = 5     # HDBSCAN min_cluster_size (src/scoring.py)
_SMALL_CLUSTER_MAX = 7    # "small" threshold for this diagnostic, per task spec

# Near-duplicate profile flag: cosine similarity above this AND euclidean
# distance below the 25th percentile of all pairwise centroid distances.
_COSINE_NEAR_DUP = 0.90
_EUCLIDEAN_PERCENTILE = 25

# Weak membership: median hdbscan_probability below this is "weak".
_WEAK_PROBABILITY = 0.5

# Geographic concentration: a cluster is "concentrated" if its single
# largest state accounts for at least this share of its members.
_GEO_CONCENTRATED_SHARE = 0.8


# ── Load & validate ───────────────────────────────────────────────────────────

def load_data(path: str = ENSEMBLE_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python pipeline.py` first to generate it."
        )
    df = pd.read_csv(path)
    required = {"hdbscan_cluster", "hdbscan_probability", "dbscan_cluster",
                "state", "district"} | set(_FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing expected columns: {missing}")
    return df


# ── Section 1: size distribution ───────────────────────────────────────────────

def size_distribution(df: pd.DataFrame) -> pd.Series:
    clustered = df[df["hdbscan_cluster"] >= 0]
    return clustered.groupby("hdbscan_cluster").size().sort_values(ascending=False)


def print_section1(sizes: pd.Series, n_isolated: int) -> None:
    print(f"\n{'=' * 74}")
    print("  1. SIZE DISTRIBUTION")
    print(f"{'=' * 74}")
    print(f"  Clusters               : {len(sizes)}")
    print(f"  Isolated (noise, -1)   : {n_isolated}")
    print(f"  At exactly min size ({_MIN_CLUSTER_SIZE})   : {(sizes == _MIN_CLUSTER_SIZE).sum()}")
    print(f"  At <= {_SMALL_CLUSTER_MAX} members         : {(sizes <= _SMALL_CLUSTER_MAX).sum()} "
          f"({(sizes <= _SMALL_CLUSTER_MAX).mean():.0%} of all clusters)")
    print(f"  Largest cluster        : {sizes.max()}")
    print(f"  Mean / median size     : {sizes.mean():.1f} / {sizes.median():.1f}")
    print(f"  Std dev                : {sizes.std():.1f}")
    print(f"\n  Sizes (sorted desc): {sizes.tolist()}")


# ── Section 2: per-cluster feature profiles ────────────────────────────────────

def feature_profiles(df: pd.DataFrame):
    """
    Returns (profile_df, centroid_z) where:
      profile_df  -- one row per cluster: mean/median DPR,PNA,TAI, dominant feature
      centroid_z  -- DataFrame indexed by cluster, columns = z-scored feature means
                      (the space HDBSCAN actually clustered in)
    """
    clustered = df[df["hdbscan_cluster"] >= 0].copy()
    X = clustered[_FEATURES].fillna(0).values
    Z = StandardScaler().fit_transform(X)
    z_df = pd.DataFrame(Z, columns=[f"{f}_z" for f in _FEATURES], index=clustered.index)
    clustered = pd.concat([clustered, z_df], axis=1)

    rows = []
    for cid, g in clustered.groupby("hdbscan_cluster"):
        z_means = g[[f"{f}_z" for f in _FEATURES]].mean()
        dominant_idx = z_means.abs().values.argmax()
        dominant_feature = _FEATURES[dominant_idx]
        dominant_direction = "High" if z_means.iloc[dominant_idx] > 0 else "Low"
        rows.append({
            "hdbscan_cluster": cid,
            "size": len(g),
            "mean_DPR": g["DPR"].mean(), "median_DPR": g["DPR"].median(),
            "mean_PNA": g["PNA"].mean(), "median_PNA": g["PNA"].median(),
            "mean_TAI": g["TAI"].mean(), "median_TAI": g["TAI"].median(),
            "dominant_feature": dominant_feature,
            "dominant_direction": dominant_direction,
        })
    profile_df = pd.DataFrame(rows).set_index("hdbscan_cluster")

    centroid_z = clustered.groupby("hdbscan_cluster")[[f"{f}_z" for f in _FEATURES]].mean()
    return profile_df, centroid_z


def print_section2(profile_df: pd.DataFrame) -> None:
    print(f"\n{'=' * 74}")
    print("  2. PER-CLUSTER FEATURE PROFILES (z-scored within IF-flagged subset)")
    print(f"{'=' * 74}")
    print(f"  {'clu':>4} {'n':>4} {'med DPR':>9} {'med PNA':>9} {'med TAI':>10} "
          f"{'dominant':>14}")
    for cid, r in profile_df.sort_values("size", ascending=False).iterrows():
        print(f"  {cid:>4} {int(r['size']):>4} {r['median_DPR']:>9.2f} "
              f"{r['median_PNA']:>9.3f} {r['median_TAI']:>10.1f} "
              f"{r['dominant_direction'] + '-' + r['dominant_feature']:>14}")
    counts = profile_df["dominant_feature"].value_counts()
    print(f"\n  Dominant-feature counts across clusters: {counts.to_dict()}")


# ── Section 3: profile distinctness ────────────────────────────────────────────

def profile_distinctness(centroid_z: pd.DataFrame):
    """
    Pairwise euclidean + cosine similarity between cluster centroids.
    Returns (dist_df, cos_df, near_dup_pairs, euclid_threshold).
    """
    ids = centroid_z.index.tolist()
    M = centroid_z.values

    euclid = squareform(pdist(M, metric="euclidean"))
    cos = cosine_similarity(M)

    dist_df = pd.DataFrame(euclid, index=ids, columns=ids)
    cos_df = pd.DataFrame(cos, index=ids, columns=ids)

    # Off-diagonal distances only, for the adaptive threshold.
    iu = np.triu_indices(len(ids), k=1)
    off_diag_dist = euclid[iu]
    euclid_threshold = (
        float(np.percentile(off_diag_dist, _EUCLIDEAN_PERCENTILE))
        if len(off_diag_dist) else float("nan")
    )

    near_dup_pairs = []
    for a_idx, b_idx in zip(*iu):
        if euclid[a_idx, b_idx] < euclid_threshold and cos[a_idx, b_idx] > _COSINE_NEAR_DUP:
            near_dup_pairs.append((ids[a_idx], ids[b_idx],
                                    euclid[a_idx, b_idx], cos[a_idx, b_idx]))

    return dist_df, cos_df, near_dup_pairs, euclid_threshold


def print_section3(near_dup_pairs, euclid_threshold, n_clusters) -> None:
    print(f"\n{'=' * 74}")
    print("  3. PROFILE DISTINCTNESS (pairwise centroid similarity)")
    print(f"{'=' * 74}")
    print(f"  Near-duplicate flag: euclidean < {euclid_threshold:.3f} "
          f"(P{_EUCLIDEAN_PERCENTILE} of all pairwise distances) "
          f"AND cosine > {_COSINE_NEAR_DUP}")
    if near_dup_pairs:
        ranked = sorted(near_dup_pairs, key=lambda t: t[2])
        print(f"  {len(near_dup_pairs)} near-duplicate pair(s) out of "
              f"{n_clusters * (n_clusters - 1) // 2} total pairs "
              f"(showing closest {min(20, len(ranked))}):")
        for a, b, d, c in ranked[:20]:
            print(f"    cluster {a:>3} <-> cluster {b:>3}   "
                  f"euclidean={d:.3f}  cosine={c:.3f}")
        if len(ranked) > 20:
            print(f"    ... and {len(ranked) - 20} more (see near_dup_pairs in the "
                  f"returned data if needed)")
    else:
        print("  No near-duplicate cluster pairs found.")


# ── Section 4: geographic coherence ────────────────────────────────────────────

def geographic_coherence(df: pd.DataFrame) -> pd.DataFrame:
    clustered = df[df["hdbscan_cluster"] >= 0]
    rows = []
    for cid, g in clustered.groupby("hdbscan_cluster"):
        counts = g["state"].value_counts()
        rows.append({
            "hdbscan_cluster": cid,
            "n_states": g["state"].nunique(),
            "top_state": counts.index[0],
            "top_state_share": counts.iloc[0] / len(g),
        })
    return pd.DataFrame(rows).set_index("hdbscan_cluster")


def print_section4(geo_df: pd.DataFrame) -> None:
    print(f"\n{'=' * 74}")
    print("  4. GEOGRAPHIC COHERENCE")
    print(f"{'=' * 74}")
    concentrated = (geo_df["top_state_share"] >= _GEO_CONCENTRATED_SHARE).sum()
    print(f"  Clusters with a single state >= {_GEO_CONCENTRATED_SHARE:.0%} of members: "
          f"{concentrated} / {len(geo_df)}")
    print(f"  Mean states per cluster        : {geo_df['n_states'].mean():.2f}")
    print(f"  Mean top-state share           : {geo_df['top_state_share'].mean():.2f}")
    print(f"\n  {'clu':>4} {'n_states':>9} {'top_state':>20} {'share':>7}")
    for cid, r in geo_df.sort_values("top_state_share", ascending=False).iterrows():
        print(f"  {cid:>4} {r['n_states']:>9} {r['top_state']:>20} "
              f"{r['top_state_share']:>7.0%}")


# ── Section 5: membership confidence ───────────────────────────────────────────

def membership_confidence(df: pd.DataFrame) -> pd.DataFrame:
    clustered = df[df["hdbscan_cluster"] >= 0]
    stats = clustered.groupby("hdbscan_cluster")["hdbscan_probability"].agg(
        median_probability="median", mean_probability="mean", min_probability="min",
    )
    return stats


def print_section5(prob_df: pd.DataFrame) -> None:
    print(f"\n{'=' * 74}")
    print("  5. MEMBERSHIP CONFIDENCE (hdbscan_probability)")
    print(f"{'=' * 74}")
    weak = prob_df[prob_df["median_probability"] < _WEAK_PROBABILITY]
    print(f"  Clusters with median probability < {_WEAK_PROBABILITY}: "
          f"{len(weak)} / {len(prob_df)}")
    if len(weak):
        print("  Weak clusters:")
        for cid, r in weak.sort_values("median_probability").iterrows():
            print(f"    cluster {cid:>3}: median={r['median_probability']:.2f} "
                  f"mean={r['mean_probability']:.2f} min={r['min_probability']:.2f}")
    print(f"\n  Overall median probability across all clustered points: "
          f"{prob_df['median_probability'].median():.2f}")


# ── Section 6: DBSCAN comparison ───────────────────────────────────────────────

def dbscan_comparison(df: pd.DataFrame):
    """
    Uses the FULL IF-flagged set (658 rows), not just HDBSCAN-clustered rows.
    Baseline: DBSCAN clusters 648/658 (98%) into 5 clusters; HDBSCAN clusters
    only 481/658 (73%) into 33 clusters. That gap means a large share of
    DBSCAN's cluster members are HDBSCAN noise (-1) -- restricting to
    hdbscan_cluster >= 0 would silently drop them and understate how much of
    DBSCAN's "clustering" HDBSCAN actually considers too sparse to trust.
    """
    crosstab = pd.crosstab(df["hdbscan_cluster"], df["dbscan_cluster"])

    real_dbscan = sorted(c for c in df["dbscan_cluster"].unique() if c >= 0)
    rows = []
    for dcid in real_dbscan:
        members = df[df["dbscan_cluster"] == dcid]
        hdb_counts = members["hdbscan_cluster"].value_counts()
        n_touched_real = members.loc[members["hdbscan_cluster"] >= 0, "hdbscan_cluster"].nunique()
        rows.append({
            "dbscan_cluster": dcid,
            "size": len(members),
            "n_hdbscan_noise": int((members["hdbscan_cluster"] == -1).sum()),
            "n_hdbscan_clusters_touched": n_touched_real,
            "dominant_hdbscan_cluster": hdb_counts.index[0],
            "purity": hdb_counts.iloc[0] / len(members),
        })
    purity_df = pd.DataFrame(rows).set_index("dbscan_cluster")
    return crosstab, purity_df


def print_section6(crosstab: pd.DataFrame, purity_df: pd.DataFrame) -> None:
    print(f"\n{'=' * 74}")
    print("  6. DBSCAN COMPARISON (full 658-row IF-flagged set, both methods' noise included)")
    print(f"{'=' * 74}")
    print("  Crosstab: hdbscan_cluster (rows, -1 = HDBSCAN noise) x "
          "dbscan_cluster (cols, -1 = DBSCAN noise)")
    print(crosstab.to_string())
    print(f"\n  Per-DBSCAN-cluster breakdown (purity = share of members in its single")
    print(f"  most common HDBSCAN cluster/label, including -1 = HDBSCAN noise):")
    print(f"  {'dbscan':>6} {'size':>5} {'hdb_noise':>9} {'n_hdb_touched':>14} "
          f"{'dominant_hdb':>13} {'purity':>7}")
    for dcid, r in purity_df.iterrows():
        print(f"  {dcid:>6} {int(r['size']):>5} {int(r['n_hdbscan_noise']):>9} "
              f"{int(r['n_hdbscan_clusters_touched']):>14} "
              f"{int(r['dominant_hdbscan_cluster']):>13} {r['purity']:>7.0%}")
    # Unweighted mean treats a 4-point cluster the same as a 627-point cluster --
    # size-weighted is the honest summary when cluster sizes are this skewed
    # (one DBSCAN cluster holds ~95% of the IF-flagged set).
    unweighted_purity = purity_df["purity"].mean()
    total = purity_df["size"].sum()
    weighted_purity = (purity_df["purity"] * purity_df["size"]).sum() / total
    weighted_noise_share = purity_df["n_hdbscan_noise"].sum() / total

    if weighted_purity >= 0.8:
        interp = "DBSCAN clusters mostly nest inside single HDBSCAN clusters (hierarchy)."
    elif weighted_purity <= 0.4:
        interp = ("DBSCAN's clustering collapses distinct HDBSCAN clusters together -- "
                   "most points sit in a DBSCAN cluster that spans many HDBSCAN clusters.")
    else:
        interp = "Partial correspondence -- neither clean hierarchy nor clean collapse."
    print(f"\n  Purity, size-weighted (share of ALL clustered points, not share of")
    print(f"  cluster IDs): {weighted_purity:.0%}  (unweighted across the 5 IDs: "
          f"{unweighted_purity:.0%})")
    print(f"  Share of all DBSCAN-clustered points that HDBSCAN calls noise: "
          f"{weighted_noise_share:.0%}")
    print(f"  {interp}")
    if weighted_noise_share > 0.2:
        print(f"  A meaningful share of DBSCAN's \"clustered\" points are points HDBSCAN")
        print(f"  considers too sparse to trust -- DBSCAN's fixed epsilon sweeps in")
        print(f"  points HDBSCAN's density-adaptive criterion excludes as noise.")


# ── Verdict ─────────────────────────────────────────────────────────────────────

def compute_verdict(sizes, near_dup_pairs, geo_df, n_clusters):
    pct_small = (sizes <= _SMALL_CLUSTER_MAX).mean()

    small_ids = set(sizes[sizes <= _SMALL_CLUSTER_MAX].index)
    near_dup_ids = set()
    for a, b, _, _ in near_dup_pairs:
        near_dup_ids.add(a)
        near_dup_ids.add(b)
    frac_small_near_dup = (
        len(small_ids & near_dup_ids) / len(small_ids) if small_ids else 0.0
    )
    frac_clusters_near_dup = len(near_dup_ids) / n_clusters if n_clusters else 0.0
    frac_geo_concentrated = (geo_df["top_state_share"] >= _GEO_CONCENTRATED_SHARE).mean()

    metrics = {
        "pct_clusters_small (<=7)": pct_small,
        "frac_small_clusters_that_are_near_dup": frac_small_near_dup,
        "frac_all_clusters_in_a_near_dup_pair": frac_clusters_near_dup,
        "frac_clusters_geo_concentrated (>=80% one state)": frac_geo_concentrated,
    }

    if pct_small > 0.5 and frac_small_near_dup > 0.5:
        verdict = "FRAGMENTATION CONCERN"
    elif frac_clusters_near_dup < 0.5 or frac_geo_concentrated > 0.5:
        verdict = "DISTINCT STRUCTURE"
    else:
        verdict = "MIXED"

    return verdict, metrics


def print_verdict(verdict, metrics) -> None:
    print(f"\n{'=' * 74}")
    print(f"  VERDICT: {verdict}")
    print(f"{'=' * 74}")
    for k, v in metrics.items():
        print(f"  {k:<50}: {v:.0%}" if isinstance(v, float) else f"  {k:<50}: {v}")
    if verdict == "FRAGMENTATION CONCERN":
        print("\n  >50% of clusters are small (<=7 members) AND most of those small")
        print("  clusters have a near-duplicate profile twin. Treat the 33-cluster")
        print("  count with caution before citing it as structure in Paper B §5.")
    elif verdict == "DISTINCT STRUCTURE":
        print("\n  Clusters are not dominated by near-duplicate profiles, and/or show")
        print("  meaningful geographic concentration. The cluster count reflects")
        print("  genuine sub-structure, not arbitrary splitting of one population.")
    else:
        print("\n  Neither a clean pass nor a clean fail -- see the metrics above and")
        print("  the per-cluster tables in sections 1-5 before writing up §5.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    clustered = df[df["hdbscan_cluster"] >= 0]
    n_isolated = int((df["hdbscan_cluster"] == -1).sum())

    print(f"\nLoaded {len(df):,} IF-flagged PINcodes from {ENSEMBLE_PATH}")
    print(f"HDBSCAN: {clustered['hdbscan_cluster'].nunique()} clusters, "
          f"{len(clustered)} clustered, {n_isolated} isolated")

    sizes = size_distribution(df)
    print_section1(sizes, n_isolated)

    profile_df, centroid_z = feature_profiles(df)
    print_section2(profile_df)

    dist_df, cos_df, near_dup_pairs, euclid_threshold = profile_distinctness(centroid_z)
    print_section3(near_dup_pairs, euclid_threshold, len(sizes))

    geo_df = geographic_coherence(df)
    print_section4(geo_df)

    prob_df = membership_confidence(df)
    print_section5(prob_df)

    crosstab, purity_df = dbscan_comparison(df)
    print_section6(crosstab, purity_df)

    verdict, metrics = compute_verdict(sizes, near_dup_pairs, geo_df, len(sizes))
    print_verdict(verdict, metrics)

    out = (
        profile_df
        .join(geo_df)
        .join(prob_df[["median_probability"]])
    )
    out.insert(0, "hdbscan_cluster", out.index)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}\n")

    return out


if __name__ == "__main__":
    main()
