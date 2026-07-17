"""
scripts/figures.py
══════════════════════════════════════════════════════════════════════════════
Paper B figures 2-6 (Figure 1, the pipeline diagram, is done separately in
TikZ -- not this script's job).

Read-only w.r.t. the pipeline: does not import or modify anything in src/,
does not touch config.py, does not rerun pipeline.py. Reads existing
data/processed/*.csv and writes only outputs/figures/*.pdf.

DATA-QUALITY FLAG (found while building this script, not assumed):
  CLAUDE.md and the original figure spec both name "PINcode 400003,
  AHMADNAGAR" as the DPR=193 headline anomaly. data/processed/sentinel_final.csv
  has exactly one row with DPR == 193.0, and it is PINcode 414001 (AHMADNAGAR,
  MAHARASHTRA) -- pincode 400003 exists (4 rows, spanning AHMADNAGAR AND
  MUMBAI districts) but tops out at DPR=1.13. This script annotates the
  empirically-verified row (414001). Flagged to the user; the "400003" claim
  in CLAUDE.md may be stale and worth checking before it goes into the paper.

ROW IDENTITY: pincode is not unique (spans districts). Verified while
building this: (pincode, district, state) is ALSO not unique in this dataset
(26,299 unique triples of 32,898 rows) -- a naive merge on that key inflates
row counts (empirically: merging outliers_stat.csv into sentinel_final.csv on
that triple turns 1,918 rows into 2,403 matches). To avoid that trap
entirely, every figure below derives Statistical/IF/LOF membership as
row-level boolean columns computed directly on sentinel_final.csv (which
carries anomaly_score and lof_flag for all 32,898 rows), rather than joining
files by key. fig3 additionally verifies its recomputed Statistical mask
against outliers_stat.csv by exact multiset comparison of value tuples (not
a key-based join) -- see verify_statistical_mask().

Stat-IF OVERLAP DISCREPANCY: pipeline.py's own console summary computes
"Stat-IF overlap" via PINCODE-SET intersection
(set(outliers_stat.pincode) & set(if_outliers.pincode)) = 656/658. That
undercounts because it dedupes by pincode alone -- exactly the trap this
task's spec warns against. This script's row-level computation (fig3) finds
Stat∩IF = 658/658: every single IF-flagged ROW is also statistically
flagged. IF∩LOF = 54 matches the documented baseline under both methods,
because scoring.py's own overlap stat already uses the pandas row index, not
pincode. Both numbers are printed for cross-reference.

Run:
    python scripts/figures.py   (run pipeline.py first; needs data/processed/*)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

PROCESSED = Path(config.PROCESSED_DIR)
OUT_DIR = Path("outputs/figures")

# ── Okabe-Ito colorblind-safe palette ──────────────────────────────────────
OI = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}
GRAY = "#B0B0B0"   # neutral "background/unflagged" -- not a compared category

# Consistent color roles used across all 5 figures
C_IF, C_LOF, C_BOTH, C_STAT = OI["blue"], OI["vermillion"], OI["green"], OI["orange"]

# ── Global style, set once ────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,      # embed real (Type 1/TrueType) fonts, not Type 3
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SINGLE_COL = 3.5    # inches
DOUBLE_COL = 7.16   # inches
RASTER_DPI = 300
RNG_SEED = 42

DPR193_PINCODE = 414001
DPR193_DISTRICT = "AHMADNAGAR"
DPR193_STATE = "MAHARASHTRA"


# ── Shared helpers ────────────────────────────────────────────────────────

def clip_for_log(s: pd.Series) -> tuple:
    """Clip non-positive values to half the minimum positive value (per-axis).
    Returns (clipped_series, clip_value)."""
    pos = s[s > 0]
    clip_val = float(pos.min() / 2) if len(pos) else float("nan")
    return s.where(s > 0, clip_val), clip_val


def find_dpr193_row(sf: pd.DataFrame) -> pd.Series:
    hot = sf[(sf["pincode"] == DPR193_PINCODE) & (sf["district"] == DPR193_DISTRICT)
             & (sf["DPR"] == 193.0)]
    if len(hot) != 1:
        raise ValueError(
            f"Expected exactly one DPR=193 row at pincode {DPR193_PINCODE}, "
            f"found {len(hot)}."
        )
    return hot.iloc[0]


# ── fig2: DPR-PNA scatter, full 32,898 rows ────────────────────────────────

def fig2_dpr_pna_scatter(sf: pd.DataFrame) -> None:
    print("\n--- fig2_dpr_pna_scatter ---")
    if_mask = sf["anomaly_score"] == -1
    lof_mask = sf["lof_flag"].astype(bool)
    both = if_mask & lof_mask
    if_only = if_mask & ~lof_mask
    lof_only = ~if_mask & lof_mask
    unflagged = ~if_mask & ~lof_mask
    assert int(if_only.sum() + lof_only.sum() + both.sum() + unflagged.sum()) == len(sf)

    print(f"Rows: {len(sf):,} | IF={int(if_mask.sum())} LOF={int(lof_mask.sum())} | "
          f"IF-only={int(if_only.sum())} LOF-only={int(lof_only.sum())} "
          f"IF∩LOF={int(both.sum())} unflagged={int(unflagged.sum())}")

    pna_c, pna_clip = clip_for_log(sf["PNA"])
    dpr_c, dpr_clip = clip_for_log(sf["DPR"])
    print(f"Zero-clip (half min positive): PNA -> {pna_clip:.6g}, DPR -> {dpr_clip:.6g}")

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.6))

    layers = [
        (unflagged, GRAY, 3, 0.3, f"Unflagged (n={int(unflagged.sum()):,})"),
        (if_only, C_IF, 5, 0.6, f"IF only (n={int(if_only.sum()):,})"),
        (lof_only, C_LOF, 5, 0.6, f"LOF only (n={int(lof_only.sum()):,})"),
        (both, C_BOTH, 9, 0.9, f"IF ∩ LOF (n={int(both.sum()):,})"),
    ]
    for mask, color, size, alpha, label in layers:
        ax.scatter(pna_c[mask], dpr_c[mask], s=size, c=color, alpha=alpha,
                   linewidths=0, label=label, rasterized=True)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PNA (log scale; zeros clipped to half-min-positive)")
    ax.set_ylabel("DPR (log scale; zeros clipped to half-min-positive)")

    hot = find_dpr193_row(sf)
    ax.annotate(
        "DPR=193 (Ahilyanagar)",
        xy=(hot["PNA"], hot["DPR"]), xytext=(1.0, 150),
        fontsize=6.5, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=OI["black"], lw=0.7),
    )
    print(f"Annotated pincode {int(hot['pincode'])}: PNA={hot['PNA']:.6g}, DPR={hot['DPR']:.1f}")

    ax.legend(loc="upper left", frameon=False, markerscale=2.2, handletextpad=0.4)
    out = OUT_DIR / "fig2_dpr_pna_scatter.pdf"
    fig.savefig(out, dpi=RASTER_DPI)
    plt.close(fig)
    print(f"Saved -> {out}")


# ── fig3: three-way flag overlap (manual Venn) ─────────────────────────────

def verify_statistical_mask(sf: pd.DataFrame, stat_mask: pd.Series, stat_csv: pd.DataFrame) -> None:
    """
    Exact multiset comparison (not a key-based join, since no key here is
    guaranteed unique) between the rows my recomputed mask selects and the
    rows actually saved in outliers_stat.csv.
    """
    cols = ["pincode", "district", "state", "DPR", "PNA", "TAI"]
    mine = sorted(map(tuple, sf.loc[stat_mask, cols].values.tolist()))
    theirs = sorted(map(tuple, stat_csv[cols].values.tolist()))
    match = mine == theirs
    print(f"Recomputed Statistical mask vs outliers_stat.csv: "
          f"{'EXACT MATCH' if match else 'MISMATCH'} "
          f"({len(mine)} recomputed vs {len(theirs)} on disk)")
    if not match:
        print("  WARNING: recomputed statistical flags diverge from outliers_stat.csv -- "
              "investigate before trusting fig3.")


def fig3_flag_overlap(sf: pd.DataFrame, stat_csv: pd.DataFrame) -> None:
    print("\n--- fig3_flag_overlap ---")
    tai_t = sf["TAI"].quantile(config.OUTLIER_PERCENTILE)
    dpr_t = sf["DPR"].quantile(config.OUTLIER_PERCENTILE)
    pna_t = sf["PNA"].quantile(config.OUTLIER_PERCENTILE)
    stat = (sf["TAI"] >= tai_t) | (sf["DPR"] >= dpr_t) | (sf["PNA"] >= pna_t)
    verify_statistical_mask(sf, stat, stat_csv)

    if_ = sf["anomaly_score"] == -1
    lof = sf["lof_flag"].astype(bool)

    regions = {
        "Stat only":     int((stat & ~if_ & ~lof).sum()),
        "IF only":       int((~stat & if_ & ~lof).sum()),
        "LOF only":      int((~stat & ~if_ & lof).sum()),
        "Stat∩IF only":  int((stat & if_ & ~lof).sum()),
        "Stat∩LOF only": int((stat & ~if_ & lof).sum()),
        "IF∩LOF only":   int((~stat & if_ & lof).sum()),
        "Stat∩IF∩LOF":   int((stat & if_ & lof).sum()),
    }
    none_of = int((~stat & ~if_ & ~lof).sum())
    assert sum(regions.values()) + none_of == len(sf)

    print(f"Set sizes: Statistical={int(stat.sum())} (expect 1918), "
          f"IF={int(if_.sum())} (658), LOF={int(lof.sum())} (623)")
    print("7 exclusive regions:", regions)
    print(f"Pairwise totals -- Stat∩IF: {int((stat & if_).sum())} "
          f"(row-level; NOTE pipeline.py's own console stat reports 656 via "
          f"pincode-set dedup, see module docstring) | "
          f"IF∩LOF: {int((if_ & lof).sum())} (expect 54) | "
          f"Stat∩LOF: {int((stat & lof).sum())}")

    # -- Euler diagram reflecting verified containment (IF ⊆ Stat) ------------
    # Confirmed above: "IF only" and "IF∩LOF only" (both outside Stat) are
    # always 0 here -- IF is fully contained in Stat. Drawing IF nested inside
    # Stat (rather than a symmetric 3-circle Venn with two empty regions) is
    # only valid because of that containment; assert it before trusting the
    # geometry below.
    assert regions["IF only"] == 0 and regions["IF∩LOF only"] == 0, (
        "IF is not fully contained in Stat -- the nested-circle Euler layout "
        "below assumes containment and would misrepresent the data."
    )
    euler = {
        "Stat only":            regions["Stat only"],       # 1181
        "IF (outside LOF)":     regions["Stat∩IF only"],     # 604
        "IF∩LOF":               regions["Stat∩IF∩LOF"],      # 54
        "Stat∩LOF (outside IF)": regions["Stat∩LOF only"],   # 79
        "LOF only":             regions["LOF only"],         # 490
    }
    sizes = {"Stat": int(stat.sum()), "IF": int(if_.sum()), "LOF": int(lof.sum())}
    assert euler["Stat only"] + euler["IF (outside LOF)"] + euler["IF∩LOF"] \
        + euler["Stat∩LOF (outside IF)"] == sizes["Stat"]
    assert euler["IF (outside LOF)"] + euler["IF∩LOF"] == sizes["IF"]
    assert euler["IF∩LOF"] + euler["Stat∩LOF (outside IF)"] + euler["LOF only"] == sizes["LOF"]
    print(f"Euler region arithmetic verified: sums to Stat={sizes['Stat']}, "
          f"IF={sizes['IF']}, LOF={sizes['LOF']}")

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.3))
    max_r = 1.25
    scale = max_r / np.sqrt(max(sizes.values()))
    r = {k: scale * np.sqrt(v) for k, v in sizes.items()}

    # IF nested inside Stat; LOF mostly outside, clipping both Stat's edge
    # and part of IF -- approximate positioning (areas are not to scale;
    # counts carry the precise values).
    centers = {"Stat": (0.0, 0.0), "IF": (0.4, 0.0), "LOF": (1.55, 0.0)}
    colors = {"Stat": C_STAT, "IF": C_IF, "LOF": C_LOF}
    for key in ("Stat", "LOF", "IF"):
        ax.add_patch(plt.Circle(centers[key], r[key], color=colors[key], alpha=0.35,
                                lw=1.2, ec=colors[key]))

    count_pos = {
        "Stat only":             (-0.65, 0.0),
        "IF (outside LOF)":      (0.05, 0.35),
        "IF∩LOF":                (0.95, 0.30),
        # Geometric centroid of the Stat & LOF & ~IF region (computed by
        # sampling the actual circle equations, not eyeballed) -- the lens
        # is narrow (x in [0.99, 1.25]) so off-centroid placement lands the
        # label right on the IF boundary stroke.
        "Stat∩LOF (outside IF)": (1.14, 0.0),
        "LOF only":              (1.95, 0.15),
    }
    for key, (x, y) in count_pos.items():
        ax.text(x, y, str(euler[key]), fontsize=6.5, ha="center", va="center",
               fontweight="bold")

    set_label_pos = {"Stat": (-0.85, 1.35), "IF": (0.4, -0.85)}
    for key, (x, y) in set_label_pos.items():
        ax.text(x, y, f"{key} (n={sizes[key]:,})", fontsize=6.5, ha="center",
               color=colors[key], fontweight="bold")
    # LOF label: left-anchored just past the circle's right edge (2.26) so
    # the text extends rightward, away from the circle, instead of a
    # center-anchored label whose left half bled back into the circle.
    ax.text(2.32, 0.0, f"LOF (n={sizes['LOF']:,})", fontsize=6.5, ha="left",
           va="center", color=colors["LOF"], fontweight="bold")

    ax.set_xlim(-1.7, 3.6)
    ax.set_ylim(-1.6, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")

    out = OUT_DIR / "fig3_flag_overlap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


# ── fig4: sensitivity sweeps ────────────────────────────────────────────────

def fig4_sensitivity(sens: pd.DataFrame) -> None:
    print("\n--- fig4_sensitivity ---")
    # 3 panels stacked vertically, single column. The earlier 3-across layout
    # at 3.5in was too cramped (oversized fonts, truncated y-label, colliding
    # x-labels); stacking gives each panel full column width.
    fig, axes = plt.subplots(3, 1, figsize=(SINGLE_COL, 4.6))
    fig.subplots_adjust(hspace=0.45)
    panels = [
        ("IF contamination", "IF contamination", "(a)"),
        ("LOF n_neighbors", "LOF n_neighbors", "(b)"),
        ("DPR weight", "DPR weight", "(c)"),
    ]
    # Tag corner verified against each curve's shape (values checked from
    # sensitivity_results.csv, not guessed):
    #  (a) IF contamination is flat at Jaccard=1.0 -- the curve hugs the top,
    #      so the lower-left is empty; placed above the "baseline" annotation
    #      that sits at the very bottom near x=0.02.
    #  (b) LOF (edges 0.14 / 0.08) and (c) DPR (edges 0.54 / 0.67) both peak
    #      at their centred baseline and fall off toward the edges, leaving
    #      the top-left corner well clear of the curve.
    tag_pos = {
        "(a)": (0.04, 0.28, "left", "bottom"),
        "(b)": (0.04, 0.94, "left", "top"),
        "(c)": (0.04, 0.94, "left", "top"),
    }
    for (sweep_name, xlabel, tag), ax in zip(panels, axes):
        sub = sens[sens["sweep"] == sweep_name].sort_values("value")
        if not len(sub):
            print(f"WARNING: no rows for sweep={sweep_name!r}")
            continue
        print(f"{sweep_name}: {len(sub)} rows, jaccard range "
              f"[{sub['jaccard_vs_baseline'].min():.3f}, {sub['jaccard_vs_baseline'].max():.3f}]")

        ax.plot(sub["value"], sub["jaccard_vs_baseline"], "-o", color=OI["blue"],
               markersize=3, linewidth=1)
        ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=0.8, zorder=0)

        base_row = sub[sub["is_baseline"]]
        if len(base_row):
            bv = float(base_row.iloc[0]["value"])
            ax.axvline(bv, color=OI["vermillion"], linestyle="--", linewidth=0.8)
            ax.annotate("baseline", xy=(bv, 0.03), fontsize=6, color=OI["vermillion"],
                       ha="center", va="bottom")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])

        x, y, ha, va = tag_pos[tag]
        ax.text(x, y, tag, transform=ax.transAxes, fontsize=8, fontweight="bold",
               ha=ha, va=va)

    fig.supylabel("Jaccard similarity (top-20 vs. baseline)", fontsize=8)

    out = OUT_DIR / "fig4_sensitivity.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


# ── fig5: RGI correction-factor distribution ────────────────────────────────

def fig5_rgi_shift(sf: pd.DataFrame) -> None:
    print("\n--- fig5_rgi_shift ---")
    years = config.TARGET_YEAR - config.BASE_YEAR
    print(f"Projection horizon (from src/config.py: BASE_YEAR={config.BASE_YEAR}, "
          f"TARGET_YEAR={config.TARGET_YEAR}): {years} years")

    excluded = sf["growth_source"] == "no_census_match"
    print(f"Excluding growth_source == 'no_census_match': {int(excluded.sum())} rows "
          f"of {len(sf)}")
    d = sf.loc[~excluded].copy()
    d["f"] = (1.0 + d["annual_growth_rate"]) ** years
    d["PNA_static"] = d["PNA"] * d["f"]

    mean_f = float(d["f"].mean())
    state_f = d.groupby("state")["f"].mean()
    lowest_state = state_f.idxmin()
    lowest_f = float(state_f.min())
    delhi_f = float(state_f.get("NCT OF DELHI", float("nan")))
    ut_f = state_f.loc[state_f.index.intersection(config.RGI_HIGH_GROWTH_UTS)].sort_values()

    print(f"Correction factor f=(1+r)^{years}: national mean={mean_f:.4f}, "
          f"range=[{d['f'].min():.4f}, {d['f'].max():.4f}]")
    print(f"Lowest-growth state: {lowest_state.title()} (f={lowest_f:.4f})")
    print(f"NCT OF DELHI: f={delhi_f:.4f}")
    print("Whitelisted high-growth UTs (config.RGI_HIGH_GROWTH_UTS): "
          + ", ".join(f"{k.title()}={v:.4f}" for k, v in ut_f.items()))

    # -- PNA-only 98th-percentile flags, current vs static (printed, not plotted) --
    thr_cur = d["PNA"].quantile(config.OUTLIER_PERCENTILE)
    thr_static = d["PNA_static"].quantile(config.OUTLIER_PERCENTILE)
    flag_cur = d["PNA"] >= thr_cur
    flag_static = d["PNA_static"] >= thr_static
    n_cur, n_static = int(flag_cur.sum()), int(flag_static.sum())
    sym_diff = int((flag_cur != flag_static).sum())
    print(f"PNA-component 98th-pct flags -- current: {n_cur} (threshold={thr_cur:.4g}) | "
          f"static: {n_static} (threshold={thr_static:.4g}) | "
          f"symmetric difference: {sym_diff}")

    # -- Plot: distribution of f, linear x, log y (whitelisted UTs are rare) --
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))
    ax.hist(d["f"], bins=70, range=(1.0, float(d["f"].max())), color=GRAY,
           edgecolor="white", linewidth=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Correction factor f = (1 + annual growth rate)$^{15}$")
    ax.set_ylabel("PINcodes (count, log scale)")

    markers = [
        (mean_f, OI["black"], f"mean={mean_f:.2f}"),
        (lowest_f, OI["green"], f"{lowest_state.title()}={lowest_f:.2f}"),
        (delhi_f, OI["purple"], f"Delhi={delhi_f:.2f}"),
    ]
    for name, fval in ut_f.items():
        markers.append((float(fval), OI["vermillion"], f"{name.title()}={fval:.2f}"))

    # mean/Odisha/Delhi/Puducherry all sit within f in [1.0, 1.45] -- sort by
    # x-position and alternate top/bottom vertical anchors so any two labels
    # adjacent in x land in different vertical bands instead of colliding.
    markers.sort(key=lambda m: m[0])
    y_fracs = [0.82, 0.22]

    ymin, ymax = ax.get_ylim()
    for i, (fval, color, label) in enumerate(markers):
        ax.axvline(fval, color=color, linestyle="--", linewidth=0.8)
        y_frac = y_fracs[i % 2]
        va = "top" if y_frac > 0.5 else "bottom"
        y = ymin * (ymax / ymin) ** y_frac
        ax.text(fval, y, label, rotation=90, fontsize=5.5, color=color,
               ha="right", va=va)

    out = OUT_DIR / "fig5_rgi_shift.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


# ── fig6: Ahilyanagar case study + HDBSCAN cluster 1 ────────────────────────

def fig6_case_study(sf: pd.DataFrame, ens: pd.DataFrame) -> None:
    print("\n--- fig6_case_study ---")
    ahm = (
        sf[(sf["district"] == DPR193_DISTRICT) & (sf["state"] == DPR193_STATE)]
        .reset_index(drop=True)
        .copy()
    )
    print(f"Ahilyanagar (AHMADNAGAR) PINcode rows: {len(ahm)}")
    median_dpr = float(ahm["DPR"].median())
    print(f"District median DPR (per-pincode median, n={len(ahm)}): {median_dpr:.4f}")

    hot = ahm[(ahm["pincode"] == DPR193_PINCODE) & (ahm["DPR"] == 193.0)]
    if len(hot) != 1:
        raise ValueError(f"Expected exactly one DPR=193 row in ahm, found {len(hot)}.")
    hidx = hot.index[0]

    rng = np.random.default_rng(RNG_SEED)
    jitter = rng.normal(0, 0.04, len(ahm))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.2),
                                   gridspec_kw={"wspace": 0.3})
    axA.text(0.03, 0.97, "(a)", transform=axA.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="top")
    axB.text(0.03, 0.97, "(b)", transform=axB.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="top")

    dpr_c, dpr_clip = clip_for_log(ahm["DPR"])
    print(f"Panel A zero-clip (half min positive): DPR -> {dpr_clip:.6g}")
    axA.scatter(jitter, dpr_c, s=6, c=GRAY, alpha=0.5, linewidths=0, rasterized=True)
    axA.axhline(median_dpr, color=OI["black"], linestyle="--", linewidth=0.8)
    axA.text(0.47, median_dpr, f"median={median_dpr:.2f}", fontsize=6,
             ha="right", va="bottom")
    axA.scatter([jitter[hidx]], [ahm.loc[hidx, "DPR"]], s=32, c=OI["vermillion"],
               zorder=5, linewidths=0)
    axA.annotate(
        f"PINcode {DPR193_PINCODE}\nDPR={ahm.loc[hidx, 'DPR']:.0f}",
        xy=(jitter[hidx], ahm.loc[hidx, "DPR"]), xytext=(0.30, 35),
        fontsize=6, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", lw=0.7, color=OI["black"]),
    )
    axA.set_yscale("log")
    axA.set_xlim(-0.5, 0.5)
    axA.set_xticks([])
    axA.set_ylabel("DPR (log scale)")
    axA.set_xlabel(f"Ahilyanagar (AHMADNAGAR) PINcodes (n={len(ahm)}, jittered)")

    # Panel B: HDBSCAN cluster 1 over the full IF-flagged set
    c1 = ens[ens["hdbscan_cluster"] == 1]
    state_counts = c1["state"].value_counts().to_dict()
    print(f"HDBSCAN cluster 1: n={len(c1)}, states={state_counts}")

    axB.scatter(ens["TAI"], ens["DPR"], s=4, c=GRAY, alpha=0.4, linewidths=0,
               label=f"IF-flagged set (n={len(ens)})", rasterized=True)
    axB.scatter(c1["TAI"], c1["DPR"], s=22, c=OI["purple"], linewidths=0.3,
               edgecolors=OI["black"], label=f"HDBSCAN cluster 1 (n={len(c1)})", zorder=5)
    # Cluster-1 members sit tightly bunched in (TAI, DPR) space -- per-point text
    # labels collide there regardless of offset, so label the cluster once instead.
    state_summary = ", ".join(
        f"{v}× {k.title()}" for k, v in
        sorted(state_counts.items(), key=lambda kv: -kv[1])
    )
    # Empty region confirmed by 2D density grid over ens (TAI, DPR): the block
    # TAI in [21, 1041] x DPR in [0.08, 2.2] contains zero points.
    axB.annotate(
        state_summary,
        xy=(c1["TAI"].median(), c1["DPR"].median()),
        xytext=(150, 0.4),
        fontsize=6, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", lw=0.7, color=OI["black"]),
    )
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel("TAI (log scale)")
    axB.set_ylabel("DPR (log scale)")
    axB.legend(loc="upper right", frameon=False, fontsize=6)

    out = OUT_DIR / "fig6_case_study.pdf"
    fig.savefig(out, dpi=RASTER_DPI)
    plt.close(fig)
    print(f"Saved -> {out}")


# ── gitignore status check ─────────────────────────────────────────────────

def check_gitignore_status() -> str:
    import subprocess
    repo_root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(OUT_DIR / "probe.pdf")],
            cwd=repo_root,
        )
        return "IGNORED" if result.returncode == 0 else "NOT ignored (git add would track it)"
    except FileNotFoundError:
        return "UNKNOWN (git not available)"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    for name in ("sentinel_final.csv", "outliers_stat.csv", "ensemble_final.csv",
                 "sensitivity_results.csv"):
        if not (PROCESSED / name).exists():
            raise FileNotFoundError(
                f"{PROCESSED / name} not found. Run `python pipeline.py` "
                f"(and scripts/sensitivity.py for sensitivity_results.csv) first."
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed CSVs...")
    sf = pd.read_csv(PROCESSED / "sentinel_final.csv")
    ens = pd.read_csv(PROCESSED / "ensemble_final.csv")
    stat_csv = pd.read_csv(PROCESSED / "outliers_stat.csv")
    sens = pd.read_csv(PROCESSED / "sensitivity_results.csv")
    print(f"sentinel_final: {len(sf):,} | ensemble_final: {len(ens):,} | "
          f"outliers_stat: {len(stat_csv):,} | sensitivity_results: {len(sens):,}")

    fig2_dpr_pna_scatter(sf)
    fig3_flag_overlap(sf, stat_csv)
    fig4_sensitivity(sens)
    fig5_rgi_shift(sf)
    fig6_case_study(sf, ens)

    status = check_gitignore_status()
    print(f"\noutputs/figures/*.pdf gitignore status: {status}")
    print(f"All 5 figures written to {OUT_DIR}/\n")


if __name__ == "__main__":
    main()
