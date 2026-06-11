"""
src/maps.py
══════════════════════════════════════════════════════════════════════════════
Geospatial Visualisation — Aadhaar Sentinel V2

Generates two interactive HTML maps, called by pipeline.py after scoring:

  1. national_map.html  — HeatMap of all PINcodes weighted by audit_priority_score
  2. audit_targets_map.html — Annotated markers for the top-N audit priority PINcodes

Maps use district-level lat/lon centroids from LATLON_PATH, joined on
(district, state) to avoid the same repeated-district-name collision that
plagued V1's single-key joins in engine.py.

Usage in pipeline.py:
    from src.maps import generate_all_maps
    generate_all_maps(scored_df)

Output:
    outputs/maps/national_map.html
    outputs/maps/audit_targets_map.html
"""

import logging
import os
from typing import Optional

import folium
import pandas as pd
from folium.plugins import HeatMap

from src import config

logger = logging.getLogger(__name__)

# India geographic centre — sensible default map origin
_INDIA_LAT = 22.5
_INDIA_LON = 82.0
_DEFAULT_ZOOM = 5

# Top-N audit targets shown on the marker map
_TOP_N_TARGETS = 20


# ── Public entry point ────────────────────────────────────────────────────────

def generate_all_maps(df: pd.DataFrame) -> None:
    """
    Pipeline entry point. Called by pipeline.py after scoring.

    Parameters
    ----------
    df : DataFrame
        Output of compute_risk_score() and add_population_uncertainty().
        Required columns: pincode, district, state, TAI, DPR, PNA, audit_priority_score.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    coords_df = _load_coordinates(config.LATLON_PATH)

    if coords_df is None:
        logger.error(
            "Map generation skipped: coordinates file not found at '%s'.",
            config.LATLON_PATH,
        )
        return

    merged = _merge_coordinates(df, coords_df)

    _build_national_heatmap(
        merged,
        output_path=os.path.join(config.OUTPUT_DIR, "national_map.html"),
    )
    _build_audit_targets_map(
        merged,
        output_path=os.path.join(config.OUTPUT_DIR, "audit_targets_map.html"),
        top_n=_TOP_N_TARGETS,
    )


# ── Map builders ─────────────────────────────────────────────────────────────

def _build_national_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """
    HeatMap of all PINcodes, intensity = audit_priority_score.
    Hotter/brighter areas have higher composite anomaly priority.
    """
    valid = df.dropna(subset=["lat", "lon", "audit_priority_score"]).copy()

    # Clamp negative scores to zero for HeatMap (negative scores are normal
    # in Z-normalised composites and just mean "below average")
    valid["_heat_weight"] = valid["audit_priority_score"].clip(lower=0)

    heat_data = valid[["lat", "lon", "_heat_weight"]].values.tolist()

    m = folium.Map(
        location=[_INDIA_LAT, _INDIA_LON],
        zoom_start=_DEFAULT_ZOOM,
        tiles="CartoDB positron",
    )
    HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)

    _add_map_title(m, "Aadhaar Sentinel — National Audit Priority HeatMap")
    m.save(output_path)
    logger.info("National heatmap saved: '%s' (%d PINcodes)", output_path, len(valid))


def _build_audit_targets_map(
    df: pd.DataFrame, output_path: str, top_n: int = 20
) -> None:
    """
    Annotated marker map of the top-N highest audit-priority PINcodes.
    Marker colour encodes risk type: dual risk / integrity / capacity / elevated.
    Popup shows all key metrics.
    """
    valid = df.dropna(subset=["lat", "lon", "audit_priority_score"]).copy()
    top = valid.nlargest(top_n, "audit_priority_score")

    # Compute national thresholds for risk classification (from full dataset, not just top)
    pna_thresh = df["PNA"].quantile(0.98)
    dpr_thresh = df["DPR"].quantile(0.98)

    m = folium.Map(
        location=[_INDIA_LAT, _INDIA_LON],
        zoom_start=_DEFAULT_ZOOM,
        tiles="CartoDB positron",
    )

    for _, row in top.iterrows():
        colour = _risk_colour(row, pna_thresh, dpr_thresh)
        popup_html = _build_popup(row)
        tooltip = (
            f"PINCODE {row.get('pincode', '?')} | "
            f"Score: {row.get('audit_priority_score', 0):.2f}"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=9,
            color=colour,
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=tooltip,
        ).add_to(m)

    _add_legend(m)
    _add_map_title(m, f"Aadhaar Sentinel — Top {top_n} Audit Targets")
    m.save(output_path)
    logger.info(
        "Audit targets map saved: '%s' (%d markers)", output_path, len(top)
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_coordinates(path: str) -> Optional[pd.DataFrame]:
    """Load district lat/lon centroids. Returns None if file not found."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    # Rename common variants to canonical 'lat'/'lon'
    col_map = {}
    for col in df.columns:
        if col in {"latitude", "lat"}:
            col_map[col] = "lat"
        elif col in {"longitude", "lon", "long"}:
            col_map[col] = "lon"
    df = df.rename(columns=col_map)

    if "lat" not in df.columns or "lon" not in df.columns:
        logger.error(
            "Coordinates file '%s' must have 'lat'/'lon' (or 'latitude'/'longitude') "
            "columns. Found: %s",
            path,
            df.columns.tolist(),
        )
        return None

    # Normalize keys for safe join
    if "district" in df.columns:
        df["_district_key"] = df["district"].astype(str).str.upper().str.strip()
    if "state" in df.columns:
        df["_state_key"] = df["state"].astype(str).str.upper().str.strip()

    logger.info("Coordinates loaded: %d districts from '%s'", len(df), path)
    return df


def _merge_coordinates(df: pd.DataFrame, coords_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sentinel dataframe with lat/lon data on (district, state).
    Uses normalized uppercase keys matching engine.py conventions.
    Falls back to district-only join if state is missing from coords.
    """
    df = df.copy()
    df["_district_key"] = df["district"].astype(str).str.upper().str.strip()
    df["_state_key"] = df["state"].astype(str).str.upper().str.strip()

    join_cols = ["_district_key"]
    if "_state_key" in coords_df.columns:
        join_cols.append("_state_key")
    elif "state" in coords_df.columns:
        coords_df["_state_key"] = coords_df["state"].astype(str).str.upper().str.strip()
        join_cols.append("_state_key")

    merged = df.merge(
        coords_df[join_cols + ["lat", "lon"]],
        on=join_cols,
        how="left",
    )

    n_missing = merged["lat"].isna().sum()
    if n_missing > 0:
        logger.warning(
            "%d PINcodes could not be matched to district coordinates "
            "and will be excluded from maps.",
            n_missing,
        )

    merged = merged.drop(columns=["_district_key", "_state_key"], errors="ignore")
    return merged


def _risk_colour(row: pd.Series, pna_thresh: float, dpr_thresh: float) -> str:
    """
    Encode risk type as marker colour for the audit targets map.
    Matches the risk composition logic in app.py Tab 3.
    """
    high_pna = row.get("PNA", 0) >= pna_thresh
    high_dpr = row.get("DPR", 0) >= dpr_thresh

    if high_pna and high_dpr:
        return "#8B0000"   # Dark red    — Dual Risk (Critical)
    elif high_dpr:
        return "#800080"   # Purple      — Integrity Anomaly
    elif high_pna:
        return "#003366"   # Dark blue   — Capacity Risk
    else:
        return "#CC6600"   # Amber       — Elevated composite score


def _build_popup(row: pd.Series) -> str:
    dpr_v2 = row.get("DPR_v2", None)
    dpr_v2_str = f"{dpr_v2:.2f}" if dpr_v2 is not None and not pd.isna(dpr_v2) else "N/A"

    return f"""
    <b>PINCODE: {row.get('pincode', 'N/A')}</b><br>
    {row.get('district', '')} &nbsp;|&nbsp; {row.get('state', '')}<br>
    <hr style="margin:4px 0">
    <table style="font-size:12px">
      <tr><td><b>TAI</b></td><td>&nbsp;{int(row.get('TAI', 0)):,}</td></tr>
      <tr><td><b>DPR</b></td><td>&nbsp;{row.get('DPR', 0):.2f}</td></tr>
      <tr><td><b>DPR v2</b></td><td>&nbsp;{dpr_v2_str}</td></tr>
      <tr><td><b>PNA</b></td><td>&nbsp;{row.get('PNA', 0):.4f}</td></tr>
      <tr><td><b>Audit Score</b></td><td>&nbsp;{row.get('audit_priority_score', 0):.2f}</td></tr>
    </table>
    """


def _add_legend(m: folium.Map) -> None:
    legend_html = """
    <div style="position:fixed; bottom:30px; right:30px; z-index:9999;
                background:white; padding:10px 14px; border-radius:6px;
                border:1px solid #ccc; font-size:12px; line-height:1.8">
      <b>Risk Type</b><br>
      <span style="color:#8B0000">&#11044;</span> Dual Risk (Critical)<br>
      <span style="color:#800080">&#11044;</span> Integrity Anomaly<br>
      <span style="color:#003366">&#11044;</span> Capacity Risk<br>
      <span style="color:#CC6600">&#11044;</span> Elevated Score
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def _add_map_title(m: folium.Map, title: str) -> None:
    title_html = f"""
    <div style="position:fixed; top:12px; left:50%; transform:translateX(-50%);
                z-index:9999; background:white; padding:6px 16px;
                border-radius:4px; border:1px solid #ccc;
                font-size:14px; font-weight:bold; font-family:sans-serif">
      {title}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))