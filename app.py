"""
app.py — Aadhaar Sentinel V2 Dashboard
══════════════════════════════════════════════════════════════════════════════
Run: streamlit run app.py

V2 CHANGES:
  - Plotly charts throughout (replacing st.bar_chart and matplotlib).
  - 5 tabs instead of 4: added Overview tab with scatter + KPIs.
  - V2 columns displayed conditionally (DPR_v2, growth_source, Privacy_Masked,
    DBSCAN clusters) — gracefully degrades if V1 CSV is loaded.
  - Updated Methodology tab with V2 formulas (DPR_v2, PNA uncertainty bounds).
  - Fixed pandas rename() deprecation warning in risk_counts column renaming.
  - Added ensemble_final.csv loading for DBSCAN cluster visualization.
  - sidebar shows V2 data quality summary (growth source breakdown).

REQUIRES: plotly in requirements.txt
    pip install plotly  (or add plotly>=5.0.0 to requirements.txt)
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import OUTLIER_PERCENTILE

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aadhaar Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS ───────────────────────────────────────────────────────────────
# Only overrides what Streamlit doesn't expose natively.
# Does not hijack the global theme — safe to run alongside any st.theme config.
st.markdown("""
<style>
/* Tighten metric value font */
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #6c757d; }
/* Risk type badge colours in dataframes */
.risk-dual    { color: #8B0000; font-weight: 600; }
.risk-capacity { color: #003366; font-weight: 600; }
.risk-integrity { color: #800080; font-weight: 600; }
/* Divider spacing */
hr { margin: 0.6rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette (consistent with maps.py) ─────────────────────────────────
COLOUR = {
    "dual":      "#8B0000",
    "integrity": "#800080",
    "capacity":  "#003366",
    "normal":    "#adb5bd",
    "accent":    "#1f77b4",
}
RISK_COLOUR_MAP = {
    "Dual Risk (Critical)": COLOUR["dual"],
    "Integrity Anomaly":    COLOUR["integrity"],
    "Capacity Risk":        COLOUR["capacity"],
    "Normal":               COLOUR["normal"],
}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    sentinel = pd.read_csv("data/processed/sentinel_final.csv")
    outliers = pd.read_csv("data/processed/outliers_ml.csv")
    ensemble = None
    ensemble_path = "data/processed/ensemble_final.csv"
    if os.path.exists(ensemble_path):
        ensemble = pd.read_csv(ensemble_path)

    return sentinel, outliers, ensemble

sentinel_df, outliers_df, ensemble_df = load_data()

# Detect which schema version the CSVs were produced by
HAS_V2 = "DPR_v2" in sentinel_df.columns
HAS_ISO = "iso_score" in sentinel_df.columns
HAS_MASKED = "Privacy_Masked" in sentinel_df.columns
HAS_GROWTH = "growth_source" in sentinel_df.columns

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Aadhaar Sentinel")
    st.caption("Geospatial Risk Analytics · V2" if HAS_V2 else "Geospatial Risk Analytics · V1")
    st.divider()

    st.subheader("Filters")
    states = ["All"] + sorted(sentinel_df["state"].dropna().unique().tolist())
    selected_state = st.selectbox("State", states)

    district_pool = (
        sentinel_df[sentinel_df["state"] == selected_state]
        if selected_state != "All"
        else sentinel_df
    )
    districts = ["All"] + sorted(district_pool["district"].dropna().unique().tolist())
    selected_district = st.selectbox("District", districts)

    st.divider()

    # V2: growth source breakdown in sidebar
    if HAS_GROWTH:
        st.subheader("Population Data Quality")
        gsrc = sentinel_df["growth_source"].value_counts()
        st.caption("RGI growth rate coverage:")
        for src, count in gsrc.items():
            pct = count / len(sentinel_df) * 100
            icon = "✅" if src == "district_rgi" else ("⚠️" if src == "state_rgi_fallback" else "🔶")
            st.caption(f"{icon} {src}: {count:,} ({pct:.0f}%)")
        st.divider()

    if HAS_MASKED:
        n_masked = int(sentinel_df["Privacy_Masked"].sum())
        if n_masked > 0:
            st.caption(f"🔒 {n_masked:,} PINcodes privacy-flagged (k-anonymity)")


# ── Apply filters ─────────────────────────────────────────────────────────────
filtered_df = sentinel_df.copy()
if selected_state != "All":
    filtered_df = filtered_df[filtered_df["state"] == selected_state]
if selected_district != "All":
    filtered_df = filtered_df[filtered_df["district"] == selected_district]

# Global thresholds — always from FULL dataset, not filtered
pna_thresh = sentinel_df["PNA"].quantile(OUTLIER_PERCENTILE)
dpr_thresh = sentinel_df["DPR"].quantile(OUTLIER_PERCENTILE)

# ── Risk type classifier (shared across tabs) ─────────────────────────────────
def classify_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_type"] = "Normal"
    df.loc[(df["PNA"] >= pna_thresh) & (df["DPR"] >= dpr_thresh), "risk_type"] = "Dual Risk (Critical)"
    df.loc[(df["PNA"] >= pna_thresh) & (df["DPR"] < dpr_thresh),  "risk_type"] = "Capacity Risk"
    df.loc[(df["PNA"] < pna_thresh)  & (df["DPR"] >= dpr_thresh), "risk_type"] = "Integrity Anomaly"
    return df

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_capacity, tab_integrity, tab_risk, tab_method = st.tabs([
    "📊 Overview",
    "📍 Capacity Planning",
    "⚠️ Integrity Review",
    "🔍 Risk Composition",
    "📖 Methodology",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    # ── KPI row ───────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("PINcodes (Filtered)", f"{len(filtered_df):,}")
    kpi2.metric("Total Activity (TAI)", f"{int(filtered_df['TAI'].sum()):,}")
    kpi3.metric("ML Anomalies (National)", f"{len(outliers_df):,}")
    kpi4.metric(
        "High-DPR PINcodes",
        int(filtered_df[filtered_df["DPR"] >= dpr_thresh].shape[0]),
    )
    kpi5.metric(
        "High-PNA PINcodes",
        int(filtered_df[filtered_df["PNA"] >= pna_thresh].shape[0]),
    )

    st.divider()

    # ── DPR vs PNA scatter ────────────────────────────────────────────────
    st.subheader("DPR vs PNA — All PINcodes")
    st.caption(
        "Each point is one PINCODE. Log scale on DPR axis to handle extreme values (e.g. 193). "
        "Colour encodes risk type. Vertical/horizontal lines mark 98th percentile thresholds."
    )

    plot_df = classify_risk(filtered_df)

    # Sample normals for performance (keep all anomalies)
    flagged_plot = plot_df[plot_df["risk_type"] != "Normal"]
    normal_plot = plot_df[plot_df["risk_type"] == "Normal"]
    if len(normal_plot) > 2000:
        normal_plot = normal_plot.sample(2000, random_state=42)
    plot_df_sample = pd.concat([flagged_plot, normal_plot])

    hover_cols = ["pincode", "district", "state", "TAI", "DPR", "PNA"]
    if HAS_V2:
        hover_cols.append("DPR_v2")

    scatter_fig = px.scatter(
        plot_df_sample,
        x="PNA",
        y="DPR",
        color="risk_type",
        color_discrete_map=RISK_COLOUR_MAP,
        log_y=True,
        opacity=0.65,
        hover_data={c: True for c in hover_cols if c in plot_df_sample.columns},
        labels={
            "PNA": "PNA (Population-Normalised Activity)",
            "DPR": "DPR (Demographic Pressure Ratio, log scale)",
        },
        height=420,
    )
    scatter_fig.add_vline(x=pna_thresh, line_dash="dash", line_color="#003366",
                          annotation_text="PNA 98th pct", annotation_position="top right")
    scatter_fig.add_hline(y=dpr_thresh, line_dash="dash", line_color="#800080",
                          annotation_text="DPR 98th pct", annotation_position="top right")
    scatter_fig.update_layout(
        legend_title_text="Risk Type",
        plot_bgcolor="#fafafa",
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.divider()

    # ── Top 5 audit targets ───────────────────────────────────────────────
    st.subheader("Top 5 Audit Priority PINcodes (Filtered)")
    top5_cols = ["pincode", "district", "state", "DPR", "PNA", "TAI", "audit_priority_score"]
    if HAS_V2:
        top5_cols.insert(4, "DPR_v2")
    top5 = filtered_df.nlargest(5, "audit_priority_score")[
        [c for c in top5_cols if c in filtered_df.columns]
    ]
    st.dataframe(top5.reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CAPACITY PLANNING
# ══════════════════════════════════════════════════════════════════════════════
with tab_capacity:
    st.subheader("High-Activity Regions (Population-Normalised)")
    st.caption(
        "PNA > 1.0 means total enrollment activity exceeds the estimated resident population. "
        + ("V2: denominator is RGI-projected 2026 population (not Census 2011)."
           if HAS_V2 else "V1: denominator is Census 2011 population.")
    )

    pna_show_cols = ["pincode", "district", "state", "TAI", "PNA"]
    if "PNA_conservative" in filtered_df.columns:
        pna_show_cols += ["PNA_conservative", "PNA_upper_bound"]

    top_pna = (
        filtered_df
        .sort_values("PNA", ascending=False)
        [pna_show_cols]
        .head(20)
        .reset_index(drop=True)
    )
    st.dataframe(top_pna, use_container_width=True)

    st.divider()
    st.subheader("Top 10 Districts by Average PNA")

    pna_chart_data = (
        filtered_df
        .groupby("district", as_index=False)["PNA"]
        .mean()
        .sort_values("PNA", ascending=False)
        .head(10)
    )
    fig_pna = px.bar(
        pna_chart_data,
        x="PNA",
        y="district",
        orientation="h",
        color="PNA",
        color_continuous_scale=[[0, "#cfe2ff"], [1, COLOUR["capacity"]]],
        labels={"PNA": "Average PNA", "district": ""},
        height=360,
    )
    fig_pna.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
        plot_bgcolor="#fafafa",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_pna, use_container_width=True)

    if "PNA_conservative" in filtered_df.columns:
        st.caption(
            "PNA_conservative (lower bound): fewer residents assumed → higher stress reading. "
            "PNA_upper_bound: more residents assumed → lower stress reading."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INTEGRITY REVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_integrity:
    st.subheader("PINcodes Requiring Audit Review (High DPR)")
    st.caption(
        "High DPR indicates demographic attribute updates without corresponding biometric "
        "re-verification — a process anomaly flag, not a fraud determination."
    )

    dpr_show_cols = ["pincode", "district", "state", "demo_total", "bio_total", "DPR"]
    if HAS_V2:
        dpr_show_cols.append("DPR_v2")

    high_dpr = (
        filtered_df
        .sort_values("DPR", ascending=False)
        [[c for c in dpr_show_cols if c in filtered_df.columns]]
        .head(20)
        .reset_index(drop=True)
    )
    st.dataframe(high_dpr, use_container_width=True)

    st.divider()

    col_chart, col_v2 = st.columns([2, 1])

    with col_chart:
        st.subheader("Top 10 Districts by Average DPR")
        dpr_chart_data = (
            filtered_df
            .groupby("district", as_index=False)["DPR"]
            .mean()
            .sort_values("DPR", ascending=False)
            .head(10)
        )
        fig_dpr = px.bar(
            dpr_chart_data,
            x="DPR",
            y="district",
            orientation="h",
            color="DPR",
            color_continuous_scale=[[0, "#e7d4f5"], [1, COLOUR["integrity"]]],
            labels={"DPR": "Average DPR", "district": ""},
            height=360,
        )
        fig_dpr.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            plot_bgcolor="#fafafa",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_dpr, use_container_width=True)

    with col_v2:
        if HAS_V2:
            st.subheader("DPR vs DPR_v2")
            st.caption(
                "DPR (V1): transaction ratio — demographic / biometric updates.\n\n"
                "DPR_v2 (V2): RGI-normalized — updates relative to expected "
                "demographic growth. Higher values indicate activity above "
                "natural population demand."
            )
            scatter_v2 = px.scatter(
                filtered_df.dropna(subset=["DPR", "DPR_v2"]).sample(
                    min(500, len(filtered_df)), random_state=42
                ),
                x="DPR",
                y="DPR_v2",
                color_discrete_sequence=[COLOUR["integrity"]],
                opacity=0.5,
                labels={"DPR": "DPR (V1)", "DPR_v2": "DPR_v2 (V2)"},
                height=340,
            )
            scatter_v2.update_layout(plot_bgcolor="#fafafa", margin=dict(t=10, b=10))
            st.plotly_chart(scatter_v2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK COMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.subheader("Risk Type Composition")
    st.caption(
        f"PNA threshold: ≥ {pna_thresh:.4f} · "
        f"DPR threshold: ≥ {dpr_thresh:.4f} (both 98th percentile, national)"
    )

    risk_df = classify_risk(filtered_df)
    flagged_df = risk_df[risk_df["risk_type"] != "Normal"]
    risk_counts = (
        flagged_df["risk_type"]
        .value_counts()
        .reset_index()
    )
    risk_counts.columns = ["Risk Type", "Count"]  # explicit rename — no deprecation

    col_pie, col_table = st.columns([1, 1])

    with col_pie:
        if len(risk_counts) > 0:
            fig_pie = px.pie(
                risk_counts,
                names="Risk Type",
                values="Count",
                color="Risk Type",
                color_discrete_map=RISK_COLOUR_MAP,
                hole=0.35,
                height=360,
            )
            fig_pie.update_traces(textposition="outside", textinfo="percent+label")
            fig_pie.update_layout(
                showlegend=False,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No anomalies in current filter selection.")

    with col_table:
        st.subheader("Detection Method Comparison")
        stat_count = len(
            sentinel_df[
                (sentinel_df["TAI"] >= sentinel_df["TAI"].quantile(OUTLIER_PERCENTILE))
                | (sentinel_df["DPR"] >= dpr_thresh)
                | (sentinel_df["PNA"] >= pna_thresh)
            ]
        )
        ml_count = len(outliers_df)

        if "pincode" in outliers_df.columns:
            stat_pins = set(
                sentinel_df[
                    (sentinel_df["TAI"] >= sentinel_df["TAI"].quantile(OUTLIER_PERCENTILE))
                    | (sentinel_df["DPR"] >= dpr_thresh)
                    | (sentinel_df["PNA"] >= pna_thresh)
                ]["pincode"].astype(str)
            )
            ml_pins = set(outliers_df["pincode"].astype(str))
            overlap = len(stat_pins & ml_pins)
            overlap_pct = f"{overlap / ml_count * 100:.0f}%" if ml_count > 0 else "N/A"
        else:
            overlap, overlap_pct = "N/A", "N/A"

        comparison_data = pd.DataFrame({
            "Method": ["Statistical (98th pct)", "Isolation Forest (ML)", "Overlap (high-confidence)"],
            "Count": [f"{stat_count:,}", f"{ml_count:,}", f"{overlap} ({overlap_pct})"],
        })
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)

        # V2: DBSCAN cluster breakdown
        if ensemble_df is not None and "dbscan_cluster" in ensemble_df.columns:
            st.divider()
            st.subheader("DBSCAN Cluster Analysis")

            n_clusters = len(set(ensemble_df["dbscan_cluster"].unique()) - {-1})
            n_clustered = int(ensemble_df["dbscan_is_clustered"].sum())
            n_noise = int((ensemble_df["dbscan_cluster"] == -1).sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("Clusters Found", n_clusters)
            c2.metric("Clustered PINcodes", n_clustered)
            c3.metric("Isolated Outliers", n_noise)

            if n_clusters > 0:
                cluster_sizes = (
                    ensemble_df[ensemble_df["dbscan_is_clustered"]]
                    .groupby("dbscan_cluster")["pincode"]
                    .count()
                    .reset_index()
                )
                cluster_sizes.columns = ["Cluster ID", "PINcodes"]
                cluster_sizes = cluster_sizes.merge(
                    ensemble_df[ensemble_df["dbscan_is_clustered"]]
                    .groupby("dbscan_cluster")["state"]
                    .agg(lambda x: x.value_counts().index[0])
                    .reset_index()
                    .rename(columns={"state": "Dominant State", "dbscan_cluster": "Cluster ID"}),
                    on="Cluster ID",
                )
                st.dataframe(cluster_sizes, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
with tab_method:
    v2_badge = " *(V2 — RGI projected)*" if HAS_V2 else " *(V1 — Census 2011)*"

    st.markdown(f"""
### Core Metrics

**TAI — Total Activity Index** (Paper B Eq. 1)
```
TAI_i = Enrolments_i + DemographicUpdates_i + BiometricUpdates_i
```
Measures overall operational throughput per PINCODE.

---

**DPR — Demographic Pressure Ratio** (Paper B Eq. 2 baseline)
```
DPR_i = DemographicUpdates_i / (BiometricUpdates_i + 1)
```
Process integrity proxy. High DPR indicates demographic attribute changes
without corresponding biometric re-verification.
""")

    if HAS_V2:
        st.markdown("""
**DPR_v2 — RGI-Normalized Demographic Pressure** (Paper B Eq. 2 V2)
```
DPR_v2_i = ΔT_demo_i / (γ_d × P_i^{2026} + ε)
```
Where γ_d is the RGI projected annual growth rate for district d, P_i^{2026}
is the RGI-projected PINCODE population, and ε = 1e-5 prevents division by
zero in static-population zones. Values > 1 mean demographic updates exceed
expected natural demographic activity.
""")

    st.markdown(f"""
---

**PNA — Population-Normalised Activity** (Paper B Eq. 3){v2_badge}
```
PNA_i = TAI_i / P_i^{{year}}
```
Capacity stress indicator. PNA > 1.0 means total enrollment activity exceeds
estimated resident population.

**Population Estimation:**
P_i = District_Projected_Population ÷ active PINcodes in district.
{'V2: District population is RGI-projected 2026 (not Census 2011). Uncertainty bounds use RGI projection range (±5%).' if HAS_V2 else 'V1: Census 2011 district population. Uncertainty bounds: ±20% flat growth factor.'}

---

### Anomaly Detection

**Method 1 — Statistical Baseline**
PINcodes in the top 2% on any of TAI, DPR, or PNA (union of 98th percentiles).

**Method 2 — Isolation Forest (Stage 1)**
`IsolationForest(n_estimators=200, contamination=0.02, random_state=42)`
on joint (TAI, DPR, PNA) feature space. Identifies multivariate anomalies.
iso_score retained for continuous ranking within flagged set.

{'**Method 3 — DBSCAN Clustering (Stage 2, V2)**  ' if HAS_V2 else ''}
{'`DBSCAN(eps=0.5, min_samples=3)` in StandardScaler-normalized feature space applied to the IsolationForest-flagged subset. Clustered points (dbscan_cluster ≥ 0) represent coordinated multi-PINCODE anomaly patterns — stronger evidence than isolated outliers.' if HAS_V2 else ''}

---

### Composite Risk Score

```
audit_priority_score = (dpr_z × 0.50) + (pna_z × 0.35) + (activity_z × 0.15)
```
Z-score normalised. Weights in config.py. Shifted to ≥ 0 for readability.

---

### Limitations

- PINCODE population is a district-level proxy (district pop ÷ n PINcodes).
  Official PINCODE shapefiles are not publicly available in India.
- DPR anomalies indicate statistical deviation — not confirmed irregularity.
  Ground-truth audit outcomes required to evaluate precision/recall.
- Analysis is cross-sectional; temporal trend detection deferred to V3.
{'- RGI growth rates from state-level averages where district data unavailable.' if HAS_V2 else '- Census 2011 population estimates introduce ~20% uncertainty for high-growth urban districts.'}

---

### Privacy

All analysis uses aggregated PINCODE or district-level data only.
No Aadhaar numbers, biometric templates, or individual records are processed.
{'k-Anonymity: districts with Projected_2026 < 500 are flagged and excluded from the public IndiaID-Bench dataset (Paper A §4).' if HAS_V2 else ''}
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Aadhaar Sentinel V2 · Geospatial Risk Analytics for National Identity Infrastructure"
    + (" · RGI-Projected Population" if HAS_V2 else " · Census 2011 Population")
)