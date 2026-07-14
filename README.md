# Aadhaar‑Sentinel
### Geospatial Risk Analytics for National Identity Infrastructure

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Anomaly_Detection-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

---

> **Aadhaar‑Sentinel** is a geospatial analytics framework for India's national digital identity infrastructure.
> It separates *capacity stress* from *identity‑integrity anomalies* at PINCODE resolution, moving audit focus from complaint‑based selection to severity‑based prioritisation.

**Status (July 2026 — V2.2):**

- Full ensemble pipeline implemented and tested: statistical baseline + Isolation Forest + Local Outlier Factor + HDBSCAN clustering (DBSCAN as ablation).
- District name reconciliation complete (494 → 163 unmatched, fully documented provenance).
- RGI 2011–2036 state‑level growth rates integrated as tiered population projection (replacing the national fallback).
- Sensitivity analysis complete (IF contamination, LOF neighbourhood size, composite weights).
- 67 unit tests, all passing; results exactly reproducible (`random_state=42`).
- In progress: publication figures, IndiaID‑Bench Zenodo release, methods paper.

---

## Motivation

Aadhaar is the world's largest biometric identity infrastructure, handling billions of authentications and updates each year. Operational dashboards are largely aggregate and volume‑based: they report enrolments, demographic updates, and biometric updates at state/district level and rank high‑volume locations for audit.

This creates three problems:

1. **Silent anomalies**
   Low‑volume centres can process hundreds of demographic updates with almost no biometric verification. They look "normal" in volume dashboards but weaken identity integrity guarantees.

2. **Mixed signals**
   High update activity can mean genuine demand (capacity stress), process irregularity (identity risk), or both. Aggregate dashboards cannot disentangle these.

3. **Outdated denominators**
   The 2021 Population Census was never conducted; Census 2011 is over a decade old and the RGI 2011–2036 projections exist only as a PDF report. Most analyses still use district‑level 2011 counts, blurring capacity signals.

**Aadhaar‑Sentinel** addresses these issues by:

- Joining three UIDAI update streams with population denominators at PINCODE resolution.
- Defining dual metrics for capacity stress and identity integrity lag.
- Applying multi‑method unsupervised anomaly detection over this feature space.
- Producing a reusable benchmark dataset (**IndiaID‑Bench**) for further research.

---

## What Sentinel Does Differently

| Conventional Approach              | Sentinel Approach                                                    |
|------------------------------------|----------------------------------------------------------------------|
| Volume‑based ranking               | Severity‑weighted composite scoring (Z‑normalised)                  |
| Single metric (volume)             | Dual metrics (Capacity vs Integrity) + composite risk               |
| Independent thresholds per metric  | Multi‑method detection (Isolation Forest + LOF + HDBSCAN ensemble)  |
| Static district maps               | Interactive PINCODE‑resolution dashboard                            |
| Point estimates from 2011 Census   | RGI‑projected 2026 denominators with tiered fallback + uncertainty  |

---

## Repository Architecture

```text
aadhaar-sentinel/
├── src/
│   ├── config.py        # Thresholds, weights, paths, policy constants
│   ├── loader.py        # Raw CSV ingestion and PINCODE aggregation
│   ├── projections.py   # Tiered RGI population projection to 2026 + validation
│   ├── engine.py        # Metric computation (TAI, DPR, PNA) + Census join
│   ├── scoring.py       # IF + LOF + HDBSCAN/DBSCAN ensemble + composite score
│   └── maps.py          # Geospatial output (currently inactive)
├── scripts/
│   ├── build_district_map.py   # Two-stage fuzzy district reconciliation
│   └── sensitivity.py          # Jaccard stability sweeps
├── data/
│   ├── raw/             # UIDAI & Census sources (untracked); RGI growth rates
│   │                    #   and district reconciliation map (tracked)
│   └── processed/       # Derived features and outputs (regeneratable)
├── outputs/maps/        # Interactive HTML maps
├── notebooks/           # Exploratory analysis only
├── tests/               # 67 unit tests
├── app.py               # Streamlit dashboard
├── pipeline.py          # End-to-end pipeline runner
└── requirements.txt
```

Design principle: **separation of concerns** — projection, metrics, scoring, orchestration, and visualisation are cleanly separated and individually testable.

---

## Core Metrics (PINCODE Level)

### 1. Total Activity Index (TAI)

```text
TAI = Enrolments + Demographic Updates + Biometric Updates
```

Overall operational throughput; low‑weight context in composite scoring.

### 2. Demographic‑to‑Biometric Processing Ratio (DPR)

```text
DPR = Demographic Updates / (Biometric Updates + 1)
```

- **Identity integrity proxy.** High DPR indicates demographic changes (name, address, date of birth) occurring without corresponding biometric re‑verification.
- Example: a single PINCODE in Ahilyanagar district (Maharashtra) records DPR = 193 — i.e. 193 demographic updates per biometric re‑verification — while the district‑level aggregate DPR is 0.64. The anomaly is invisible at district resolution; this is precisely the class of signal Sentinel targets.

### 3. Population‑Normalised Activity (PNA)

```text
PNA = TAI / Estimated PINCODE Population
```

- **Capacity stress indicator.** Values significantly above 1.0 suggest activity exceeding the estimated resident population (extreme demand or cross‑boundary footfall).

#### Population Estimation (Tiered RGI Projection)

District populations are projected from Census 2011 to 2026 using a tiered growth‑rate system:

- **Tier 1 — district‑level RGI rates** (not publicly available; reserved for future integration).
- **Tier 2 — state‑level RGI rates (ACTIVE):** compound annual growth rates derived from Table 8 of the official RGI/NCP report *Population Projections for India and States 2011–2036* (Nov 2019), covering all 37 states/UTs. Shipped with the repository as `data/raw/RGI_growth_rates.csv`.
- **Tier 3 — national fallback** (1.2%/yr; unused on the current baseline — coverage is 100% Tier 2 for all Census‑matched districts).

Three small UTs (Daman & Diu, Dadra & Nagar Haveli, Puducherry) carry genuine RGI projections outside normal plausibility bounds (migration‑driven, up to ~2.9× over 15 years). These published rates are retained **uncapped** and reported as expected special cases rather than validation errors — see `config.RGI_HIGH_GROWTH_UTS`.

PNA is reported with uncertainty bounds; each district carries a `growth_source` provenance tag.

---

## Anomaly Detection Framework

### Method 1 — Statistical Baseline (98th‑Percentile Thresholding)

Univariate flags on TAI, DPR, or PNA. Simple, interpretable, reproducible — but blind to multivariate structure.

### Method 2 — Isolation Forest (Primary Multivariate Detector)

`sklearn.ensemble.IsolationForest` (`n_estimators=200`, `contamination=0.02`) over the joint `(TAI, DPR, PNA)` space. The binary flag provides the anomaly signal; the continuous `iso_score` ranks severity **within** the flagged set, supporting prioritised audit queues.

### Method 3 — Local Outlier Factor (Complementary Detector)

`sklearn.neighbors.LocalOutlierFactor` (`n_neighbors=20`, novelty mode, mirroring the IF fit/score protocol). LOF detects **locally** anomalous points — deviations from neighbourhood density — where IF detects **globally** isolated ones.

**Complementarity finding:** IF and LOF agree on only 8% of flags, and their respective top‑20 lists are fully disjoint. This is not noise — the two mechanisms surface different anomaly classes, which is the empirical argument for an ensemble over any single method.

### Clustering Stage — HDBSCAN (primary) + DBSCAN (ablation)

`sklearn.cluster.HDBSCAN` (`min_cluster_size=5`) over the IF‑flagged subset detects coordinated multi‑PINCODE anomaly structure with soft cluster membership; DBSCAN with fixed ε serves as the ablation baseline (33 vs 5 clusters — HDBSCAN adapts to variable density).

### Sensitivity Analysis

Top‑20 audit‑list stability (Jaccard) swept across IF contamination {0.01–0.10}, LOF `n_neighbors` {10–50}, and DPR weight {0.40–0.60}:

- IF contamination: **Jaccard 1.0 across all configurations** — the most extreme PINcodes are fully threshold‑robust.
- Composite weights: Jaccard 0.54–0.74 — ranking shifts moderately with weights; the underlying flag set does not.
- LOF `n_neighbors`: sensitive (a known property of LOF), disclosed as a limitation.

### Composite Risk Score

```text
audit_priority_score = 0.50 * Z(DPR) + 0.35 * Z(PNA) + 0.15 * Z(TAI)
```

DPR weighted highest: integrity anomalies are hardest to detect and highest audit urgency. Weights are configurable in `src/config.py` and validated by the sensitivity analysis above.

---

## Key Results (Current Validated Baseline)

| Quantity                          | Value        |
|-----------------------------------|--------------|
| PINCODEs analysed                 | 32,898       |
| Statistical flags (98th pct)      | 1,918        |
| Isolation Forest flags            | 658          |
| Statistical–IF overlap            | 656 (100%)   |
| LOF flags                         | 623          |
| IF∩LOF overlap                    | 54 (8%)      |
| HDBSCAN clusters (primary)        | 33           |
| DBSCAN clusters (ablation)        | 5            |
| Unmatched districts (documented)  | 163          |

**Highest‑DPR PINcodes (verified on current baseline):**

| District           | State        | DPR   |
|--------------------|--------------|-------|
| Ahilyanagar (AHMADNAGAR) | Maharashtra | 193.0 |
| Didwana‑Kuchaman   | Rajasthan    | 115.0 |
| Khairthal‑Tijara   | Rajasthan    | 114.0 |
| Ahilyanagar (2nd PINcode) | Maharashtra | 107.0 |
| Jalor              | Rajasthan    | 104.0 |

Notably, three of the top five sit in Rajasthan, and two of those are districts created after 2011 — a pattern under further analysis.

> **Important:** High DPR is classified as a *process anomaly requiring audit investigation*, **not** as evidence of fraud. Ground‑truth audit outcomes are required to distinguish operator error, data entry patterns, and intentional irregularity.

---

## IndiaID‑Bench (Dataset)

The pipeline produces an ML‑ready benchmark dataset, **IndiaID‑Bench** — to our knowledge the first public PINCODE‑resolution joined UIDAI × Census × RGI dataset — designed for:

- Demographic and public policy analysis (under‑enrolment vs capacity).
- Benchmarking unsupervised anomaly detectors on real government data.
- AI governance and digital identity audit research.

**Dataset contents:**

- PINCODE‑level features: `TAI`, `DPR`, `DPR_v2`, `PNA` + uncertainty bounds, `audit_priority_score`, `if_flag`, `iso_score`, `lof_flag`, `lof_score`, HDBSCAN/DBSCAN cluster labels.
- District metadata: reconciled names, states, projected populations, `growth_source` provenance tags.
- Privacy: k‑anonymity masking (threshold 500); no individual‑level or personally identifiable data.

A DOI‑backed Zenodo release (CC BY 4.0) is in preparation.

---

## Streamlit Dashboard

`app.py` provides four views: **Capacity Planning** (PNA rankings), **Integrity Review** (high‑DPR drill‑downs), **Risk Composition** (flags by detector), and **Methodology** (definitions, caveats, limitations).

```bash
python pipeline.py      # generate processed data
streamlit run app.py    # launch dashboard
```

---

## Quick Start

**Requirements:** Python 3.10+ (scikit‑learn ≥ 1.3 for `sklearn.cluster.HDBSCAN`).

```bash
git clone https://github.com/saad-research/Sentinel-Geospatial-Anomaly-Detection-System.git
cd Sentinel-Geospatial-Anomaly-Detection-System

pip install -r requirements.txt

# Place UIDAI CSV folders in data/raw/
# Place district population CSV in data/raw/Census_2011.csv
#   (columns: 'District name', 'State name', 'Population')
# RGI_growth_rates.csv and district_name_map.csv ship with the repository.

python pipeline.py            # full pipeline
streamlit run app.py          # dashboard
python -m pytest tests -v     # 67 tests
```

---

## Data Sources

| Dataset                  | Source                                        | Granularity      |
|--------------------------|-----------------------------------------------|------------------|
| Aadhaar Enrolment Data   | UIDAI Open Data Portal                        | PINCODE × Period |
| Demographic Update Data  | UIDAI Open Data Portal                        | PINCODE × Period |
| Biometric Update Data    | UIDAI Open Data Portal                        | PINCODE × Period |
| District Population      | Census of India 2011 (Primary Census Abstract)| District         |
| Growth Projections       | RGI/NCP *Population Projections for India and States 2011–2036* (Nov 2019), Table 8 | State |

All analysis uses **aggregated, non‑personal** data. No Aadhaar numbers, biometric templates, or individual records are processed.

---

## Privacy & Ethical Considerations

- Only public, aggregated government data is used; no PII is ingested or derivable.
- Small‑population districts are k‑anonymity masked (threshold 500) and excluded from model fitting while still being scored.
- Findings are **signals for audit and governance**, not adjudications of misconduct.
- Unsupervised detection operates without labelled anomalies; evaluation is via cross‑method agreement, sensitivity analysis, and plausibility — formal precision/recall requires ground‑truth audit outcomes that do not publicly exist.

---

## Limitations & Roadmap

**Current limitations:**

- Cross‑sectional analysis (single snapshot); temporal patterns not yet modelled.
- Growth rates are state‑level (Tier 2); district‑level RGI rates are not publicly available.
- 163 post‑2011 districts (e.g. Palghar, Hapur) have no Census 2011 equivalent and are documented as unmatched rather than force‑matched.

**Planned extensions:**

- Publication figures and formal write‑up of the ensemble framework.
- IndiaID‑Bench public release with DOI.
- District‑level (Tier 1) growth rates if a public source materialises.
- Time‑series anomaly detection on quarterly update streams.
- State‑level choropleth mapping.

---

## Tech Stack

| Component              | Technology                                      |
|------------------------|-------------------------------------------------|
| Data Engineering       | Python, Pandas, NumPy                           |
| Anomaly Detection      | scikit‑learn (IsolationForest, LOF, HDBSCAN, DBSCAN) |
| Reconciliation         | rapidfuzz (two‑stage fuzzy matching)            |
| Statistics             | SciPy (Z‑score normalisation)                   |
| Interactive Dashboard  | Streamlit                                       |
| Testing                | pytest (67 tests)                               |

---

## Acknowledgements

Population estimation uses Census of India 2011 district statistics projected to 2026 via official RGI state‑level growth rates. This is suitable for research; production deployment in a government context would require official registry denominators and policy review.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built as part of applied research in privacy‑preserving analytics and AI security.*