# Aadhaar‑Sentinel
### Geospatial Risk Analytics for National Identity Infrastructure

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Anomaly_Detection-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

---

> **Aadhaar‑Sentinel** is a geospatial analytics framework for India's national digital identity infrastructure.  
> It separates *capacity stress* from *identity‑integrity anomalies* at PINCODE resolution, moving audit focus from complaint‑based selection to severity‑based prioritisation.

**Status (June 2026):**

- Core pipeline (metrics + statistical baseline + Isolation Forest) implemented and tested.
- District name reconciliation and PINCODE aggregation working.
- DPR=193 Ahilyanagar anomaly verified in the reconciled pipeline.
- Extensions in progress: Local Outlier Factor (LOF), HDBSCAN clustering, sensitivity analysis, RGI growth integration.

---

## Motivation

Aadhaar is the world’s largest biometric identity infrastructure, handling billions of authentications and updates each year.[web:574] Operational dashboards are largely aggregate and volume‑based: they report enrolments, demographic updates, and biometric updates at state/district level and rank high‑volume locations for audit.

This creates three problems:

1. **Silent anomalies**  
   Low‑volume centres can process hundreds of demographic updates with almost no biometric verification. They look “normal” in volume dashboards but weaken identity integrity guarantees.

2. **Mixed signals**  
   High update activity can mean genuine demand (capacity stress), process irregularity (identity risk), or both. Aggregate dashboards cannot disentangle these.

3. **Outdated denominators**  
   The 2021 Population Census was never conducted; Census 2011 is over a decade old and RGI 2011‑2036 projections are stuck in PDF form.[web:574] Most analyses still use district‑level 2011 counts, blurring capacity signals.

**Aadhaar‑Sentinel** addresses these issues by:

- Joining three UIDAI update streams with population denominators at PINCODE resolution.
- Defining dual metrics for capacity stress and identity integrity lag.
- Applying unsupervised anomaly detection over this feature space.
- Producing a reusable benchmark dataset (**IndiaID‑Bench**) for further research.

---

## What Sentinel Does Differently

Most government analytics stacks:

- Aggregate to state/district level.
- Rank locations by raw volume.
- Use one or two simple thresholds.

Sentinel introduces three key changes:

| Conventional Approach               | Sentinel Approach                                                   |
|-------------------------------------|---------------------------------------------------------------------|
| Volume‑based ranking                | Severity‑weighted composite scoring (Z‑normalised)                 |
| Single metric (volume)             | Dual metrics (Capacity vs Integrity) + composite risk              |
| Independent thresholds per metric  | Multivariate anomaly detection (Isolation Forest; LOF planned)     |
| Static district maps               | Interactive PINCODE‑resolution dashboard                           |
| Point estimates from 2011 Census   | Population uncertainty bounds + planned RGI growth integration     |

---

## Repository Architecture

```text
aadhaar-sentinel/
├── src/
│   ├── config.py      # Thresholds, weights, paths
│   ├── loader.py      # Raw CSV ingestion and normalisation
│   ├── engine.py      # Metric computation (TAI, DPR, PNA, etc.)
│   ├── scoring.py     # Anomaly detection (statistical + Isolation Forest; LOF planned)
│   └── maps.py        # Folium-based map generation
├── data/
│   ├── raw/           # UIDAI & population source datasets (gitignored)
│   └── processed/     # Derived features and outputs (CSV)
├── outputs/maps/      # Interactive HTML maps
├── notebooks/         # Exploratory analysis only
├── tests/             # Unit tests for core pipeline
├── app.py             # Streamlit dashboard
├── pipeline.py        # End-to-end pipeline runner
└── requirements.txt
```

Design principle: **separation of concerns** — metrics, scoring, orchestration, and visualisation are cleanly separated and individually testable.

---

## Core Metrics (PINCODE Level)

Sentinel derives three primary indicators for each PINCODE:

### 1. Total Activity Index (TAI)

```text
TAI = Enrolments + Demographic Updates + Biometric Updates
```

- Measures overall operational throughput.
- Used as context for other metrics and as a low‑weight component in composite scoring.

### 2. Demographic‑to‑Biometric Processing Ratio (DPR)

```text
DPR = Demographic Updates / (Biometric Updates + 1)
```

- **Identity integrity proxy.**
- High DPR indicates demographic changes (name, address, date of birth, etc.) occurring without corresponding biometric re‑verification.
- DPR=193 in Ahilyanagar (Maharashtra) means 193 demographic updates per biometric re‑verification on the analysed snapshot — a pattern that merits audit attention.

### 3. Population‑Normalised Activity (PNA)

```text
PNA = TAI / Estimated PINCODE Population
```

- **Capacity stress indicator.**
- Values significantly above 1.0 suggest total enrolment/update activity exceeding the estimated resident population (extreme demand or cross‑boundary footfall).

#### Population Estimation & Uncertainty

- Baseline: **District population (Census 2011)** divided by number of PINCODEs in the district.
- PNA is reported with uncertainty bounds using plausible growth factors (e.g. 0–20%) to reflect post‑2011 population change.
- Planned: replace simple bounds with **state/district‑specific RGI 2011–2036 growth rates** once extracted.

Derived columns include:

- `PNA_point_estimate`
- `PNA_conservative` (lower growth assumption)
- `PNA_upper_bound` (higher growth assumption)

---

## Anomaly Detection Framework

Sentinel currently implements two detection methods and is designed to host a third:

### Method 1 — Statistical Baseline (Percentile Thresholding)

- PINCODEs exceeding the **98th percentile** on any of:
  - TAI,
  - DPR,
  - or PNA
  are flagged as univariate outliers.
- Advantages:
  - Simple, interpretable,
  - Easy to reproduce and explain to non‑technical stakeholders.
- Limitation:
  - Treats each metric independently; cannot capture multivariate anomaly structure.

### Method 2 — Isolation Forest (Primary Multivariate Detector)

A `sklearn.ensemble.IsolationForest` model is trained on the joint feature space:

```python
X = df[["TAI", "DPR", "PNA"]].values
model = IsolationForest(
    n_estimators=200,
    contamination=0.02,  # ~2% anomalies, mirroring the 98th percentile
    random_state=42,
)
df["if_flag"]    = (model.fit_predict(X) == -1)  # True = anomaly
df["iso_score"]  = model.decision_function(X)    # Continuous isolation score
```

- The binary flag (`if_flag`) provides a clear anomaly signal.
- The `iso_score` enables **ranking within anomalies**: more negative scores are stronger anomalies.
- This supports severity‑based audit queues rather than “all flags equal”.

### Planned Method 3 — Local Outlier Factor (LOF) & HDBSCAN

To strengthen robustness and provide richer structure:

- **Local Outlier Factor (LOF)**  
  Density‑based anomaly detection to capture local neighbourhood deviations in the same feature space.

- **HDBSCAN‑based clustering**  
  Hierarchical density‑based clustering over `(TAI, DPR, PNA)` and spatial coordinates to detect **coordinated multi‑PINCODE anomaly regions**.

- **Sensitivity analysis**  
  Systematic sweeps over:
  - Isolation Forest contamination rates,
  - LOF neighbourhood sizes,
  - composite risk weights,  
  to quantify stability of the top‑k audit list.

These are part of the planned methods expansion and will be integrated into `src/scoring.py` and analysis notebooks.

### Composite Risk Score

Sentinel also computes a Z‑score normalised composite for audit prioritisation:

```text
audit_priority_score = 0.50 * Z(DPR) + 0.35 * Z(PNA) + 0.15 * Z(TAI)
```

- DPR anomalies => potential **process non‑compliance** (highest urgency).
- PNA anomalies => **capacity stress** (operational response).
- TAI => overall load context.

Weights are configurable via `src/config.py` and can be tuned in sensitivity analysis.

---

## Key Findings (Example Snapshot)

On one national snapshot (millions of update records aggregated to ~33,000 PINCODEs), Sentinel surfaced several patterns:

| Risk Category           | Location              | Key Metric(s)              | Interpretation                                                  |
|-------------------------|-----------------------|----------------------------|-----------------------------------------------------------------|
| **Critical Dual Risk**  | West Delhi            | PNA ≫ 1, DPR elevated      | Operating at several times estimated capacity; integrity risk. |
| **Critical Dual Risk**  | North East Delhi      | High PNA, moderate DPR     | Severe capacity overload; demands kit reallocation.            |
| **High‑DPR Cluster**    | Ahilyanagar (MH)      | DPR ≈ 193, low PNA         | Extreme identity‑lag pattern at low volume; process anomaly.   |
| **Capacity Deficit**    | Moradabad             | High PNA, near‑normal DPR  | Infrastructure stress independent of integrity concerns.       |
| **Border Corridor**     | Sribhumi (Assam)      | DPR ≫ regional baseline    | High demographic churn; plausible residency‑status dynamics.   |

> **Important:** High DPR is classified as a *process anomaly requiring audit investigation*, **not** as evidence of fraud. Ground‑truth audit outcomes are required to distinguish operator error, data entry patterns, and intentional irregularity.

In validation:

- Isolation Forest flagged a smaller, high‑confidence multivariate anomaly set.
- A large fraction of these IF anomalies were independently flagged by the statistical baseline, indicating strong overlap between simple thresholds and multivariate structure.
- Statistical thresholding flagged additional single‑metric extremes that do not exhibit multivariate behaviour and can be reviewed separately.

Exact counts will vary between runs; refer to `data/processed/` for current totals.

---

## IndiaID‑Bench (Dataset)

The pipeline produces an ML‑ready benchmark dataset, **IndiaID‑Bench**, designed for:

- Demographic and public policy analysis (under‑enrolment vs capacity).
- Benchmarking unsupervised anomaly detectors on real government data.
- AI governance and digital identity audit research.

**Dataset contents (planned):**

- PINCODE‑level features:
  - `TAI`, `DPR`, `DPR_v2`, `PNA`, uncertainty bounds.
  - `audit_priority_score`, `if_flag`, `iso_score`, future `lof_flag`, cluster labels.
- District metadata:
  - Reconciled district names and state codes.
  - Population denominators and growth source tags (Census vs RGI).
- Privacy & masking:
  - Flags for k‑anonymity / minimum population.
  - No individual‑level or personally identifiable data.

A public, DOI‑backed release is planned once reconciliation, RGI integration, and masking are finalised.

---

## Streamlit Dashboard

An interactive Streamlit app (`app.py`) provides four main views:

- **Capacity Planning**  
  District‑level PNA rankings; top‑N charts for infrastructure planning.

- **Integrity Review**  
  High‑DPR PINCODE drill‑downs with breakdown by update type.

- **Risk Composition**  
  Classification of flagged PINCODEs by risk type and detector (statistical vs Isolation Forest; LOF later).

- **Methodology**  
  Metric definitions, caveats, and limitations.

Geospatial outputs (e.g., national risk map, top‑k audit map) are generated as interactive HTML files in `outputs/maps/` via `pipeline.py`.

**Run locally:**

```bash
# 1. Generate processed data
python pipeline.py

# 2. Launch dashboard
streamlit run app.py
```

---

## Quick Start

**Requirements:** Python 3.9+, optional Jupyter for notebooks.

```bash
# Clone the repository
git clone https://github.com/saad-research/aadhaar-sentinel.git
cd aadhaar-sentinel

# Install dependencies
pip install -r requirements.txt

# Place UIDAI CSV folders in data/raw/
# Place district population CSV in data/raw/Census_2011.csv
#   (e.g., india-districts-census-2011; columns: 'District name', 'State name', 'Population')

# Run the full pipeline
python pipeline.py

# Launch dashboard
streamlit run app.py

# Run tests
python -m pytest tests/ -v
```

---

## Data Sources

| Dataset                  | Source                     | Granularity         |
|--------------------------|----------------------------|---------------------|
| Aadhaar Enrolment Data   | UIDAI Open Data Portal     | PINCODE × Period    |
| Demographic Update Data  | UIDAI Open Data Portal     | PINCODE × Period    |
| Biometric Update Data    | UIDAI Open Data Portal     | PINCODE × Period    |
| District Population      | Census of India 2011       | District            |
| Growth Projections (WIP) | RGI 2011–2036 PDF          | State / District    |
| District Coordinates     | Public GitHub (centroids)  | District            |

All analysis uses **aggregated, non‑personal** data at PINCODE or district level. No Aadhaar numbers, biometric templates, or individual records are processed.

---

## Privacy & Ethical Considerations

- Only public, aggregated government data is used.
- No Personally Identifiable Information (PII) is ingested or derivable.
- Findings are **signals for audit and governance**, not adjudications of misconduct.

Constraints acknowledged:

- Population denominators from Census 2011 introduce uncertainty, especially in high‑growth urban districts.
- DPR anomalies reflect statistical deviation from regional norms, not confirmed irregularity.
- Ground‑truth audit outcomes are needed to formally evaluate precision/recall of detection methods.

---

## Limitations & Roadmap

**Current limitations:**

- Cross‑sectional analysis (single snapshot); temporal patterns not yet modelled.
- RGI growth rates not fully integrated; PNA uses approximate growth bounds.
- Unsupervised detectors operate without labelled anomalies; evaluation is overlap‑ and plausibility‑based.

**Planned extensions:**

- Time‑series anomaly detection on quarterly update streams.
- LOF as complementary detector to Isolation Forest.
- HDBSCAN clustering for geospatial anomaly regions.
- Sensitivity analysis over detector parameters and composite weights.
- Full integration of state/district‑level RGI growth rates.
- Improved mapping and EDA for IndiaID‑Bench.

---

## Tech Stack

| Component              | Technology                    |
|------------------------|-------------------------------|
| Data Engineering       | Python, Pandas, NumPy         |
| Anomaly Detection      | scikit‑learn (IsolationForest; LOF planned) |
| Statistics             | SciPy (Z‑score normalisation) |
| Geospatial Visuals     | Folium, plugins               |
| Interactive Dashboard  | Streamlit                     |
| Testing                | pytest                        |
| Version Control        | Git                           |

---

## Project Structure Rationale

- **`src/config.py`** — single source of truth for thresholds, weights, paths; no magic numbers in pipeline code.
- **`src/engine.py`** — pure metric computation; no anomaly logic.
- **`src/scoring.py`** — anomaly detection and composite scoring; easy to test and extend.
- **`pipeline.py`** — orchestration; regenerates all processed data in one command.  
- **`app.py`** — read‑only dashboard; never mutates underlying data.

---

## Acknowledgements

Population estimation currently uses Census of India 2011 district statistics as a scalable proxy for PINCODE‑level population. This is suitable for research and prototyping; production deployment in a government context would require official registry denominators and policy review.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built as part of applied research in privacy‑preserving analytics and AI security.*