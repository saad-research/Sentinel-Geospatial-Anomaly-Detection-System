# IndiaID-Bench v1.0.0

PINCODE-resolution compilation of India's UIDAI (Aadhaar) public enrolment
and update statistics, joined to Census 2011 district populations projected
to 2026 via official RGI growth rates, with derived integrity/capacity
metrics and unsupervised anomaly-detection outputs.

**Author:** Mohammed Saad Shareef (ORCID: 0009-0004-1341-3557) · Independent Researcher, Hyderabad, India
**Version:** 1.0.0 · **License (this dataset):** CC BY 4.0
**Code:** https://github.com/saad-research/aadhaar-sentinel (MIT)
**Companion paper:** "Aadhaar-Sentinel: Dual-Metric Anomaly Detection for
India's Identity Infrastructure" (submitted, IEEE BigData 2026).

## Files
| File | Description |
|---|---|
| `indiaid_bench_v1.0.0.csv` | 32,898 PINCODE rows × 28 columns (schema below) |
| `district_name_map_v1.0.0.csv` | UIDAI→Census district reconciliation crosswalk with per-match provenance |
| `checksums.txt` | SHA-256 of both CSVs |

## Row semantics (read first)
Counts are **row-level**: `pincode` is NOT a unique key (a small number of
PINCODEs span multiple districts and appear as multiple rows), and
`(pincode, district, state)` is also not guaranteed unique. Do not
deduplicate by pincode or join on it as a key; use boolean masks.

## Schema
| Column | Type | Description |
|---|---|---|
| pincode, district, state | int, str, str | Identifiers; district names reconciled to Census 2011 nomenclature |
| enrol_total, demo_total, bio_total | float | Aggregate UIDAI counts over the observation window |
| TAI | float | Total Activity Index = enrol + demo + bio |
| DPR | float | Demographic-to-Biometric Processing Ratio = demo/(bio+1); integrity proxy |
| PNA (+ PNA_conservative, PNA_upper_bound) | float | Population-Normalised Activity = TAI/est. population, with uncertainty bounds |
| est_pincode_pop (+ lower/upper) | float | Projected 2026 PINCODE population (district total ÷ n PINCODEs, uniform) |
| annual_growth_rate, growth_source | float, str | Applied RGI growth rate and its provenance tier |
| dpr_z, pna_z, activity_z | float | Z-normalised metrics |
| audit_priority_score | float | 0.50·z(DPR) + 0.35·z(PNA) + 0.15·z(TAI) |
| if_flag, iso_score | bool, float | Isolation Forest flag and decision score |
| lof_flag, lof_score | bool, float | Local Outlier Factor flag and score |
| hdbscan_cluster, hdbscan_probability | Int64, float | Cluster label and soft membership; **null for non-IF-flagged rows** (clustering runs on the 658-row IF-flagged subset) |
| dbscan_cluster | Int64 | Ablation baseline cluster label; null likewise |
| Privacy_Masked | bool | k-anonymity flag (threshold 500); all False in v1.0.0 — see Privacy |

## Privacy
Only public, aggregated UIDAI data is used; no individual-level records,
identifiers, or biometric data exist at any stage. A k-anonymity mechanism
(threshold 500 on projected district population) suppresses small-population
rows from release. On the current baseline **no district falls below the
threshold, so zero rows are suppressed** and this release is the complete
table; the mechanism remains active for future versions.

## Reproducibility
Key counts recomputable from this file: 32,898 rows; 1,918 statistical
flags (98th-percentile union on TAI/DPR/PNA); 658 IF flags (subset of the
statistical set); 623 LOF flags (54 overlap with IF); 33 HDBSCAN clusters
over the IF-flagged subset. The full pipeline (fixed seeds, 67 tests) is at
the repository above. Verify file integrity against `checksums.txt`.

## Upstream data attribution (GODL-India)
The underlying enrolment, demographic-update, and biometric-update
statistics are published by the Unique Identification Authority of India
(UIDAI) on data.gov.in under the Government Open Data License – India
(GODL). This dataset is a derived work; attribution to UIDAI/data.gov.in
is hereby given as required. **UIDAI does not endorse this work or its
findings.** Census 2011 Primary Census Abstract and the RGI/NCP report
*Population Projections for India and States 2011–2036* are publications
of the Office of the Registrar General of India.

## Interpretation caution
All flags are statistical deviations, not adjudications of misconduct.
High DPR admits legitimate explanations; ground-truth audit outcomes would
be required to distinguish them.

## AI assistance disclosure
Anthropic Claude (chat interface and Claude Code) assisted in implementing
the pipeline and preparing documentation. All design decisions and
verification are the author's.

## Citation
Please cite this dataset (DOI above) and the companion paper.

## Changelog
- v1.0.0 — initial release accompanying the IEEE BigData 2026 submission.