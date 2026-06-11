"""
src/loader.py
══════════════════════════════════════════════════════════════════════════════
Raw Data Ingestion — Aadhaar Sentinel V2

V2 CHANGES:
  - Replaced global warnings.filterwarnings('ignore') with targeted suppression
    of pandas DtypeWarning per-file. Global suppression hides legitimate issues.
  - Added logging so silent column mismatches become visible in the pipeline log.
  - Logic unchanged — existing column names (demo_age_5_17, bio_age_5_17, etc.)
    are the real UIDAI CSV headers.
"""

import logging
import os
import warnings

import pandas as pd
from pandas.errors import DtypeWarning

logger = logging.getLogger(__name__)


def load_and_concat_csvs(folder_path: str) -> pd.DataFrame:
    """
    Load all CSV files from a folder and concatenate into one DataFrame.

    DtypeWarning is suppressed per-file (mixed-type columns from UIDAI CSVs
    are expected). All other warnings remain visible.

    Raises FileNotFoundError if the folder does not exist.
    Returns an empty DataFrame if the folder contains no CSV files
    (pipeline.py will fail downstream with a clear error, not a silent empty run).
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"Data folder not found: '{folder_path}'. "
            f"Check that RAW_DATA_DIR in config.py points to the correct location."
        )

    csv_files = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".csv")
    )

    if not csv_files:
        logger.warning("No CSV files found in '%s'. Returning empty DataFrame.", folder_path)
        return pd.DataFrame()

    dfs = []
    for path in csv_files:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DtypeWarning)
            dfs.append(pd.read_csv(path))

    result = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Loaded %d files from '%s' → %d rows",
        len(csv_files), folder_path, len(result),
    )
    return result


def preprocess_and_aggregate(
    demo_df: pd.DataFrame,
    bio_df: pd.DataFrame,
    enr_df: pd.DataFrame,
) -> tuple:
    """
    Standardize text fields, compute row totals, and aggregate to PINCODE level.

    Aggregates three separate UIDAI update categories into one row per
    (pincode, district, state). This is the input shape expected by engine.py.

    Column expectations (real UIDAI CSV headers):
      demo_df : demo_age_5_17, demo_age_17_   → demo_total
      bio_df  : bio_age_5_17,  bio_age_17_    → bio_total
      enr_df  : age_0_5, age_5_17, age_18_greater → enrol_total

    Returns
    -------
    (demo_pin, bio_pin, enr_pin) — each aggregated per (pincode, district, state)
    """
    # Normalize text keys in-place (same as V1; loader.py mutates the DFs
    # before aggregation — this is acceptable since they're not used elsewhere)
    for df in [demo_df, bio_df, enr_df]:
        if "district" in df.columns:
            df["district"] = df["district"].astype(str).str.upper().str.strip()
        if "state" in df.columns:
            df["state"] = df["state"].astype(str).str.upper().str.strip()

    # Row totals
    demo_df["demo_total"] = (
        demo_df["demo_age_5_17"].astype(int) + demo_df["demo_age_17_"].astype(int)
    )
    bio_df["bio_total"] = (
        bio_df["bio_age_5_17"].astype(int) + bio_df["bio_age_17_"].astype(int)
    )
    enr_df["enrol_total"] = (
        enr_df["age_0_5"].astype(int)
        + enr_df["age_5_17"].astype(int)
        + enr_df["age_18_greater"].astype(int)
    )

    # Aggregate to PINCODE level
    group_keys = ["pincode", "district", "state"]
    demo_pin = demo_df.groupby(group_keys, as_index=False)["demo_total"].sum()
    bio_pin = bio_df.groupby(group_keys, as_index=False)["bio_total"].sum()
    enr_pin = enr_df.groupby(group_keys, as_index=False)["enrol_total"].sum()

    logger.info(
        "Aggregated: %d demo PINcodes | %d bio PINcodes | %d enrolment PINcodes",
        len(demo_pin), len(bio_pin), len(enr_pin),
    )
    return demo_pin, bio_pin, enr_pin