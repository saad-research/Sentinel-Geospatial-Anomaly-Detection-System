"""
scripts/build_district_map.py
District Name Reconciliation — Aadhaar Sentinel V2 (v2: overrides + fuzzy fallback)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from rapidfuzz import process, fuzz

from src import config
from src.loader import load_and_concat_csvs

MATCH_THRESHOLD = 85

STATE_OVERRIDES = {
    "JAMMU AND KASHMIR": "JAMMU & KASHMIR",
    "WEST BANGAL": "WEST BENGAL",
    "WESTBENGAL": "WEST BENGAL",
    "WEST BENGLI": "WEST BENGAL",
    "WEST  BENGAL": "WEST BENGAL",
    "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY",
    "LADAKH": "JAMMU & KASHMIR",
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA & NAGAR HAVELI",
}

DISTRICT_OVERRIDES = {
    ("WEST BENGAL", "HOOGHLY"): "HUGLI", ("WEST BENGAL", "HOOGHIY"): "HUGLI",
    ("WEST BENGAL", "HOWRAH"): "HAORA", ("WEST BENGAL", "HAWRAH"): "HAORA",
    ("WEST BENGAL", "COOCH BEHAR"): "KOCH BIHAR", ("WEST BENGAL", "COOCHBEHAR"): "KOCH BIHAR",
    ("WEST BENGAL", "DARJEELING"): "DARJILING",
    ("WEST BENGAL", "BURDWAN"): "BARDDHAMAN",
    ("WEST BENGAL", "WEST MIDNAPORE"): "PASCHIM MEDINIPUR",
    ("WEST BENGAL", "MEDINIPUR WEST"): "PASCHIM MEDINIPUR",
    ("WEST BENGAL", "WEST MEDINIPUR"): "PASCHIM MEDINIPUR",
    ("WEST BENGAL", "24 PARAGANAS NORTH"): "NORTH TWENTY FOUR PARGANAS",
    ("WEST BENGAL", "24 PARAGANAS SOUTH"): "SOUTH TWENTY FOUR PARGANAS",
    ("KARNATAKA", "MYSURU"): "MYSORE",
    ("KARNATAKA", "BENGALURU"): "BANGALORE", ("KARNATAKA", "BENGALURU URBAN"): "BANGALORE",
    ("KARNATAKA", "BENGALURU RURAL"): "BANGALORE RURAL",
    ("KARNATAKA", "BELAGAVI"): "BELGAUM",
    ("KARNATAKA", "SHIVAMOGGA"): "SHIMOGA",
    ("KARNATAKA", "BALLARI"): "BELLARY",
    ("KARNATAKA", "VIJAYAPURA"): "BIJAPUR",
    ("HARYANA", "GURUGRAM"): "GURGAON",
    ("PUNJAB", "FEROZEPUR"): "FIROZPUR",
    ("ODISHA", "BOUDH"): "BAUDH", ("ODISHA", "ANUGAL"): "ANUGUL",
    ("ASSAM", "SIBSAGAR"): "SIVASAGAR",
    ("JAMMU & KASHMIR", "BUDGAM"): "BADGAM",
    ("JAMMU & KASHMIR", "BANDIPUR"): "BANDIPORE",
    ("JAMMU & KASHMIR", "POONCH"): "PUNCH",
    ("JAMMU & KASHMIR", "SHOPIAN"): "SHUPIYAN",
    ("MAHARASHTRA", "AHILYANAGAR"): "AHMADNAGAR",
    ("ANDHRA PRADESH", "Y. S. R."): "Y.S.R.", ("ANDHRA PRADESH", "CUDDAPAH"): "Y.S.R.",
    ("ANDHRA PRADESH", "K.V.RANGAREDDY"): "RANGAREDDY",
    ("BIHAR", "PURNEA"): "PURNIA", ("BIHAR", "MONGHYR"): "MUNGER",
    ("HIMACHAL PRADESH", "LAHAUL AND SPITI"): "LAHUL & SPITI",
    ("PUDUCHERRY", "PONDICHERRY"): "PUDUCHERRY",
    ("GUJARAT", "DAHOD"): "DOHAD",
    ("MADHYA PRADESH", "NARSINGHPUR"): "NARSIMHAPUR",
    ("SIKKIM", "NORTH SIKKIM"): "NORTH DISTRICT", ("SIKKIM", "SOUTH SIKKIM"): "SOUTH DISTRICT",
    ("SIKKIM", "EAST SIKKIM"): "EAST DISTRICT", ("SIKKIM", "WEST SIKKIM"): "WEST DISTRICT",
}

TELANGANA_TO_AP_PARENT = {
    "ADILABAD": "ADILABAD", "KOMARAM BHEEM": "ADILABAD", "MANCHERIAL": "ADILABAD", "NIRMAL": "ADILABAD",
    "HYDERABAD": "HYDERABAD",
    "KARIMNAGAR": "KARIMNAGAR", "JAGITIAL": "KARIMNAGAR", "JAGTIAL": "KARIMNAGAR",
    "PEDDAPALLI": "KARIMNAGAR", "RAJANNA SIRCILLA": "KARIMNAGAR",
    "KHAMMAM": "KHAMMAM", "BHADRADRI KOTHAGUDEM": "KHAMMAM",
    "MAHABUBNAGAR": "MAHBUBNAGAR", "MAHABUBABAD": "WARANGAL",
    "JOGULAMBA GADWAL": "MAHBUBNAGAR", "NAGARKURNOOL": "MAHBUBNAGAR",
    "WANAPARTHY": "MAHBUBNAGAR", "NARAYANPET": "MAHBUBNAGAR",
    "MEDAK": "MEDAK", "SANGAREDDY": "MEDAK", "SIDDIPET": "MEDAK",
    "NALGONDA": "NALGONDA", "SURYAPET": "NALGONDA", "YADADRI": "NALGONDA",
    "NIZAMABAD": "NIZAMABAD", "KAMAREDDY": "NIZAMABAD",
    "RANGAREDDY": "RANGAREDDY", "RANGA REDDY": "RANGAREDDY", "VIKARABAD": "RANGAREDDY",
    "MEDCHAL MALKAJGIRI": "RANGAREDDY", "MEDCHAL-MALKAJGIRI": "RANGAREDDY",
    "K.V. RANGAREDDY": "RANGAREDDY",
    "WARANGAL": "WARANGAL", "WARANGAL URBAN": "WARANGAL", "WARANGAL RURAL": "WARANGAL",
    "WARANGAL (URBAN)": "WARANGAL", "HANUMAKONDA": "WARANGAL",
    "JANGAON": "WARANGAL", "JANGOAN": "WARANGAL",
    "JAYASHANKAR BHUPALPALLY": "WARANGAL", "MULUGU": "WARANGAL",
}


def normalize(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def main():
    print("Loading UIDAI data to extract unique (state, district) pairs...")
    frames = []
    for folder in [
        "aadhaar_demographic_updates",
        "aadhaar_biometric_update_pincode",
        "aadhaar_enrolment_pincode",
    ]:
        df = load_and_concat_csvs(f"{config.RAW_DATA_DIR}/{folder}")
        df["district"] = normalize(df["district"])
        df["state"] = normalize(df["state"])
        frames.append(df[["state", "district"]])

    uidai_pairs = pd.concat(frames).drop_duplicates().reset_index(drop=True)
    print(f"  Unique UIDAI (state, district) pairs: {len(uidai_pairs)}")

    census = pd.read_csv(config.CENSUS_PATH)
    census["district"] = normalize(census["District name"])
    census["state"] = normalize(census["State name"])

    census_states = sorted(census["state"].unique())
    census_districts_by_state = {
        s: sorted(census.loc[census["state"] == s, "district"].unique())
        for s in census_states
    }

    state_match_cache = {}

    def fuzzy_match_state(uidai_state):
        if uidai_state in state_match_cache:
            return state_match_cache[uidai_state]
        if uidai_state in census_states:
            result = (uidai_state, 100)
        else:
            best = process.extractOne(uidai_state, census_states, scorer=fuzz.WRatio)
            result = (best[0], best[1]) if best else (None, 0)
        state_match_cache[uidai_state] = result
        return result

    def fuzzy_match_district(uidai_district, matched_state):
        candidates = census_districts_by_state.get(matched_state, [])
        if not candidates:
            return None, 0
        if uidai_district in candidates:
            return uidai_district, 100
        best = process.extractOne(uidai_district, candidates, scorer=fuzz.WRatio)
        return (best[0], best[1]) if best else (None, 0)

    print("Resolving (overrides -> Telangana crosswalk -> fuzzy fallback)...")
    rows = []
    for _, r in uidai_pairs.iterrows():
        raw_state, raw_district = r["state"], r["district"]

        state = STATE_OVERRIDES.get(raw_state, raw_state)

        if raw_state == "TELANGANA" and raw_district in TELANGANA_TO_AP_PARENT:
            parent = TELANGANA_TO_AP_PARENT[raw_district]
            c_district, d_score = fuzzy_match_district(parent, "ANDHRA PRADESH")
            rows.append({
                "uidai_state": raw_state, "uidai_district": raw_district,
                "census_state": "ANDHRA PRADESH", "census_district": c_district,
                "state_score": 100, "district_score": d_score,
                "match_method": "telangana_proxy",
            })
            continue

        if (state, raw_district) in DISTRICT_OVERRIDES:
            hint = DISTRICT_OVERRIDES[(state, raw_district)]
            c_district, d_score = fuzzy_match_district(hint, state)
            rows.append({
                "uidai_state": raw_state, "uidai_district": raw_district,
                "census_state": state, "census_district": c_district,
                "state_score": 100, "district_score": d_score,
                "match_method": "manual_override",
            })
            continue

        c_state, state_score = fuzzy_match_state(state)
        c_district, district_score = fuzzy_match_district(raw_district, c_state)
        method = "fuzzy_auto" if district_score >= MATCH_THRESHOLD else "fuzzy_review"
        rows.append({
            "uidai_state": raw_state, "uidai_district": raw_district,
            "census_state": c_state, "census_district": c_district,
            "state_score": state_score, "district_score": district_score,
            "match_method": method,
        })

    out = pd.DataFrame(rows)
    out["needs_review"] = out["match_method"] == "fuzzy_review"
    out.to_csv("data/raw/district_name_map.csv", index=False)

    print(f"\nSaved {len(out)} mappings -> data/raw/district_name_map.csv")
    print(out["match_method"].value_counts().to_string())
    print(f"\nStill needs manual review: {int(out['needs_review'].sum())}")

    if out["needs_review"].sum() > 0:
        print("\nRemaining low-confidence rows:")
        print(
            out[out["needs_review"]]
            .sort_values("district_score")
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
