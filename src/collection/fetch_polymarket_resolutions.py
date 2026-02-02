import csv
import time
from typing import Optional, Dict

import requests
import pandas as pd

# ------------------ CONFIG (EDIT THESE IF NEEDED) ------------------
# Your input CSV and where to save the merged file:
INPUT_CSV  = r"C:\Users\ketil\OneDrive\Desktop\TU Delft\Y2\Thesis\master_forecasting_dataset2.csv"
OUTPUT_CSV = r"C:\Users\ketil\OneDrive\Desktop\TU Delft\Y2\Thesis\master_forecasting_dataset2_with_resolution.csv"

# If your ID column is named differently, add it here (first match is used)
ID_COL_CANDIDATES = ["id", "market_id"]

# Polymarket endpoint for individual markets
API_TEMPLATE = "https://gamma-api.polymarket.com/markets/{mid}"

# Request settings
TIMEOUT_S = 12
MAX_RETRIES = 4
RETRY_BACKOFF_S = 1.5
POLITENESS_DELAY_S = 0.05
# -------------------------------------------------------------------


def pick_id_col(df: pd.DataFrame) -> str:
    for c in ID_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"None of {ID_COL_CANDIDATES} found in columns: {list(df.columns)}")


def norm_id_series(s: pd.Series) -> pd.Series:
    """
    Normalize IDs to clean strings for merging:
    - If numeric-like, round to integer and cast to string
    - Strip trailing '.0'
    """
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s):
        s = pd.to_numeric(s, errors="coerce").round(0).astype("Int64").astype(str)
    else:
        s = s.astype(str)
    return s.str.replace(r"\.0$", "", regex=True).str.strip()


def fetch_one(mid: str) -> Dict[str, Optional[int]]:
    """
    Fetch one market and return {'id': mid, 'resolution': 0/1/None}.
    Maps common outcome fields to {yes:1, no:0}. Unresolved → None.
    """
    url = API_TEMPLATE.format(mid=mid)
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT_S, headers={"Accept": "application/json"})
            if r.status_code == 200:
                data = r.json()

                # Try a few field names that might contain the final outcome
                outcome = (
                    data.get("outcome")
                    or data.get("result")
                    or data.get("finalOutcome")
                    or data.get("resolution")
                )

                res = None
                if isinstance(outcome, str):
                    o = outcome.strip().lower()
                    if o == "yes":
                        res = 1
                    elif o == "no":
                        res = 0
                elif isinstance(outcome, bool):
                    res = 1 if outcome else 0

                return {"id": str(mid), "resolution": res}

            elif r.status_code in (429, 500, 502, 503, 504):
                time.sleep((RETRY_BACKOFF_S ** attempt))
            else:
                # Non-retryable HTTP
                return {"id": str(mid), "resolution": None}

        except Exception as e:
            last_exc = e
            time.sleep((RETRY_BACKOFF_S ** attempt))

    # After retries
    print(f"[warn] giving up on {mid}: {last_exc}")
    return {"id": str(mid), "resolution": None}


def main():
    # --- Load CSV and pick ID column ---
    df = pd.read_csv(INPUT_CSV)
    id_col = pick_id_col(df)

    # Normalize ID column to string (prevents merge dtype errors)
    df[id_col] = norm_id_series(df[id_col])

    # Unique market IDs
    mids = pd.Series(df[id_col].dropna().astype(str).unique())
    print(f"Found {len(mids)} unique markets. Fetching resolutions…")

    # --- Fetch outcomes ---
    rows = []
    for i, mid in enumerate(mids, 1):
        rows.append(fetch_one(mid))
        if i % 50 == 0:
            print(f"  fetched {i}/{len(mids)}…")
        time.sleep(POLITENESS_DELAY_S)

    res_df = pd.DataFrame(rows)

    # Align column names and normalize IDs on result side too
    if id_col != "id":
        res_df = res_df.rename(columns={"id": id_col})
    res_df[id_col] = norm_id_series(res_df[id_col])

    # --- Merge ---
    merged = df.merge(res_df, on=id_col, how="left")

    # --- Quick summary (unique markets) ---
    uniq = merged.drop_duplicates(subset=[id_col])
    total = len(uniq)
    resolved = uniq["resolution"].notna().sum()
    yes = uniq["resolution"].eq(1).sum()
    no = uniq["resolution"].eq(0).sum()
    unresolved = total - resolved

    print("\nSummary (unique markets)")
    print(f"  total:      {total}")
    print(f"  resolved:   {resolved} (YES={yes}, NO={no})")
    print(f"  unresolved: {unresolved}")

    # Save unresolved IDs for review
    unresolved_ids = uniq.loc[uniq["resolution"].isna(), id_col].tolist()
    if unresolved_ids:
        pd.DataFrame({id_col: unresolved_ids}).to_csv(
            OUTPUT_CSV.replace(".csv", "_unresolved_ids.csv"), index=False
        )
        print(f"  → wrote unresolved IDs to {OUTPUT_CSV.replace('.csv', '_unresolved_ids.csv')}")

    # --- Save merged file ---
    merged.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\n✅ Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()