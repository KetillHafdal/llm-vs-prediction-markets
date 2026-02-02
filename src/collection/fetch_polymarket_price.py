#Þessi virkar, tekur stök úr open_markets_filtered.json og gefur mér prísinn sem er í dag.
#Dose not give a market id, have to fix that, good enough for now

import json
import requests
from datetime import datetime
import time

# -------------------- CONFIG -------------------- #
INPUT_FILE = "open_markets_wip.json" #Markets_polymarket_featcher.py
OUTPUT_FILE_TEMPLATE = "token_prices_snapshot(this is 11.08){}.json"
PRICE_URL = "https://clob.polymarket.com/price"
REQUEST_DELAY = 0.2  # Short delay to respect API

# -------------------- FUNCTIONS -------------------- #
def get_token_price(token_id, side="buy"):
    """Fetch current price for a token_id and side (buy/sell)."""
    params = {"token_id": token_id, "side": side}
    resp = requests.get(PRICE_URL, params=params)
    if resp.status_code == 200:
        return resp.json().get("price")
    else:
        print(f"❌ Error fetching price for token {token_id}: {resp.status_code}")
        return None

def main():
    # Load markets
    with open(INPUT_FILE) as f:
        markets = json.load(f)
    print(f"Loaded {len(markets)} markets from {INPUT_FILE}")

    snapshot_time = datetime.utcnow().isoformat() + "Z"
    snapshot = {
        "timestamp": snapshot_time,
        "markets": []
    }

    for m in markets:
        market_id = m.get("market_id") or m.get("condition_id") or "unknown"
        question = m.get("question", "unknown")

        # Parse clobTokenIds safely
        clob_token_ids_raw = m.get("clobTokenIds")
        if not clob_token_ids_raw:
            print(f"⚠️ Missing clobTokenIds for market {market_id}")
            continue

        try:
            clob_token_ids = json.loads(clob_token_ids_raw)
            yes_token_id = clob_token_ids[0] if len(clob_token_ids) > 0 else None
            no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else None
        except json.JSONDecodeError:
            print(f"⚠️ Could not decode clobTokenIds for market {market_id}")
            continue

        if not yes_token_id or not no_token_id:
            print(f"⚠️ Missing token_ids for market {market_id}")
            continue

        print(f"Fetching prices for market: {question}")

        yes_price = get_token_price(yes_token_id, side="buy")
        no_price = get_token_price(no_token_id, side="buy")

        snapshot["markets"].append({
            "market_id": market_id,
            "question": question,
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id,
            "yes_price": yes_price,
            "no_price": no_price
        })

        time.sleep(REQUEST_DELAY)

    # Save snapshot
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    output_file = OUTPUT_FILE_TEMPLATE.format(date_str)

    with open(output_file, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"✅ Saved snapshot of {len(snapshot['markets'])} markets to {output_file}")

# -------------------- ENTRYPOINT -------------------- #
if __name__ == "__main__":
    main()
