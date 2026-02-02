import requests
import json

# API endpoint and parameters
API_URL = "https://gamma-api.polymarket.com/markets"
LIMIT = 300  # Number of markets
OFFSET = 0   # Offset for pagination

# Example date range (adjust as needed)
END_DATE_MIN = "2025-08-30T00:00:00Z"
END_DATE_MAX = "2025-09-15T23:59:59Z"

# Example volume filter (adjust as needed)
VOLUME_MIN = 100000
VOLUME_MAX = None  # Set to None if no upper limit

def fetch_markets(limit=100, offset=0, end_date_min=None, end_date_max=None, volume_min=None, volume_max=None):
    """Fetch active markets filtered by end date and volume."""
    params = {
        "limit": limit,
        "offset": offset,
        "active": "true"
    }
    if end_date_min:
        params["end_date_min"] = end_date_min
    if end_date_max:
        params["end_date_max"] = end_date_max
    if volume_min is not None:
        params["volume_num_min"] = volume_min
    if volume_max is not None:
        params["volume_num_max"] = volume_max

    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    return response.json()

def main():
    print(f"Fetching {LIMIT} active markets with end dates between {END_DATE_MIN} and {END_DATE_MAX}...")
    markets = fetch_markets(
        limit=LIMIT,
        offset=OFFSET,
        end_date_min=END_DATE_MIN,
        end_date_max=END_DATE_MAX,
        volume_min=VOLUME_MIN,
        volume_max=VOLUME_MAX
    )

    # Now filter for markets that are NOT closed
    open_markets = [m for m in markets if not m.get("closed", True)]
    print(f"Found {len(open_markets)} markets that are still ongoing (not closed).")

    with open("open_markets_filtered_test.json", "w") as f:
        json.dump(open_markets, f, indent=2)
    print("Saved ongoing markets to 'open_markets_filtered_test.json'.")

if __name__ == "__main__":
    main()
