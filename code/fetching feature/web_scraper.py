import requests
import pandas as pd
import os

# Serper API key (WARNING: better to keep in ENV variable)
SERPER_API_KEY = "de9bf41c5e1089bd3f721a7f054a5e12e30b79f7"
SEARCH_URL = "https://google.serper.dev/search"

def web_search_reviews(drug_name, num_results=20, max_results=None):
    if max_results is not None:
        num_results = max_results

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {
        "q": f"{drug_name} reviews site:drugs.com OR site:reddit.com OR site:healthgrades.com"
    }

    resp = requests.post(SEARCH_URL, headers=headers, json=payload)
    resp.raise_for_status()
    results = resp.json()

    items = []
    for r in results.get("organic", [])[:num_results]:
        items.append({
            "drugName": drug_name,
            "text": r.get("snippet"),   # normalize to 'text'
            "title": r.get("title"),
            "url": r.get("link")
        })

    return pd.DataFrame(items, columns=["drugName", "text", "title", "url"])

if __name__ == "__main__":
    drug = "Cialis"
    df = web_search_reviews(drug, num_results=15)
    print(df.head())
    out_path = f"outputs/websearch_{drug}.csv"
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved web search results -> {out_path}")
