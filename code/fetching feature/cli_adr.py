# cli_adr.py
import pandas as pd
from utils_pipeline import analyze_reviews, summarize_results

# Load dataset
df = pd.read_csv("/Users/kunikabhadra/Music/ADR-MINING/outputs/drug_reviews_with_matchedADRs.csv")

def get_reviews_for_drug(drug_name, max_reviews=500):
    drug_reviews = df[df["drugName"].str.lower() == drug_name.lower()]
    return drug_reviews.head(max_reviews)

if __name__ == "__main__":
    drug_name = input("Enter a drug name: ").strip()
    reviews = get_reviews_for_drug(drug_name)

    if reviews.empty:
        print(f"No reviews found for {drug_name}")
    else:
        results = analyze_reviews(reviews["clean_review"].tolist())
        summary = summarize_results(results)
print("\nSummary:")
for k, v in summary.items():
    print(f"{k}: {v}")

from utils_pipeline import recommendation
print("\nRecommendation:", recommendation(summary))

