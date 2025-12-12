import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("outputs/drug_reviews_with_matchedADRs.csv").dropna(subset=["matched_ADRs"])

# Convert string list → Python list
def safe_eval(x):
    try:
        if isinstance(x, str) and x.startswith("["):
            return eval(x)
        elif isinstance(x, str):
            return [x]
        elif isinstance(x, list):
            return x
        else:
            return []
    except:
        return []

df["matched_ADRs"] = df["matched_ADRs"].apply(safe_eval)

print(f"Dataset size: {len(df)}")
print(f"Unique drugs: {df['drugName'].nunique()}")
print(f"ADR rows: {df['ADR_flag'].sum()}")

# ---------------- SENTIMENT NORMALIZATION ----------------
sentiment_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "NEGATIVE": "Negative",
    "NEUTRAL": "Neutral",
    "POSITIVE": "Positive"
}
df["sentiment"] = df["sentiment"].str.upper().map(sentiment_map).fillna("Neutral")

# ---------------- ADR FREQUENCY PER DRUG ----------------
all_adrs = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Collecting ADRs"):
    for adr in row["matched_ADRs"]:
        all_adrs.append((row["drugName"], adr))

adr_df = pd.DataFrame(all_adrs, columns=["drugName", "ADR"])

# Top ADRs overall
top_adrs = adr_df["ADR"].value_counts().head(20)
print("\nTop 20 ADRs overall:")
print(top_adrs)

# Top ADRs per drug
top_adrs_per_drug = (
    adr_df.groupby("drugName")["ADR"]
    .apply(lambda x: x.value_counts().head(5).to_dict())
    .reset_index()
    .rename(columns={"ADR": "Top_ADRs"})
)

# ---------------- SENTIMENT PER DRUG ----------------
sentiment_per_drug = (
    df.groupby("drugName")["sentiment"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)

# Merge ADR + Sentiment
drug_analysis = pd.merge(top_adrs_per_drug, sentiment_per_drug, on="drugName", how="left")

# ---------------- SAVE ----------------
out_path = "outputs/drug_ADR_sentiment_summary.csv"
drug_analysis.to_csv(out_path, index=False)
print(f"\nSaved drug-level ADR + sentiment summary -> {out_path}")

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(12, 6))
sns.barplot(x=top_adrs.values, y=top_adrs.index, palette="Reds_r")
plt.title("Top 20 Most Common ADRs")
plt.xlabel("Frequency")
plt.ylabel("ADR")
plt.tight_layout()
plt.savefig("outputs/top20_ADRs.png")
print(" Saved plot -> outputs/top20_ADRs.png")

# Sentiment distribution overall
sent_counts = df["sentiment"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(sent_counts, labels=sent_counts.index, autopct='%1.1f%%', colors=["#ff6b6b","#feca57","#1dd1a1"])
plt.title("Overall Sentiment Distribution")
plt.savefig("outputs/sentiment_distribution.png")
print("Saved plot -> outputs/sentiment_distribution.png")

# Example: Top 10 drugs with most ADR mentions
drug_counts = adr_df["drugName"].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=drug_counts.values, y=drug_counts.index, palette="Blues_r")
plt.title("Top 10 Drugs with Most ADR Mentions")
plt.xlabel("ADR Mentions")
plt.ylabel("Drug")
plt.tight_layout()
plt.savefig("outputs/top10_drugs_ADRs.png")
print("Saved plot -> outputs/top10_drugs_ADRs.png")

print("\n Analysis + Visualization complete!")
