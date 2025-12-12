import os, re, ast
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- Config ----------------
INPUT = "outputs/drug_reviews_with_matchedADRs.csv"
OUT_DIR = "outputs/insights"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
REPORT_DIR = os.path.join(OUT_DIR, "drug_reports")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Severity dictionary
MILD = {"headache", "nausea", "rash", "vomiting", "diarrhoea", "constipation",
        "dizziness", "fatigue", "insomnia", "pruritus", "abdominal pain"}
SERIOUS = {"haemorrhage", "hemorrhage", "arrhythmia", "rhabdomyolysis",
           "suicidal thought", "suicide", "hepatitis", "anaphylaxis",
           "sepsis", "kidney failure", "death"}

# Sentiment mapping
SENT_MAP = {
    "LABEL_0": "Negative", "LABEL_1": "Positive", "LABEL_2": "Neutral",
    "NEGATIVE": "Negative", "POSITIVE": "Positive", "NEUTRAL": "Neutral",
    "Negative": "Negative", "Positive": "Positive", "Neutral": "Neutral"
}

def safe_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', "_", str(name))

def safe_parse_list(x):
    if pd.isna(x): return []
    if isinstance(x, list): items = x
    elif isinstance(x, str):
        s = x.strip()
        try:
            if s.startswith("[") and s.endswith("]"):
                items = ast.literal_eval(s)
            else:
                items = re.split(r",\s*", s)
        except: items = []
    else: return []
    return [str(i).lower().strip() for i in items if str(i).strip()]

def classify_severity(adrs):
    mild = serious = unknown = 0
    for a in adrs:
        if a in MILD: mild += 1
        elif a in SERIOUS: serious += 1
        else: unknown += 1
    return {"mild": mild, "serious": serious, "unknown": unknown}

# ---------------- Load ----------------
print("Loading data:", INPUT)
df = pd.read_csv(INPUT, low_memory=False)
print("Rows loaded:", len(df))

for col in ["drugName", "review", "matched_ADRs", "sentiment", "date"]:
    if col not in df.columns: df[col] = pd.NA

df["sentiment"] = df["sentiment"].astype(str).map(SENT_MAP).fillna(df["sentiment"])
df["adrs_list"] = df["matched_ADRs"].apply(safe_parse_list)
df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
df["month"] = df["date_parsed"].dt.to_period("M").astype(str)

# Explode ADRs
df_exploded = df.explode("adrs_list").rename(columns={"adrs_list":"adr"})
df_exploded = df_exploded.dropna(subset=["adr"])
print("Exploded rows:", len(df_exploded))

# ---------------- Per-drug aggregates ----------------
profile_rows, top_adrs_rows, monthly_rows = [], [], []

for drug, grp in tqdm(df.groupby("drugName"), desc="Drugs"):
    total_reviews = len(grp)
    adr_reviews = len(grp[grp["adr"].dropna()]) if "adr" in grp else len(grp[grp["matched_ADRs"].str.len()>0])
    adr_percent = (adr_reviews / total_reviews * 100) if total_reviews else 0

    adrs_for_drug = df_exploded[df_exploded["drugName"]==drug]["adr"].dropna().tolist()
    c = Counter(adrs_for_drug)
    top_adrs = c.most_common(20)

    srs = grp["sentiment"].fillna("Neutral")
    neg, pos, neu = (srs=="Negative").sum(), (srs=="Positive").sum(), (srs=="Neutral").sum()
    denom = max(1, (neg+pos+neu))
    neg_ratio, pos_ratio, neu_ratio = neg/denom, pos/denom, neu/denom

    sev = classify_severity(adrs_for_drug)

    profile_rows.append({
        "drugName": drug, "total_reviews": total_reviews,
        "adr_reviews": adr_reviews, "adr_percent": adr_percent,
        "neg_sentiment": neg, "pos_sentiment": pos, "neu_sentiment": neu,
        "neg_ratio": neg_ratio, "pos_ratio": pos_ratio, "neu_ratio": neu_ratio,
        "mild_count": sev["mild"], "serious_count": sev["serious"],
        "unknown_severity_count": sev["unknown"]
    })

    for adr, count in top_adrs:
        top_adrs_rows.append({"drugName": drug, "adr": adr, "count": count})

    drug_expl = df_exploded[df_exploded["drugName"] == drug]
    if not drug_expl.empty:
        monthly = drug_expl.groupby("month")["adr"].count().reset_index().rename(columns={"adr":"adr_count"})
        monthly["drugName"] = drug
        monthly_rows.append(monthly)

df_profile = pd.DataFrame(profile_rows).sort_values("adr_percent", ascending=False)
df_top_adrs = pd.DataFrame(top_adrs_rows)
df_monthly = pd.concat(monthly_rows, ignore_index=True)

# ---------------- Spike Detection ----------------
print("Detecting spikes...")
spike_rows = []
for drug, grp in df_monthly.groupby("drugName"):
    counts = grp.sort_values("month")["adr_count"].values.astype(float)
    if len(counts) < 5: continue
    algo = rpt.Pelt(model="rbf").fit(counts)
    cps = algo.predict(pen=5)
    for cp in cps:
        if cp < len(counts):
            spike_rows.append({"drugName": drug, "month": grp.iloc[cp]["month"], "adr_count": counts[cp]})
df_spikes = pd.DataFrame(spike_rows)

# ---------------- Save ----------------
df_profile.to_csv(os.path.join(OUT_DIR,"drug_profile_summary.csv"), index=False)
df_top_adrs.to_csv(os.path.join(OUT_DIR,"drug_top_adrs.csv"), index=False)
df_monthly.to_csv(os.path.join(OUT_DIR,"drug_monthly_trends.csv"), index=False)
df_spikes.to_csv(os.path.join(OUT_DIR,"drug_spikes.csv"), index=False)

# ---------------- Plots ----------------
sns.set(style="whitegrid")
top_overall = df_exploded["adr"].value_counts().head(25)
plt.figure(figsize=(8,10))
sns.barplot(y=top_overall.index[::-1], x=top_overall.values[::-1], palette="Reds_r", hue=top_overall.index[::-1], legend=False)
plt.title("Top 25 ADRs (overall)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR,"top25_adrs.png")); plt.close()

cands = df_profile[df_profile["total_reviews"]>=50].head(20)
plt.figure(figsize=(8,10))
sns.barplot(x=cands["adr_percent"], y=cands["drugName"], palette="Blues_r", hue=cands["drugName"], legend=False)
plt.title("Top 20 Drugs by ADR % (min reviews=50)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR,"top20_drugs.png")); plt.close()

# ---------------- Reports ----------------
print("Generating PDF reports...")
styles = getSampleStyleSheet()
for drug in df_profile.head(10)["drugName"]:
    doc = SimpleDocTemplate(os.path.join(REPORT_DIR,f"{safe_filename(drug)}_report.pdf"), pagesize=A4)
    story = [Paragraph(f"<b>Drug Safety Report: {drug}</b>", styles["Title"]), Spacer(1,12)]
    row = df_profile[df_profile["drugName"]==drug].iloc[0].to_dict()
    summary_data = [
        ["Total Reviews", row["total_reviews"]],
        ["ADR Reviews", row["adr_reviews"]],
        ["% ADR Mentions", f"{row['adr_percent']:.2f}%"],
        ["Positive %", f"{row['pos_ratio']*100:.1f}%"],
        ["Negative %", f"{row['neg_ratio']*100:.1f}%"],
        ["Mild ADRs", row["mild_count"]],
        ["Serious ADRs", row["serious_count"]],
    ]
    story.append(Table(summary_data)); story.append(Spacer(1,12))
    top_adrs = df_top_adrs[df_top_adrs["drugName"]==drug].head(10)
    story.append(Paragraph("<b>Top ADRs</b>", styles["Heading2"]))
    story.append(Table([["ADR","Mentions"]]+top_adrs.values.tolist())); story.append(Spacer(1,12))
    trend = df_monthly[df_monthly["drugName"]==drug].sort_values("month")
    if not trend.empty:
        plt.figure(figsize=(6,3))
        plt.plot(trend["month"], trend["adr_count"], marker="o")
        plt.xticks(rotation=45,fontsize=6); plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR,f"{safe_filename(drug)}_trend.png")
        plt.savefig(plot_path); plt.close()
        story.append(Image(plot_path, width=400, height=200))
    doc.build(story)

print("All outputs saved to", OUT_DIR)
