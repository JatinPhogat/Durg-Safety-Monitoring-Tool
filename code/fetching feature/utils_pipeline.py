import re
import json
import pandas as pd
from transformers import pipeline

# ---------------- Load HuggingFace pipelines ----------------
ner = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# ---------------- Load ADR Keyword Dictionary ----------------
ADR_KEYWORDS_PATH = "/Users/kunikabhadra/Music/ADR-MINING/data/adr_keywords.json"

with open(ADR_KEYWORDS_PATH, "r") as f:
    ADR_KEYWORDS = json.load(f)

# Flatten all ADR terms into a single set
ALL_ADR_TERMS = set()
for severity, terms in ADR_KEYWORDS.items():
    ALL_ADR_TERMS.update([t.lower() for t in terms])

# ---------------- Sentiment Mapping ----------------
SENTIMENT_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# ---------------- ENTITY CLEANING ----------------
def clean_and_normalize(entities):
    if not entities:
        return []

    clean_list = []
    junk_words = {"cid", "zip", "html", "llt", "pt", "_", "c",
                  "online", "file", "http", "fda", "slb", "dpd"}

    for ent in entities:
        ent = ent.lower().strip()
        ent = re.sub(r"[^a-z\s\-]", "", ent)
        ent = re.sub(r"\s+", " ", ent).strip()

        if len(ent) < 3 or ent in junk_words:
            continue

        # Normalize common noisy fragments
        replacements = {
            "bodytemperature": "body temperature",
            "diarr hoea": "diarrhoea",
            "thenia": "asthenia",
            "zziness": "dizziness",
            "emorrhage": "haemorrhage",
            "ticaria": "urticaria",
            "slr": "rash",
            "sslb": "rash",
            "chycardia": "tachycardia"
        }
        if ent in replacements:
            ent = replacements[ent]

        # Keep only if ADR-related (from json keywords)
        if ent in ALL_ADR_TERMS:
            clean_list.append(ent)

    return list(set(clean_list))

# ---------------- PIPELINE FUNCTIONS ----------------
def extract_entities(text):
    try:
        results = ner(text)
        return clean_and_normalize([r["word"] for r in results])
    except Exception:
        return []

def get_sentiment(text):
    try:
        raw = sentiment(text[:512])[0]["label"]
        return SENTIMENT_MAP.get(raw, "Neutral")
    except Exception:
        return "Neutral"

def analyze_reviews(reviews):
    results = []
    for review in reviews:
        ents = extract_entities(review)
        sent = get_sentiment(review)
        results.append({
            "review": review,
            "entities": ents,
            "sentiment": sent,
            "ADR_flag": 1 if ents else 0
        })
    return pd.DataFrame(results)

def summarize_results(df):
    summary = {
        "total_reviews": len(df),
        "adr_reviews": df["ADR_flag"].sum(),
        "% ADR_mentions": round(df["ADR_flag"].mean() * 100, 2),
        "positive %": round((df["sentiment"].value_counts(normalize=True).get("Positive", 0) * 100), 2),
        "negative %": round((df["sentiment"].value_counts(normalize=True).get("Negative", 0) * 100), 2),
        "top_ADRs": pd.Series([e for ents in df["entities"] for e in ents]).value_counts().head(10).to_dict()
    }
    return summary
# ---------------- Recommendation Logic ----------------
def get_recommendation(summary):
    adr = summary["% ADR_mentions"]
    neg = summary["negative %"]
    pos = summary["positive %"]
    total = summary["total_reviews"]

    if total < 10:
        return f"Insufficient Data (Only {total} reviews)"

    # Safe zone: up to 80% ADR mentions
    if adr <= 80:
        if pos >= 50 and neg < 30:
            return "Generally Safe (ADR mentions acceptable, sentiment mostly positive)"
        elif neg >= 30:
            return "Monitor (ADR mentions within range, but sentiment shows concerns)"
        else:
            return "Safe (ADR within normal range, sentiment balanced)"

    # Extreme ADR case (ADR > 95%)
    if adr > 95:
        if neg >= 30:
            return "Very High ADR Risk (Almost all reviews mention ADRs + significant negatives)"
        else:
            return "Caution (ADR extremely common, but sentiment not strongly negative)"

    # Risky case (ADR > 80% + strong negatives)
    if adr > 80 and neg > 40:
        return "Risky Drug (High ADR mentions + Many negative reviews)"

    # General caution (ADR > 80% but not extreme)
    if adr > 80:
        return "Caution (High ADR mentions, but sentiment is mixed)"

    return "Inconclusive — more balanced data needed"
