import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load preprocessed
df = pd.read_csv("/Users/kunikabhadra/Music/ADR-MINING/outputs/drug_re_views_clean_all.csv").dropna(subset=["clean_review"])

# HuggingFace pipelines
ner = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

tqdm.pandas()

def extract_entities(text):
    try:
        results = ner(text)
        return [r["word"] for r in results]
    except:
        return []

def get_sentiment(text):
    try:
        return sentiment(text[:512])[0]["label"]
    except:
        return "NEUTRAL"

print(" Running NER + Sentiment...")
df["entities"] = df["clean_review"].astype(str).progress_apply(extract_entities)
df["sentiment"] = df["clean_review"].astype(str).progress_apply(get_sentiment)
df["ADR_flag"] = df["entities"].apply(lambda x: 1 if len(x) > 0 else 0)

# Save enriched
df.to_csv("outputs/drug_reviews_with_nlp.csv", index=False)
print("Enriched -> outputs/drug_reviews_with_n_lp.csv")
