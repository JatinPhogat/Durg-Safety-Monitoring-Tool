import pandas as pd
import re
from tqdm import tqdm

# --------------------------------- PATHS ------------------------------------
train_path = "/Users/kunikabhadra/Music/ADR-MINING/data/drugsComTrain_raw.csv"
test_path  = "/Users/kunikabhadra/Music/ADR-MINING/data/drugsComTest_raw.csv"
webmd_path = "/Users/kunikabhadra/Music/ADR-MINING/data/webmd.csv"

output_path = "/Users/kunikabhadra/Music/ADR-MINING/outputs/drug_re_views_clean_all.csv"

# ---------------- CLEAN FUNCTION ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)        # remove HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep only alphanum
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text

# ---------------- LOAD DATA ----------------
def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")

print("Loading datasets...")
train = safe_read_csv(train_path)
test  = safe_read_csv(test_path)
webmd = safe_read_csv(webmd_path)

print(f"Train: {len(train)} rows | Test: {len(test)} rows | WebMD: {len(webmd)} rows")

# drugs.com (train/test) → already consistent
train["source"] = "drugsComTrain"
test["source"]  = "drugsComTest"

# WebMD → rename to match
rename_map = {
    "Drug": "drugName",
    "Condition": "condition",
    "Reviews": "review",
    "Sides": "side_effects",
    "Date": "date",
    "Sex": "sex",
    "Age": "age",
    "Effectiveness": "effectiveness",
    "EaseofUse": "ease_of_use",
    "Satisfaction": "satisfaction",
    "UsefulCount": "usefulCount"
}
webmd = webmd.rename(columns=rename_map)
webmd["source"] = "WebMD"

# ---------------- CLEAN REVIEWS ----------------
tqdm.pandas()
train["clean_review"] = train["review"].progress_apply(clean_text)
test["clean_review"]  = test["review"].progress_apply(clean_text)
webmd["clean_review"] = webmd["review"].progress_apply(clean_text)

# ---------------- UNIFY ----------------
common_cols = list(set(train.columns) | set(test.columns) | set(webmd.columns))
train = train.reindex(columns=common_cols)
test  = test.reindex(columns=common_cols)
webmd = webmd.reindex(columns=common_cols)

df_all = pd.concat([train, test, webmd], ignore_index=True)
print(f" Combined dataset size: {len(df_all)} rows")

# ---------------- SAVE ----------------
df_all.to_csv(output_path, index=False, encoding="utf-8")
print(f"Preprocessed dataset saved -> {output_path}")