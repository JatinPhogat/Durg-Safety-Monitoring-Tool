# clean_entities.py
import pandas as pd
import re
from tqdm import tqdm

# ---------------- Load Dataset ----------------
df = pd.read_csv("outputs/drug_reviews_with_nlp.csv")

# ---------------- Cleaning Function ----------------
def clean_and_normalize(entities):
    if not isinstance(entities, (str, list)) or str(entities).strip() == "":
        return []

    # Convert string to list if needed
    if isinstance(entities, str):
        try:
            if entities.startswith("["):
                entities = eval(entities)  # e.g. "['rash','pain']" → ['rash','pain']
            else:
                entities = [entities]
        except:
            return []

    clean_list = []
    junk_words = {
        "cid", "zip", "html", "llt", "pt", "_", "c",
        "online", "file", "http", "fda", "slb", "dpd"
    }

    # Common replacements (spelling errors, abbreviations, merged words)
    replacements = {
        "bodytemperature": "body temperature",
        "diarr hoea": "diarrhoea",
        "thenia": "asthenia",
        "zziness": "dizziness",
        "emorrhage": "haemorrhage",
        "ticaria": "urticaria",
        "slr": "rash",
        "sslb": "rash",
        "chycardia": "tachycardia",
        "bp": "blood pressure",
        "htn": "hypertension",
        "sugar": "diabetes",
        "insomia": "insomnia",
        "vomitting": "vomiting",
        "nausia": "nausea",
        "anexity": "anxiety",
        "depresion": "depression",
        "weightgain": "weight gain",
        "weightloss": "weight loss",
        "hairfall": "hair loss",
        "hairfalling": "hair loss",
        "migren": "migraine",
        "inflamation": "inflammation",
        "itchyness": "itchiness",
        "painfull": "painful",
        "fatiuge": "fatigue",
        "tierdness": "tiredness",
        "days": "period days",
        "periods": "menstrual period",
        "mensis": "menstruation",
        "mensural": "menstrual",
        "mensturation": "menstruation",
        "pms": "premenstrual syndrome",
        "stomache": "stomach ache",
        "diarhea": "diarrhoea",
        "loosemotion": "diarrhoea",
        "loosemotions": "diarrhoea",
        "constpation": "constipation",
        "headeche": "headache",
        "headpain": "headache",
        "head ache": "headache",
        "sideeffect": "side effect",
        "sideeffects": "side effects",
        "hallucination": "hallucinations",
        "seziure": "seizure",
        "seziures": "seizures",
        "fit": "seizure",
        "fits": "seizures",
        "suicidal": "suicidal thoughts",
        "suicide": "suicidal thoughts",
        "chestpain": "chest pain",
        "heartattack": "heart attack",
        "palpitation": "palpitations",
        "vomittingblood": "vomiting blood",
        "bloodyvomit": "vomiting blood",
        "blackstool": "bloody stool",
        "darkstool": "bloody stool",
        "diarrheablood": "bloody diarrhoea",
        "rashh": "rash",
        "rashes": "rash",
        "bruises": "bruise",
        "acne": "acne breakout",
        "pimples": "acne breakout",
        "boils": "skin boils",
        "sweating": "excessive sweating",
        "sweats": "excessive sweating",
        "shivering": "tremors",
        "tremor": "tremors",
        "shakes": "tremors",
        "blurr vision": "blurred vision",
        "blurryvision": "blurred vision",
        "visionblur": "blurred vision",
        "drymouth": "dry mouth",
        "drymouththroat": "dry mouth and throat",
        "mouthulcer": "mouth ulcers",
        "stomachpain": "stomach pain",
        "stomachcramp": "stomach cramps",
        "cramping": "cramps",
        "gas": "bloating",
        "indigestion": "indigestion",
        "heartburn": "acid reflux"
    }

    for ent in entities:
        # Lowercase + clean special chars
        ent = str(ent).lower().strip()
        ent = re.sub(r"[^a-z\s\-]", "", ent)  # keep only letters, spaces, hyphen
        ent = re.sub(r"\s+", " ", ent).strip()

        # Skip junk & too-short tokens
        if len(ent) < 3 or ent in junk_words:
            continue

        # Replace normalized ADR terms
        ent = replacements.get(ent, ent)

        clean_list.append(ent)

    return list(set(clean_list))  # remove duplicates

# ---------------- Apply Cleaning with Progress ----------------
print("Cleaning entities with progress bar...")
tqdm.pandas()

df["matched_ADRs"] = df["entities"].progress_apply(clean_and_normalize)

# Update ADR_flag to reflect cleaned entities
df["ADR_flag"] = df["matched_ADRs"].apply(lambda x: 1 if len(x) > 0 else 0)

# Save cleaned dataset
out_path = "outputs/drug_reviews_with_matched_ADRs.csv"
df.to_csv(out_path, index=False)
print(f"\n Cleaned ADR entities saved -> {out_path}")

# ---------------- Quick Check ----------------
all_adrs = [adr for adrs in df["matched_ADRs"] for adr in adrs]

print(f"\n Dataset size: {len(df)}")
print(f"ADR rows: {df['ADR_flag'].sum()} ({df['ADR_flag'].mean() * 100:.2f}%)")

print("\nTop 20 cleaned ADRs:")
print(pd.Series(all_adrs).value_counts().head(20))
