import praw
import pandas as pd
from tqdm import tqdm
import os

# ---------------- Config ----------------
OUT_DIR = "outputs/social"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Reddit Auth ----------------
# Load credentials from praw.ini
reddit = praw.Reddit("ADRMiningApp")

print("Authenticated (read-only):", reddit.read_only)

# ---------------- Fetch Function ----------------
def fetch_reddit_posts(drug_name, limit=50):
    subreddits = "drugs+AskDocs+medicine+pharmacy+Health"
    posts = []

    for submission in tqdm(
        reddit.subreddit(subreddits).hot(limit=500),
        desc=f"Scanning Reddit for {drug_name}"
    ):
        text = (submission.title + " " + submission.selftext).lower()
        if drug_name.lower() in text:
            posts.append({
                "drugName": drug_name,
                "title": submission.title,
                "text": submission.selftext,
                "score": submission.score,
                "url": submission.url,
                "subreddit": submission.subreddit.display_name
            })
        if len(posts) >= limit:
            break

    return pd.DataFrame(posts)

# ---------------- Main ----------------
if __name__ == "__main__":
    drug = "Cialis"
    df = fetch_reddit_posts(drug, limit=20)

    if not df.empty:
        out_path = os.path.join(OUT_DIR, f"reddit_{drug}.csv")
        df.to_csv(out_path, index=False)
        print(f"\nCollected {len(df)} posts -> {out_path}")
    else:
        print(f"No posts found for {drug}")
