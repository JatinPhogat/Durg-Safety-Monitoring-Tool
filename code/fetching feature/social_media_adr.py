import pandas as pd
import streamlit as st
from utils_pipeline import analyze_reviews, summarize_results, get_recommendation
from reddit_fetcher import fetch_reddit_posts as fetch_reddit
from web_scraper import web_search_reviews
import base64

# ---------------- Set Background ----------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                        url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }}
        .stDataFrame {{ background-color: rgba(255,255,255,0.85); color: black; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- Load Kaggle dataset ----------------
@st.cache_data
def load_kaggle_data():
    try:
        return pd.read_csv("outputs/drug_reviews_with_matched_ADRs.csv")
    except FileNotFoundError:
        st.error("Kaggle dataset not found. Please check the file path.")
        return pd.DataFrame(columns=["drugName", "clean_review", "condition"])

df_kaggle = load_kaggle_data()

def get_all_reviews(drug_name, max_reviews=200):
    reviews = []

    # Kaggle
    if not df_kaggle.empty:
        kaggle_subset = df_kaggle[df_kaggle["drugName"].str.lower() == drug_name.lower()]
        kaggle_subset = kaggle_subset.head(max_reviews)[["drugName", "clean_review", "condition"]]
        kaggle_subset = kaggle_subset.rename(columns={"clean_review": "text"})
        kaggle_subset["source"] = "kaggle"
        reviews.append(kaggle_subset)

    # Reddit
    try:
        reddit_df = fetch_reddit(drug_name, limit=max_reviews)
        if not reddit_df.empty:
            reddit_df = reddit_df[["drugName", "text"]]
            reddit_df["source"] = "reddit"
            reddit_df["condition"] = None
            reviews.append(reddit_df)
    except Exception as e:
        st.warning(f" Reddit fetch failed: {e}")

    # Web search
    try:
        web_df = web_search_reviews(drug_name, max_results=max_reviews)
        if not web_df.empty:
            web_df = web_df.rename(columns={"snippet": "text"})
            web_df = web_df[["drugName", "text"]]
            web_df["source"] = "web"
            web_df["condition"] = None
            reviews.append(web_df)
    except Exception as e:
        st.warning(f" Web search failed: {e}")

    if reviews:
        return pd.concat(reviews, ignore_index=True)
    return pd.DataFrame(columns=["drugName", "text", "source", "condition"])

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Drug Safety Monitor", page_icon="💊", layout="wide")
set_background("/Users/kunikabhadra/Music/ADR-MINING/data/Background.png")

st.title("Drug Safety Monitor")
st.caption("Aggregating reviews from Kaggle, Reddit, and the Web to detect ADRs (Adverse Drug Reactions).")

drug_name = st.text_input("Enter a drug name:")
max_reviews = st.slider("Number of reviews to analyze per source", 50, 500, 200)

if drug_name:
    st.info(" Fetching reviews from Kaggle, Reddit, and Web...")
    reviews = get_all_reviews(drug_name, max_reviews=max_reviews)

    if reviews.empty:
        st.error(f"No reviews found for **{drug_name}**")
    else:
        st.success(f"Collected {len(reviews)} reviews from {reviews['source'].nunique()} sources")

        # ---------------- Condition Filter ----------------
        if "condition" in reviews.columns and reviews["condition"].notna().any():
            condition_list = sorted(reviews["condition"].dropna().unique())
            selected_condition = st.selectbox("Filter by Condition", ["All"] + condition_list)
            if selected_condition != "All":
                reviews = reviews[reviews["condition"] == selected_condition]

        # ---------------- Run Analysis ----------------
        results = analyze_reviews(reviews["text"].fillna("").tolist())
        summary = summarize_results(results)

        # ---------------- Summary ----------------
        st.header("Overall Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", summary["total_reviews"])
        col2.metric("% ADR Mentions", f"{summary['% ADR_mentions']}%")
        col3.metric("Positive Sentiment", f"{summary['positive %']}%")
        col4.metric("Negative Sentiment", f"{summary['negative %']}%")

        # ---------------- Top ADRs ----------------
        if summary["top_ADRs"]:
            st.subheader(" Most Reported ADRs")
            st.bar_chart(pd.Series(summary["top_ADRs"]))

        # ---------------- Per-source breakdown ----------------
        st.subheader(" Reviews by Source")
        st.dataframe(reviews.groupby("source").size().reset_index(name="count"))

        # ---------------- Per-condition breakdown ----------------
        if "condition" in reviews.columns and reviews["condition"].notna().any():
            st.subheader(" ADR Mentions by Condition")
            condition_summary = results.assign(condition=reviews["condition"].values) \
                                       .groupby("condition")["ADR_flag"].sum() \
                                       .reset_index(name="ADR Mentions")
            st.bar_chart(condition_summary.set_index("condition"))

        # ---------------- Recommendation ----------------
        st.subheader(" Recommendation")
        st.success(get_recommendation(summary))
