import pandas as pd
from utils_pipeline import analyze_reviews, summarize_results

# Some test reviews
test_reviews = [
    "I had a severe headache and dizziness after taking the drug.",
    "This medicine cured my infection but gave me nausea.",
    "No side effects at all, I feel great!",
    "Experienced swelling and rash on my arms.",
    "I was happy, no depression or anxiety anymore."
]

print("Running ADR + Sentiment Pipeline on sample reviews...")
df = analyze_reviews(test_reviews)
print(df)

print("\nSummary")
summary = summarize_results(df)
print(summary)
