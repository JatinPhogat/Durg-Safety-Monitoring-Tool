import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- Config ----------------
INPUT_PROFILE = "outputs/drug_profile_summary.csv"
INPUT_MONTHLY = "outputs/drug_monthly_trends.csv"
INPUT_TOP_ADRS = "outputs/drug_top_adrs.csv"
OUT_DIR = "outputs/insights"
REPORT_DIR = os.path.join(OUT_DIR, "drug_reports")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------- Helpers ----------------
def safe_filename(name: str) -> str:
    """Convert drug name into a safe filename"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))

# ---------------- Load ----------------
df_profile = pd.read_csv(INPUT_PROFILE)
df_monthly = pd.read_csv(INPUT_MONTHLY)
df_top_adrs = pd.read_csv(INPUT_TOP_ADRS)

# ---------------- Per-Drug Reports ----------------
print("Generating per-drug PDF reports for ALL drugs...")

styles = getSampleStyleSheet()

for drug in tqdm(df_profile["drugName"].dropna().unique()):
    safe_drug = safe_filename(drug)

    # PDF report path
    doc = SimpleDocTemplate(
        os.path.join(REPORT_DIR, f"{safe_drug}_report.pdf"), pagesize=A4
    )
    story = []
    story.append(Paragraph(f"<b>Drug Safety Report: {drug}</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    # Safety summary
    row = df_profile[df_profile["drugName"] == drug].iloc[0].to_dict()
    summary_data = [
        ["Total Reviews", row.get("total_reviews", 0)],
        ["ADR Reviews", row.get("adr_reviews", 0)],
        ["% ADR Mentions", f"{row.get('adr_percent', 0):.2f}%"],
        ["Positive %", f"{row.get('pos_ratio', 0)*100:.1f}%"],
        ["Negative %", f"{row.get('neg_ratio', 0)*100:.1f}%"],
        ["Mild ADRs", row.get("mild_count", 0)],
        ["Serious ADRs", row.get("serious_count", 0)],
    ]
    story.append(Table(summary_data))
    story.append(Spacer(1, 12))

    # Top ADRs
    top_adrs = df_top_adrs[df_top_adrs["drugName"] == drug].head(10)
    story.append(Paragraph("<b>Top ADRs</b>", styles["Heading2"]))
    if not top_adrs.empty:
        adr_table = [[r["adr"], r["count"]] for _, r in top_adrs.iterrows()]
        story.append(Table([["ADR", "Mentions"]] + adr_table))
    else:
        story.append(Paragraph("No ADR data available.", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Monthly trend plot
    trend = df_monthly[df_monthly["drugName"] == drug].sort_values("month")
    if not trend.empty:
        plt.figure(figsize=(6, 3))
        plt.plot(trend["month"], trend["adr_count"], marker="o")
        plt.xticks(rotation=45, fontsize=6)
        plt.title(f"Monthly ADR Mentions for {drug}")
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f"{safe_drug}_trend.png")
        plt.savefig(plot_path)
        plt.close()
        story.append(Image(plot_path, width=400, height=200))
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No monthly ADR trend data available.", styles["Normal"]))

    # Build PDF
    doc.build(story)

print("PDF reports saved for ALL drugs -> outputs/insights/drug_reports/")
