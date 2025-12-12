# Drug Safety Monitoring Tool

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **AI-Powered Adverse Drug Reaction (ADR) Detection** - Automatically detect and analyze adverse drug reactions from multiple data sources including social media, web reviews, and medical datasets using state-of-the-art NLP and Named Entity Recognition.

## Key Features

- **Biomedical NER & Sentiment Analysis**: Extract ADRs using specialized NER models and analyze patient sentiment with transformer models
- **Multi-Source Data Aggregation**: Collect and combine reviews from Reddit, Drugs.com, WebMD, and Kaggle datasets
- **Automated Safety Profiling**: Generate comprehensive drug safety reports with PDF documentation and visualizations
- **Intelligent Risk Classification**: AI-powered recommendation system for drug safety assessment

## Use Cases

- **Pharmacovigilance**: Monitor post-market drug safety and adverse events
- **Clinical Research**: Analyze patient-reported outcomes and side effects
- **Healthcare Professionals**: Quick access to aggregated patient experiences
- **Pharmaceutical Companies**: Track drug safety signals across multiple platforms

## Technical Architecture

```
Data Sources (Reddit, Web, Kaggle)
            ↓
    Data Fetching Layer
            ↓
    Text Preprocessing
            ↓
    ┌─────────────────────────┐
    │   NLP Pipeline          │
    │  • Biomedical NER       │
    │  • Sentiment Analysis   │
    │  • Entity Normalization │
    └─────────────────────────┘
            ↓
    ┌─────────────────────────┐
    │   Analysis Engine       │
    │  • ADR Frequency        │
    │  • Sentiment Scoring    │
    │  • Risk Assessment      │
    └─────────────────────────┘
            ↓
    Visualization & Reporting
```

**Tech Stack:**
- **Frontend**: Streamlit (Interactive Web UI)
- **NLP Models**: 
  - HuggingFace Transformers (d4data/biomedical-ner-all)
  - Cardiff NLP Twitter RoBERTa (Sentiment Analysis)
- **Data Sources**: Reddit API (PRAW), Serper API (Web Search), Kaggle Datasets
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, ReportLab (PDF Generation)
- **NER & Classification**: Transformers Pipeline, Custom Entity Normalization

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/JatinPhogat/Drug-Safety-Monitoring-Tool.git
cd Drug-Safety-Monitoring-Tool
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install pandas numpy streamlit transformers torch
pip install requests tqdm matplotlib seaborn reportlab
```

## Project Structure

```
Biomedical Text Mining/
│
├── code/
│   ├── fetching feature/
│   │   ├── social_media_adr.py      # Streamlit web interface
│   │   ├── reddit_fetcher.py        # Reddit data collection
│   │   ├── web_scraper.py           # Web search integration
│   │   ├── utils_pipeline.py        # Core NLP pipeline
│   │   ├── cli_adr.py               # Command-line interface
│   │   └── test_reddit.py           # Testing utilities
│   │
│   └── ner and nlp/
│       ├── 01_clean_reviews.py      # Data preprocessing
│       ├── 02_ner_sentiment.py      # NER & sentiment extraction
│       ├── 03_NormalizeADR.py       # Entity normalization
│       ├── 04_analysis_per_drug.py  # Drug-level analysis
│       ├── 05_next_insights.py      # PDF report generation
│       └── 06_Per_drug_safety.py    # Safety profiling
│
├── outputs/                         # Generated analysis results
│   ├── insights/
│   │   ├── drug_reports/            # PDF reports
│   │   └── plots/                   # Visualizations
│   └── social/                      # Fetched data
│
└── README.md
```

## Usage

### Run Complete Analysis Pipeline
```bash
# Step 1: Clean and preprocess data
python "code/ner and nlp/01_clean_reviews.py"

# Step 2: Extract entities and sentiment
python "code/ner and nlp/02_ner_sentiment.py"

# Step 3: Normalize ADR entities
python "code/ner and nlp/03_NormalizeADR.py"

# Step 4: Analyze per drug
python "code/ner and nlp/04_analysis_per_drug.py"

# Step 5: Generate insights and reports
python "code/ner and nlp/05_next_insights.py"
```

## Risk Classification System

The tool uses an intelligent algorithm to classify drug safety:

- **Generally Safe**: < 80% ADR mentions, mostly positive sentiment
- **Monitor**: Moderate ADR mentions with concerning sentiment
- **Caution**: High ADR frequency (>80%) with mixed sentiment
- **Risky Drug**: High ADR mentions (>80%) + significant negative reviews
- **Very High ADR Risk**: Nearly all reviews mention ADRs + strong negatives

## Configuration

Edit `code/fetching feature/utils_pipeline.py` to customize:
- NER models for entity extraction
- Sentiment analysis models
- ADR safety thresholds

## Testing

Test the NLP pipeline with sample reviews:
```bash
python "code/fetching feature/test_reddit.py"
```

Expected output:
```
Running ADR + Sentiment Pipeline on sample reviews...

Summary
{'total_reviews': 5, 'adr_reviews': 4, '% ADR_mentions': 80.0, ...}
```

## Troubleshooting

### Memory Issues
If you encounter memory errors, reduce batch processing size in the analysis scripts.

### Path Issues
Update hardcoded file paths in the scripts to match your local directory structure.

## Disclaimer

**Important**: This tool is designed for research and informational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. The analysis is based on user-generated content and AI models, which may contain inaccuracies. Always consult healthcare professionals for medical decisions.

## Developer

**Jatin Phogat**
- GitHub: [@JatinPhogat](https://github.com/JatinPhogat)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/jatin-phogat)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **HuggingFace** for transformer models and NLP tools
- **d4data** for the biomedical NER model
- **Cardiff NLP** for the sentiment analysis model
- **Streamlit** for the web framework
- **Kaggle** for providing open drug review datasets

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues before creating new ones
