import logging
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import spacy
from rapidfuzz import process, fuzz

# ----------------------------- Initialization -----------------------------

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
CORS(app)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Stopwords and prepositions to exclude from alias matching
EXCLUDED_TERMS = {"in", "from", "near", "at", "around", "on", "by"}

# Load data
try:
    db = pd.read_csv("test_db.csv").fillna("")
    for col in ["sector", "hq_state", "hq_country", "hq_city", "business_area", "business_activity", "description"]:
        if col in db.columns:
            db[col] = db[col].str.lower()

    keyword_map = pd.read_csv("keyword_map.csv").fillna("")
    keyword_aliases = {}
    display_map = {}

    for _, row in keyword_map.iterrows():
        phrase = row["User Query Phrase"].strip().lower()
        col = row["Maps To Column"].strip()
        val = row["Canonical Value"].strip().lower()
        if phrase not in EXCLUDED_TERMS:  # prevent prepositions from being matched
            keyword_aliases.setdefault(phrase, []).append((col, val))
        if col == "hq_state":
            display_map[val] = val

    # For fuzzy search support
    all_alias_phrases = list(keyword_aliases.keys())

except Exception as e:
    logger.error(f"Startup error: {e}")

# ----------------------------- Utility Functions -----------------------------

def clean_query(text):
    return re.sub(r"[^\w\s$.,+-]", " ", text.lower().strip())

def extract_funding_filter(text):
    # Examples: "$10M+", "over $10 million", "raised above $50M"
    pattern = re.findall(r"(over|above|raised\s*(at\s*least)?|more\s*than)?\s*\$?\s*([\d.]+)\s*([mb]?)", text, flags=re.I)
    if not pattern:
        return None
    max_value = 0
    for _, _, amt, mag in pattern:
        num = float(amt.replace(",", ""))
        if mag.lower() == "b":
            num *= 1_000_000_000
        elif mag.lower() == "m":
            num *= 1_000_000
        max_value = max(max_value, num)
    return {">": int(max_value)} if max_value else None

def fuzzy_match_phrases(text, score_cutoff=90):
    matched = []
    for phrase, score, _ in process.extract(text, all_alias_phrases, scorer=fuzz.partial_ratio, score_cutoff=score_cutoff):
        if phrase not in EXCLUDED_TERMS:
            matched.append(phrase)
    return matched

def extract_filters(query):
    query = clean_query(query)
    filters = {}

    # Fuzzy phrase match
    matched_aliases = fuzzy_match_phrases(query)
    for phrase in matched_aliases:
        for col, val in keyword_aliases[phrase]:
            # Location hierarchy checks
            if col == "hq_country" and ("hq_state" in filters or "hq_city" in filters):
                continue
            if col == "hq_state" and "hq_city" in filters:
                continue
            filters.setdefault(col, set()).add(val)

    # Named Entity Recognition
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            loc = ent.text.strip().lower()
            if loc in EXCLUDED_TERMS:
                continue
            if loc in keyword_aliases:
                for col, val in keyword_aliases[loc]:
                    # Apply the same hierarchy rule
                    if col == "hq_country" and ("hq_state" in filters or "hq_city" in filters):
                        continue
                    if col == "hq_state" and "hq_city" in filters:
                        continue
                    filters.setdefault(col, set()).add(val)

    # Funding filters
    funding = extract_funding_filter(query)
    if funding:
