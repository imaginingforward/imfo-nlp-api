import logging
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import spacy

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = Flask(__name__)
CORS(app)

try:
    nlp = spacy.load("en_core_web_sm")

    # Load data
    db = pd.read_csv("test_db.csv").fillna("")
    for col in ["sector", "hq_state", "hq_country", "hq_city", "business_area", "business_activity"]:
        if col in db.columns:
            db[col] = db[col].str.lower()

    keyword_map = pd.read_csv("keyword_map.csv").fillna("")
    keyword_aliases = {}
    display_map = {}

    for _, row in keyword_map.iterrows():
        phrase = row["User Query Phrase"].strip().lower()
        col = row["Maps To Column"].strip()
        val = row["Canonical Value"].strip().lower()
        keyword_aliases.setdefault(phrase, []).append((col, val))
        if col == "hq_state":
            display_map[val] = val

except Exception as e:
    logger.error(f"Startup load error: {e}")

def extract_filters(query):
    text = query.lower()
    filters = {}

    sorted_phrases = sorted(keyword_aliases.keys(), key=lambda x: -len(x))
    for phrase in sorted_phrases:
        if re.search(rf"\\b{re.escape(phrase)}\\b", text):
            for col, val in keyword_aliases[phrase]:
                if col == "hq_country" and ("hq_state" in filters or "hq_city" in filters):
                    continue
                if col == "hq_state" and "hq_city" in filters:
                    continue
                filters.setdefault(col, set()).add(val)
            text = re.sub(rf"\\b{re.escape(phrase)}\\b", " ", text)

    # Use NER to boost location extraction
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            loc = ent.text.strip().lower()
            if loc in keyword_aliases:
                for col, val in keyword_aliases[loc]:
                    filters.setdefault(col, set()).add(val)

    return filters

def search(query, filters):
    tokens = set(re.findall(r"\w+", query.lower()))
    mask = pd.Series([True] * len(db))
    for col, vals in filters.items():
        if col in db.columns:
            mask &= db[col].isin(vals)

    # Add broad token-based match as fallback
    cols = ["business_activity", "description", "hq_city", "hq_state", "hq_country", "business_area", "sector"]
    fallback = pd.Series([False] * len(db))
    for col in cols:
        if col in db.columns:
            pattern = '|'.join(re.escape(t) for t in tokens)
            fallback |= db[col].str.contains(pattern, case=False, na=False)

    return db[mask | fallback].copy()

def format_location(row):
    city = row.get("hq_city", "").title()
    state = row.get("hq_state", "").lower()
    country = row.get("hq_country", "").title()

    if country.lower() == "usa":
        state_full = display_map.get(state, state).title()
        return f"{city}, {state_full}" if state_full else city
    return f"{city}, {country}" if country else city

@app.route("/parse", methods=["POST"])
def parse():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        if not query:
            logger.warning("Missing query string")
            return jsonify({"error": "Missing query"}), 400

        logger.info(f"Received query: {query}")
        filters = extract_filters(query)
        logger.info(f"Filters applied: {filters}")
        results = search(query, filters)
        logger.info(f"Found {len(results)} matches")

        response = [{
            "business_area": row.get("business_area", ""),
            "company_name": row.get("company_name", ""),
            "description": row.get("description", ""),
            "business_activity": row.get("business_activity", ""),
            "hq_location": format_location(row),
            "website_url": row.get("website_url", "")
        } for _, row in results.iterrows()]

        return jsonify(response)

    except Exception as e:
        logger.error(f"Request error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/")
def home():
    return "ImFo NLP API is live."

if __name__ == "__main__":
    app.run(debug=True)
