import logging
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

# Initialize logging
logging.basicConfig(level=logging.INFO)

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

    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    phrase_patterns = []

    for _, row in keyword_map.iterrows():
        phrase = row["User Query Phrase"].strip().lower()
        col = row["Maps To Column"].strip()
        val = row["Canonical Value"].strip().lower()
        keyword_aliases.setdefault(phrase, []).append((col, val))
        if col == "hq_state":
            display_map[val] = val
        phrase_patterns.append(nlp.make_doc(phrase))

    phrase_matcher.add("FILTER_TERMS", phrase_patterns)

except Exception as e:
    logging.error(f"Startup load error: {e}")


def extract_filters(query):
    filters = {}
    doc = nlp(query)
    
    # Match canonical phrases
    matches = phrase_matcher(doc)
    matched_phrases = set()
    for match_id, start, end in matches:
        phrase = doc[start:end].text.lower()
        matched_phrases.add(phrase)
        for col, val in keyword_aliases.get(phrase, []):
            if col == "hq_country" and ("hq_state" in filters or "hq_city" in filters):
                continue
            if col == "hq_state" and "hq_city" in filters:
                continue
            filters.setdefault(col, set()).add(val)

    # Location fallback using NER
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            loc = ent.text.strip().lower()
            if loc in keyword_aliases and loc not in matched_phrases:
                for col, val in keyword_aliases[loc]:
                    filters.setdefault(col, set()).add(val)

    # Financial expression extraction
    funding_match = re.search(r"(?:over|at least|more than)?\s*\$?([\d,.]+)\s*(m|million|b|billion)?", query.lower())
    if funding_match:
        amount, unit = funding_match.groups()
        amount = float(amount.replace(",", ""))
        multiplier = 1_000_000 if unit in ["m", "million"] else 1_000_000_000 if unit in ["b", "billion"] else 1
        filters["total_funding_raised"] = {">": int(amount * multiplier)}

    return filters


def search(query, filters):
    tokens = set(re.findall(r"\w+", query.lower()))
    mask = pd.Series([True] * len(db))

    for col, vals in filters.items():
        if col in db.columns and isinstance(vals, set):
            mask &= db[col].isin(vals)
        elif col == "total_funding_raised":
            op, val = list(vals.items())[0]
            if op == ">":
                mask &= pd.to_numeric(db.get("total_funding_raised", 0), errors="coerce") > val

    # Add broad token-based fallback
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
            return jsonify({"error": "Missing query"}), 400

        logging.info(f"Received query: {query}")
        filters = extract_filters(query)
        logging.info(f"Filters applied: {filters}")
        results = search(query, filters)
        logging.info(f"Found {len(results)} matches")

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
        logging.error(f"Request error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/")
def home():
    return "ImFo NLP API is live."


if __name__ == "__main__":
    app.run(debug=True)
