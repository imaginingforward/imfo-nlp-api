import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import spacy
import re

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize app
app = Flask(__name__)
CORS(app)

try:
    # Load NLP model
    nlp = spacy.load("en_core_web_sm")

    # Load database
    db = pd.read_csv("test_db.csv").fillna("")
    for col in ["sector", "hq_state", "hq_country", "hq_city", "business_area", "business_activity"]:
        if col in db.columns:
            db[col] = db[col].str.lower()

    # Load keyword map
    keyword_map = pd.read_csv("keyword_map.csv").fillna("")
    keyword_aliases = {}
    keyword_display_map = {}

    # Build alias dictionary + keyword display map
    for _, row in keyword_map.iterrows():
        phrase = row["User Query Phrase"].strip().lower()
        column = row["Maps To Column"].strip()
        value = row["Canonical Value"].strip().lower()
        keyword_aliases.setdefault(phrase, []).append((column, value))
    
        # Only needed for location formatting
        if column == "hq_state":
            keyword_display_map[value] = value
except Exception as e:
    logging.error(f"Error loading data: {e}")
    app.logger.error(f"Error loading data: {e}")

def extract_alias_filters(user_input):
    """
    Build a filter dict {column: set(values)} by matching both:
    - Exact multi-word phrases
    - Individual words
    Prioritize state over country if both match the same value (e.g., "california vs canada").
    """
    user_input_lower = user_input.lower()
    filters = {}

    # Tokenize user input
    tokens = user_input_lower.split()

    # Check for state
    state_token = [
        token for token in tokens 
        if token in [row["Canonical Value"].strip().lower()
                     for _, row in keyword_map.iterrows()
                     if row["Maps To Column"].strip() == "hq_state"]
    ]
    if state_token:
        filters.setdefault("hq_state", set()).add(state_token[0])
        
    # Track already matched phrases
    sorted_phrases = sorted(keyword_aliases.keys(), key=lambda x: -len(x))
    for phrase in sorted_phrases:

    # Match exact phrases with word boundaries first
        if re.search(r'\b' + re.escape(phrase) + r'\b', user_input_lower):

            # Extract all column-value pairs mapped to this phrase
            mappings = keyword_aliases[phrase]

            # Check for conflict when country and state present
            has_state = any(col == "hq_state" for col, _ in mappings)
            has_country = any(col == "hq_country" for col, _ in mappings)
            
            # Drop country if state also exists
            for col, val in mappings:
                if has_state and col == "hq_country":
                    continue
                filters.setdefault(col, set()).add(val)
                
            user_input_lower = re.sub(r'\b' + re.escape(phrase) + r'\b', ' ', user_input_lower)
                    
return filters

def search_db(user_query, alias_filters, db):
    """
    Return rows where:
    - Any alias_filters match exactly in their columns OR
    - Any token from user_query appears as substring in key columns
    """
    user_query_lower = user_query.lower()
    tokens = set(re.findall(r'\w+', user_query_lower))

    mask_alias = pd.Series([True] * len(db))
    for col, vals in alias_filters.items():
        if col in db.columns:
            mask_alias &= db[col].isin(vals)

    columns_to_search = ['business_activity', 'description', 'hq_city', 'hq_state', 'hq_country', 'business_area', 'sector']
    mask_text = pd.Series([False] * len(db))
    for col in columns_to_search:
        if col in db.columns:
            pattern = '|'.join([re.escape(token) for token in tokens])
            mask_text |= db[col].str.contains(pattern, case=False, na=False)

    combined_mask = mask_alias | mask_text
    return db[combined_mask].copy()

def format_location(row):
    city = row.get("hq_city", "").title()
    state = row.get("hq_state", "").lower()
    country = row.get("hq_country", "").title()

    if country.lower() == "usa":
        state_full = keyword_display_map.get(state, state).title()
        return f"{city}, {state_full}" if state_full else city
    else:
        return f"{city}, {country}" if country else city

@app.route("/parse", methods=["POST"])
def parse_query():
    try:
        data = request.get_json()
        query = data.get("query", "")
    
        if not query:
            return jsonify({"error": "Missing query"}),400
       
        # validate query
        if not isinstance(query, str):
            return jsonify({"error": "Invalid query"}),400

        query = query.strip()
        if not query:
            return jsonify({"error":"Invalid query"}),400

        logging.info(f"Received query: {query}")
        alias_filters = extract_alias_filters(query)
        logging.info(f"Alias filters found: {alias_filters}")
        results = search_db(query, alias_filters, db)
        logging.info(f"Found {len(results)} matching records")

        response = []
        for _, row in results.iterrows():
            response.append({
                "business_area": row.get("business_area", ""),
                "company_name": row.get("company_name", ""),
                "description": row.get("description", ""),
                "business_activity": row.get("business_activity", ""),
                "hq_location": format_location(row),
                "website_url": row.get("website_url", "")
            })
        return jsonify(response)
except Exception as e:
    logging.error(f"Error handling request: {e}")
    app.logger.error(f"Error handling request: {e}")
    return jsonify({"error": "Internal Server Error"}),500

@app.route("/", methods=["GET"])
def home():
    return "ImFo NLP API is live."

if __name__ == "__main__":
    app.run(debug=True)
