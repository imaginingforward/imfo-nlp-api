from flask import Flask, request, jsonify
import pandas as pd
import csv
# import spacy
import re
from flask_cors import CORS

# Initialize app
app = Flask(__name__)
CORS(app)

# Load data
db = pd.read_csv("test_db.csv").fillna("")
for col in ["sector", "hq_state", "hq_country", "hq_city", "business_area", "business_activity"]:
    if col in db.columns:
        db[col] = db[col].str.lower()

# Load keyword map        
keyword_map = pd.read_csv("keyword_map.csv").fillna("")

# Build alias dictionary
keyword_aliases = {}
for _, row in keyword_map.iterrows():
    phrase = row["User Query Phrase"].strip().lower()
    column = row["Maps To Column"].strip()
    value = row["Canonical Value"].strip().lower()
    keyword_aliases.setdefault(phrase, []).append((column, value))

# For location formatting
keyword_display_map = {}
with open("keyword_map.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Maps To Column"] == "hq_state":
            keyword_display_map[row["User Query Phrase"].strip().lower()] = row["Canonical Value"].strip()

def extract_alias_filters(user_input):
    """
    Find all alias phrases in user_input and build a filter dict {column: set(values)}.
    Prioritize country over state if both match the same phrase (e.g., 'canada').
    """
    user_input_lower = user_input.lower()
    filters = {}

    # Track which phrases we've already matched
    sorted_phrases = sorted(keyword_aliases.keys(), key=lambda x: -len(x))
    for phrase in sorted_phrases:
        
        # Match exact phrase with word boundaries
        if re.search(r'\b' + re.escape(phrase) + r'\b', user_input_lower):

            # Extract all column-value pairs mapped to this phrase
            mappings = keyword_aliases[phrase]

            # Check for conflict: if both country and state present
            has_country = any(col == "hq_country" for col, _ in mappings)
            has_state = any(col == "hq_state" for col, _ in mappings)

            # Drop state if country also exists
            for col, val in mappings:
                if has_country and col == "hq_state":
                    continue  # skip incorrect state mapping
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
    tokens = set(re.findall(r'\w+', user_query_lower))  # tokenize words

    # Build mask from alias filters
    mask_alias = pd.Series([True] * len(db))  # start with True, then AND filters
    for col, vals in alias_filters.items():
        if col in db.columns:
            mask_alias &= db[col].isin(vals)
    
    # Build mask from text search in key columns
    columns_to_search = ['business_activity', 'description', 'hq_city', 'hq_state', 'hq_country', 'business_area', 'sector']
    mask_text = pd.Series([False] * len(db))
    for col in columns_to_search:
        if col in db.columns:
            # Combine tokens into regex pattern to match any token
            pattern = '|'.join([re.escape(token) for token in tokens])
            mask_text |= db[col].str.contains(pattern, case=False, na=False)
    
    # Combine masks: rows that match alias filters OR contain query tokens in text
    combined_mask = mask_alias | mask_text

    return db[combined_mask].copy()

# Format readable location string
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
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing query"}),400

    print("Received query:", query)   # <-- Debug print here
    
    alias_filters = extract_alias_filters(query)
    print("Alias filters found:", alias_filters)  # <-- Debug print here
    
    results = search_db(query, alias_filters, db)
    print(f"Found {len(results)} matching records")  # <-- Debug print here

    # Frontend Card Format
    response =[]
    for _, row in results.iterrows():
        response.append({
            "business_area": row.get("business_area",""),
            "company_name": row.get("company_name",""),
            "description": row.get("description",""),
            "business_activity": row.get("business_activity",""),
            "hq_location": format_location(row),
            "website_url":row.get("website_url","")
        })
    return jsonify(response)

# Health check route
@app.route("/", methods=["GET"])
def home():
    return "ImFo NLP API is live."

# NER based features
# nlp = spacy.load("en_core_web_sm")
# doc = nlp(query.lower())

    
