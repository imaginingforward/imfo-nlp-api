from flask import Flask, request, jsonify
import pandas as pd
import csv
# import spacy
# import re
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

# Build normalized keyword dict
keyword_dict = {}
for _, row in keyword_map.iterrows():
    phrase = row["User Query Phrase"].strip().lower()
    column = row["Maps To Column"].strip()
    value = row["Canonical Value"].strip().lower()
    keyword_dict.setdefault(phrase, []).append((column, value))

# For location formatting
keyword_display_map = {}
with open("keyword_map.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Maps To Column"] == "hq_state":
            keyword_display_map[row["User Query Phrase"].strip().lower()] = row["Canonical Value"].strip()

# Normalize and match query
def normalize_query(user_input):
    user_input_lower = user_input.lower()
    filters = {}

    # Match longer phrases first
    sorted_phrases = sorted(keyword_dict.keys(), key=lambda x: -len(x))

    for phrase in sorted_phrases:
        if phrase in user_input_lower:
            for column, value in keyword_dict[phrase]:
                filters.setdefault(column, set()).add(value)
            # Remove matched phrase to avoid overlap
            user_input_lower = user_input_lower.replace(phrase, " ")

    return filters

# Filter database based on matched filters
def filter_db(filters):
    results = db.copy()
    for column, values in filters.items():
        results = results[results[column].isin(values)]
    return results

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
    filters = normalize_query(query)
    print("Filters applied:", filters)  # <-- Debug print here
    results = filter_db(filters)
    print(f"Found {len(results)} matching records")  # <-- Debug print here

    # filters = normalize_query(query)
    # results = filter_db(filters)

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

    
