import logging
import re
import pandas as pd
import spacy
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from rapidfuzz import process, fuzz
from spacy.matcher import PhraseMatcher
from collections.abc import Iterable
from elasticsearch import Elasticsearch
from uuid import uuid4
# ----------------------------- Initialization -----------------------------

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
CORS(app)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load ElasticSearch
es = Elasticsearch(os.environ.get("ELASTICSEARCH_URL"))
if not es.ping():
    raise ValueError("Could not connect to Elasticsearch cluster")

# Stopwords and prepositions to exclude from alias matching
EXCLUDED_TERMS = {"in", "from", "near", "at", "around", "on", "by"}

# Define Acronyms
matcher = PhraseMatcher(nlp.vocab, attr="ORTH")  # Case-sensitive, token-based
acronyms = ["GNSS", "ISR", "PNT", "SAR", "SBIR", "SSA", "AFWERX", "DIU", "DARPA", "DOD", "YC", "SFA"]

# Convert to Doc patterns
patterns = [nlp.make_doc(acr) for acr in acronyms]
matcher.add("ACRONYM", patterns)

# Load Gazeteer
gazetteer_df = pd.read_csv("gazetteer.csv").fillna("")
gazetteer = dict(zip(gazetteer_df["name"].str.lower(), gazetteer_df["type"].str.lower()))

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
        if phrase not in EXCLUDED_TERMS: # prevent prepositions from being matched
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
    max_value =0
    for _, _, amt, mag in pattern:
        num = float(amt.replace(",", ""))
        if mag.lower() == "b":
            num *=1_000_000_000
        elif mag.lower() == "m":
            num *=1_000_000
        max_value = max(max_value, num)
    return {">": int(max_value)} if max_value else None

def fuzzy_match_phrases(text, score_cutoff=80):
    matched = []
    for phrase, score, _ in process.extract(text, all_alias_phrases, scorer=fuzz.partial_ratio, score_cutoff=score_cutoff):
        if phrase not in EXCLUDED_TERMS:
            if f" {phrase} " in f" {text.lower()} ":
                matched.append(phrase)
    return matched

def safe_set_filter(filters, col, val, origin, origin_map):
    # Set if not present
    if col not in filters:
        filters[col] = {val}
        origin_map[col] = origin
        return
        
    # If already set, respect NER/gazetteer priority over fuzzy
    current_origin = origin_map.get(col)
    precedence = {"gazetteer": 3, "NER": 2, "acronym":1, "fuzzy": 0}

    if precedence.get(origin, 0) > precedence.get(current_origin, 0):
        filters[col] = {val}
        origin_map[col] = origin
        return
    return
    
def extract_filters(query: str):
    query_clean = clean_query(query)
    filters = {}
    filter_origin = {}
    matched_tokens = set()
    free_text_terms = []

    # Debug if keywords is being extracted
    # logger.info(f"All alias phrases: {all_alias_phrases}")
    # logger.info(f"All keyword aliases: {keyword_aliases}")
    # logger.info(f"User query (cleaned): '{query_clean}'")
    
    # 1 - Fuzzy phrase match
    matched_aliases = fuzzy_match_phrases(query_clean)
    logger.info(f"Matched fuzzy aliases: {matched_aliases}")
    for phrase in matched_aliases:
        # mark as matched, for multi-word expressions
        matched_tokens.update(phrase.split())
        for col, val in keyword_aliases[phrase]:
            safe_set_filter(filters, col, val, "fuzzy", filter_origin)

    # 2 - Named Entity Recognition NER
    doc = nlp(query_clean)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            loc = ent.text.strip().lower()
            if loc in EXCLUDED_TERMS:
                continue
            matched_tokens.update(loc.split())
            if loc in gazetteer:
                loc_type = gazetteer[loc]
                safe_set_filter(filters, f"hq_{loc_type}", loc, "NER", filter_origin)
                
    # 3 - Acronym match, EXACT for cast-sensitive
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        token = span.text.strip().upper()
        matched_tokens.add(token.lower())
        if token in keyword_aliases:
            for col, val in keyword_aliases[token]:
                safe_set_filter(filters, col, val, "acronym", filter_origin)

    # 4 - Funding filters
    funding = extract_funding_filter(query_clean)
    if funding:
        filters['total_funding_raised'] = funding
            
    # 5 - Check in gazetteer
    for phrase in gazetteer.keys():
        phrase_tokens = set(phrase.lower().split()) 
        # Skip if matched in NER
        if phrase in matched_tokens:
            continue
        if phrase in query_clean.lower():
            loc_type = gazetteer[phrase]
            safe_set_filter(filters,f"hq_{loc_type}", phrase, "gazetteer",filter_origin)
            matched_tokens.update(phrase.lower().split())

    # 6 - All unmatched tokens/phrases as free-text
    all_tokens = set(query_clean.split())
    for token in all_tokens:
        if token not in matched_tokens and token not in EXCLUDED_TERMS:
            free_text_terms.append(token)

    #7 - Collapse filter sets to scalars (first value or keep sets if multi-valued)
    simple_filters = {k: (list(v)[0] if isinstance(v, set) and len(v) == 1 else v) for k, v in filters.items()}
        
    logger.info(f"Extracted filters: {filters}")
    logger.info(f"Free-text:{free_text_terms}")
    return {
        "filters": simple_filters,
        "free_text_terms": free_text_terms
    }
"""
def search(query_clean, filters, free_text_terms):
    try:
        df = db.copy()
        logger.info(f"Applying filters: {filters}")
        
        # 1 - Clean funding values
        def clean_funding(x):
            try:
                return float(re.sub(r"[^\d.]", "", str(x)))
            except:
                return 0.0
        
        # 2 - Apply structured filters
        mask = pd.Series([True] * len(df))
        for col, vals in filters.items():
            if col == "total_funding_raised":
                for op, threshold in vals.items():
                    if op == ">":
                        mask &= df[col].apply(lambda x: clean_funding(x) > threshold)
                    elif op == "<":
                        mask &= df[col].apply(lambda x: clean_funding(x) < threshold)
            elif col in df.columns:
                if not isinstance(vals, Iterable) or isinstance(vals, str):
                    vals = [vals]
                mask &= df[col].isin(vals)
        df_filtered = df[mask]
        logger.info(f"Number of results before fallback: {len(df_filtered)}")

        # 3 - Narrow down with free-text match
        if len(df_filtered) > 0 and free_text_terms:
            def match_row_filtered(row):
                text = " ".join([
                str(row.get("company_name","")),
                str(row.get("description","")),
                str(row.get("business_area","")),
                str(row.get("business_activity","")),
                str(row.get("hq_city","")),
                str(row.get("hq_state","")),
                str(row.get("hq_country",""))
            ]).lower()
            return any(term.lower() in text for term in free_text_terms)

            mask_ft = df_filtered.apply(match_row_filtered, axis=1)
            df_filtered = df_filtered[mask_ft]
        
        # 4 - Return results if they exist
        if len(df_filtered) > 0:
            return df_filtered.copy()
    
        # 4 - Fallback to original structured results
        if df_filtered.empty and len(df[mask]) > 0:
            return df[mask].copy()
        
        # 5 - If no results, fallback using full-text                                      
        terms = free_text_terms or re.findall(r"\w+", query_clean.lower())
        
        def match_row_fallback(row):
            text = " ".join([
                str(row.get("company_name","")),
                str(row.get("description","")),
                str(row.get("business_area","")),
                str(row.get("business_activity","")),
                str(row.get("hq_city","")),
                str(row.get("hq_state","")),
                str(row.get("hq_country",""))
            ]).lower()
            return any(term.lower() in text for term in terms)
        
        mask_fallback = df.apply(match_row_fallback, axis=1)
        df_ft = df[mask_fallback]
        logger.info(f"Number of results after fallback: {len(df_ft)}")
        
        return df_ft.copy()
   
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return pd.DataFrame()
"""
def format_location(row):
    try:
        city = row.get("hq_city", "").strip().title()
        state = row.get("hq_state", "").strip().upper()
        country = row.get("hq_country", "").strip().title()

        if city and state:
            return f"{city}, {state}"
        elif city and country:
            return f"{city}, {country}"
        return city
    except Exception as e:
        logger.error(f"Error formatting location: {str(e)}", exc_info=True)
        return ""

# ----------------------------- Flask Routes -----------------------------

@app.route("/parse", methods=["POST"])
def parse():
    data = request.get_json()
    query_clean = data.get("query", "").strip()
        
    if not query_clean:
        logger.warning("Missing query string")
        return jsonify({"error": "Missing query"}), 400

    logger.info(f"Received query: {query_clean}")
    es_query = {
        "query": {
            "multi_match": {
                "query": query_clean,
                "fields": [
                    "company_name^3",
                    "description^2",
                    "business_activity",
                    "business_area",
                    "hq_location",
                    "leadership",
                    "capital_partners",
                    "capital_partners"
                ],
                "fuzziness": "AUTO"
            }
        }
    }
                    
    try:
        res = es.search(index="space-companies", body=es_query)
        hits = res["hits"]["hits"]

        companies = []
        for hit in hits:
            source = hit["_source"]
            companies.append({
                "company_name": source.get("company_name", ""),
                "business_activity": source.get("business_activity", ""),
                "business_area": source.get("business_area", ""),
                "description": source.get("description", ""),
                "hq_location": format_location(source),
                "leadership": source.get("leadership", ""),
                "capital_partners": source.get("capital_partners", ""),
                "notable_partners": source.get("notable_partners", ""),
                "website_url": source.get("website_url", ""),
                "linkedin_url": source.get("linkedin_url",""),
                "crunchbase_url": source.get("crunchbase_url",""),
                "twitter_url": source.get("twitter_url","")  
        })

        return jsonify({"results": companies})

    except Exception as e:
        logger.error(f"Request error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/upload-to-es", methods=["POST"])
def upload_to_elasticsearch():
    for _, row in db.iterrows():
        doc = {
            "company_name": row.get("company_name", ""),
            "business_activity": row.get("business_activity", ""),
            "business_area": row.get("business_area", ""),
            "description": row.get("description", ""),
            "hq_location": format_location(row),
            "leadership": row.get("leadership", ""),
            "capital_partners": row.get("capital_partners", ""),
            "notable_partners": row.get("notable_partners", ""),
            "website_url": row.get("website_url", ""),
            "linkedin_url": row.get("linkedin_url",""),
            "crunchbase_url": row.get("crunchbase_url",""),
            "twitter_url": row.get("twitter_url","")
        }

        es.index(index="space-companies", id=str(uuid4()), document=doc)
    return jsonify({"status": "success", "message": "Uploaded to Elasticsearch"})
    
@app.route("/")
def home():
    return "ImFo NLP API is live."

if __name__ == "__main__":
    app.run()
