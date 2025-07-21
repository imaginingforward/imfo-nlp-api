import logging
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import spacy
from rapidfuzz import process, fuzz
from spacy.matcher import PhraseMatcher
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
            matched.append(phrase)
    return matched

def safe_set_filter(filters, col, val, origin, origin_map):
    # Set if not present
    if col not in filters:
        filters[col] = {val}
        origin_map[col] = origin
        return
    # If already set, respect NER/gazetteer priority over fuzzy
    if col in filters:
        if origin_map[col] == "NER" and origin in ["fuzzy", "acronym"]:
            return
        if col == "hq_country":
            filters.pop("hq_state", None)
            filters.pop("hq_city", None)
            origin_map.pop("hq_state", None)
            origin_map.pop("hq_city", None)
        elif col == "hq_state":
            filters.pop("hq_city", None)
            origin_map.pop("hq_city", None)
        filters[col] ={val}
        origin_map[col] = origin
    
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
            matched_tokens.update(phrase_lower)

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

def search(query_clean, filters):
    try:
        mask = pd.Series([True] * len(db))

        for col, vals in filters.items():
            if col == "total_funding_raised":
                for op, threshold in vals.items():
                    if op == ">":
                        mask &= db["total_funding_raised"].apply(lambda x: float(x or 0) > threshold)
            elif col in db.columns:
                mask &= db[col].isin(vals)

        # Fallback: loose token-based filter if needed
        tokens = set(re.findall(r"\w+", query_clean.lower()))
        fallback = pd.Series([False] * len(db))
        for col in ["description", "business_area", "business_activity", "sector", "hq_city", "hq_state", "hq_country"]:
            if col in db.columns:
                fallback |= db[col].str.contains("|".join(tokens), case=False, na=False)

        # Debug if fuzzy match works
        logger.info(f"Applying filters: {filters}")
        logger.info(f"Number of results before fallback: {len(db[mask])}")
        logger.info(f"Number of results after fallback: {len(db[mask | fallback])}")
        return db[mask | fallback].copy()
    except Exception as e:
        logger.error(f"An error occurred during search: {str(e)}", exc_info=True)
        return pd.DataFrame()

def format_location(row):
    try:
        city = row.get("hq_city", "").strip().title()
        state = row.get("hq_state", "").strip().upper()
        country = row.get("hq_country", "").strip().title()

        if state:
            return f"{city}, {state}"
        elif country:
            return f"{city}, {country}"
        return city
    except Exception as e:
        logger.error(f"Error formatting location: {str(e)}", exc_info=True)
        return ""

# ----------------------------- Flask Routes -----------------------------

@app.route("/parse", methods=["POST"])
def parse():
    try:
        data = request.get_json()
        query_clean = data.get("query_clean", "").strip()
        if not query_clean:
            logger.warning("Missing query string")
            return jsonify({"error": "Missing query"}), 400

        logger.info(f"Received query: {query_clean}")
        parsed_query = extract_filters(query_clean)
        filters = parsed_query["filters"]
        free_text_terms = parsed_query["free_text_terms"]
        logger.info(f"Filters applied: {filters}")
        logger.info(f"Free text terms applied: {free_text_terms}")
        
        results = search(query_clean, filters, free_text_terms)
        logger.info(f"{len(results)} results found")

        response = [{
            "company_name": row.get("company_name", ""),
            "business_activity": row.get("business_activity", ""),
            "business_area": row.get("business_area", ""),
            "description": row.get("description", ""),
            "hq_location": format_location(row),
            "website_url": row.get("website_url", "")
        } for _, row in results.iterrows()]

        return jsonify(response)

    except Exception as e:
        logger.error(f"Request error: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/")
def home():
    return "ImFo NLP API is live."

if __name__ == "__main__":
    app.run(debug=True)
