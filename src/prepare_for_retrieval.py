# Build a cleaner tailored to your *updated* JSON structure and produce a retrieval-ready index.
# We'll:
# 1) Load /mnt/data/processed_info_updated.json
# 2) Flatten ingredients from the new nested format (list[ {category: [lines...]} ])
# 3) Normalize each line for retrieval (remove numbers/units/fluff; keep meaning; fix synonyms)
# 4) Emit a line-level index CSV with: recipe_id, title, category, line_id, text_raw, text_retrieval

import json, re, unicodedata, pandas as pd
import os
from typing import List, Dict

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURR_PATH)  # Go up one level to project root
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed_info.json")
assert os.path.exists(DATA_PATH), "processed_info.json not found."

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# -------------------------
# 1) Domain dictionaries (same spirit as before, safe & conservative)
# -------------------------
UNIT_MAP = {
    # spoons/cups
    "tsp": "teaspoon", "tsps": "teaspoon",
    "tbsp": "tablespoon", "tbsps": "tablespoon",
    "cup": "cup", "cups": "cup",
    # mass/volume
    "g": "g", "kg": "kg", "gram": "g", "grams": "g",
    "ml": "ml", "l": "l", "litre": "l", "liter": "l",
    "oz": "oz", "lb": "lb", "lbs": "lb",
}
# Units to DROP from retrieval
UNITS_STOP = set(["g","kg","ml","l","cup","tablespoon","teaspoon","oz","lb"])

# Fluff phrases
PHRASE_STOP = [
    r"\bof your choice\b",
    r"\bto taste\b",
    r"\bto garnish\b",
    r"\bfor garnish\b",
    r"\bto finish\b",
    r"\bfor serving\b",
    r"\bdrizzle of\b",
    r"\bi used .*?\b",
    r"\bor to taste\b",
    r"\bamount up to you\b",
]

# Synonyms/variants
SYNONYMS = {
    "cous cous": "couscous",
    "pak choi": "bok choy",
    "pak choy": "bok choy",
    "baby boy choy": "bok choy",
    "scallion": "spring onion",
    "cilantro": "coriander",
}

# Keep meaningful words inside parentheses if present
PAREN_KEEP = {
    "onion","scallion","spring","coriander","cilantro","parsley","tomato","chili","pepper","lemon","lime",
    "yogurt","yoghurt","shallot","basil","mint","cilantro"
}

# Plural map (conservative) and words we must not truncate
PLURAL_MAP = {
    "cloves": "clove",
    "leaves": "leaf",
    "sprigs": "sprig",
    "pods": "pod",
    "eggs": "egg",
    "tomatoes": "tomato",
    "shallots": "shallot",
    "cherries": "cherry",
    "berries": "berry",
}
PROTECT_S = {"couscous","hummus","asparagus"}

# Participles normalization (no global -ed/-ing strip)
PARTICIPLE_MAP = {
    "diced":"dice","chopped":"chop","minced":"mince","grated":"grate","sliced":"slice",
    "crushed":"crush","peeled":"peel","zested":"zest","juiced":"juice","roasted":"roast",
    "toasted":"toast","bruised":"bruise","halved":"halve","mashed":"mash","rinsed":"rinse",
    "washed":"wash","thinly":"thinly","roughly":"roughly","shredded":"shred"
}

STOP_TOKENS = set(["_num_", "num"]) | UNITS_STOP | {"small","pinch","seasoning","general","ingredient","optional"}

# -------------------------
# 2) Normalization helpers
# -------------------------
def fix_synonyms(s: str) -> str:
    for k, v in SYNONYMS.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    return s

def parentheses_repl(match: re.Match) -> str:
    inner = match.group(1)
    words = re.findall(r"[a-z]+", inner)
    kept = [w for w in words if w in PAREN_KEEP]
    return (" " + " ".join(kept) + " ") if kept else " "

def normalize_units_whole_tokens(s: str) -> str:
    for k, v in UNIT_MAP.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    return s

def tokenize_letters(s: str) -> List[str]:
    return re.findall(r"[a-z]+", s)

def lemmatize_recipe_tokens(tokens: List[str]) -> List[str]:
    out = []
    for w in tokens:
        ww = w
        if ww in PROTECT_S:
            out.append(ww); continue
        if ww in PLURAL_MAP:
            ww = PLURAL_MAP[ww]
        if ww in PARTICIPLE_MAP:
            ww = PARTICIPLE_MAP[ww]
        out.append(ww)
    return out

def normalize_line_for_retrieval(line: str) -> str:
    s = unicodedata.normalize("NFKC", line).lower().strip()
    s = re.sub(r"\((.*?)\)", parentheses_repl, s)  # keep meaningful words from ()
    s = s.replace("&", " and ")
    s = fix_synonyms(s)
    s = re.sub(r"[-,/:+]", " ", s)                # unify separators
    s = re.sub(r"\b\d+(?:\.\d+)?\b", " ", s)      # drop explicit numbers
    s = normalize_units_whole_tokens(s)           # normalize unit tokens
    for pat in PHRASE_STOP:                       # remove fluff phrases
        s = re.sub(pat, " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = tokenize_letters(s)
    tokens = [t for t in tokens if t not in STOP_TOKENS]
    tokens = lemmatize_recipe_tokens(tokens)
    tokens = [t for t in tokens if t]
    return " ".join(tokens)

# -------------------------
# 3) Flatten the new ingredients structure
#    Each entry in "ingredients" is a dict: {category_name: [lines...]}
# -------------------------
def flatten_new_ingredients(ing_list: List[Dict[str, List[str]]]) -> List[Dict[str, str]]:
    out = []
    if not isinstance(ing_list, list):
        return out
    for block in ing_list:
        if not isinstance(block, dict):
            continue
        for cat, lines in block.items():
            if isinstance(lines, list):
                for i, line in enumerate(lines):
                    if not isinstance(line, str): 
                        continue
                    out.append({"category": str(cat).strip(), "line": line.strip(), "line_id": i})
    return out

# -------------------------
# 4) Build retrieval docs
# -------------------------
if __name__ == "__main__":
    
    records = []
    for ridx, row in df.iterrows():
        title = row.get("titles") or row.get("title") or f"recipe_{ridx}"
        ing_blocks = row.get("ingredients", [])
        flat = flatten_new_ingredients(ing_blocks)
        for item in flat:
            raw = item["line"]
            cat = item["category"]
            line_id = item["line_id"]
            ret = normalize_line_for_retrieval(raw)
            if not ret:
                continue
            records.append({
                "recipe_id": ridx,
                "title": title,
                "category": cat,
                "line_id": line_id,
                "text_raw": raw,
                "text_retrieval": ret,
            })

    idx_df = pd.DataFrame(records).sort_values(["recipe_id","category","line_id"])

    # Save
    OUT = os.path.join(PROJECT_ROOT, "data", "ingredient_index.csv")
    idx_df.to_csv(OUT, index=False)
    print(f"Saved retrieval index to {OUT} with {len(idx_df)} lines.")