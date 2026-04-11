"""
Product catalog loading, searching, and utility functions for Baby Catalog Bot V5.
"""

import csv
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from config import CATALOG_CSV_PATH, logger


# =========================================================
# Text normalization
# =========================================================

def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\u0980-\u09ff\s&-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# =========================================================
# Price extraction
# =========================================================

def extract_price_bdt(price_text: str) -> Optional[int]:
    price_text = price_text or ""
    nums = re.findall(r"([0-9,]+(?:\.\d+)?)", price_text)
    if not nums:
        return None
    try:
        cleaned = nums[-1].replace(",", "")
        return int(float(cleaned))
    except Exception:
        return None


# =========================================================
# Catalog loading
# =========================================================

def build_search_text(row: Dict[str, Any]) -> str:
    parts = [
        row.get("product_name", ""),
        row.get("price_text", ""),
        row.get("type", ""),
        row.get("color", ""),
        row.get("sizes", ""),
    ]
    return normalize_text(" ".join(parts))


def load_catalog() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.exists(CATALOG_CSV_PATH):
        logger.warning("Catalog file not found at: %s", CATALOG_CSV_PATH)
        return items
    with open(CATALOG_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {k: (v or "").strip() for k, v in row.items()}
            # Ensure sale_price_bdt is set from either sale_price column or price_text
            sp = clean_row.get("sale_price", "")
            if sp:
                try:
                    clean_row["sale_price_bdt"] = int(float(sp))
                except Exception:
                    clean_row["sale_price_bdt"] = extract_price_bdt(clean_row.get("price_text", ""))
            else:
                clean_row["sale_price_bdt"] = extract_price_bdt(clean_row.get("price_text", ""))
            clean_row["search_text"] = build_search_text(clean_row)
            items.append(clean_row)
    logger.info("Loaded %d catalog items", len(items))
    return items


# Global catalog loaded once at import time
CATALOG = load_catalog()


# =========================================================
# Catalog search (token-based scoring — for text queries)
# =========================================================

def score_catalog(query_text: str, catalog: List[Dict[str, Any]], top_k: int = 5, min_score: int = 6) -> List[Dict[str, Any]]:
    """
    Score catalog items against a text search query.
    Used when customer types text (not images). For image matching, use visual_match_product.
    """
    q = normalize_text(query_text)
    if not q:
        return []

    query_tokens = set(q.split())
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for row in catalog:
        score = 0
        search_text = row.get("search_text", "")
        name = normalize_text(row.get("product_name", ""))
        color_field = normalize_text(row.get("color", ""))
        ptype = normalize_text(row.get("type", ""))
        for token in query_tokens:
            if token in search_text:
                score += 2
            if token in name:
                score += 3
            if token in color_field:
                score += 4
            if token in ptype:
                score += 3
        if score >= min_score:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_k]]


def get_random_catalog_products(count: int = 3) -> List[Dict[str, Any]]:
    valid = [
        row for row in CATALOG
        if row.get("primary_image_url") and row.get("product_name")
    ]
    if not valid:
        return []
    if len(valid) <= count:
        return valid
    return random.sample(valid, count)


def _normalize_id(pid: str) -> str:
    """Normalize a product ID for fuzzy matching."""
    # Remove non-ASCII (emojis), lowercase, collapse dashes
    import re as _re
    cleaned = _re.sub(r'[^\x00-\x7f]+', '', pid).lower().strip()
    cleaned = _re.sub(r'-+', '-', cleaned).strip('-')
    return cleaned


def find_product_by_id(product_id: str) -> Optional[Dict[str, Any]]:
    """
    Find a product by its product_id (handle).
    Tries exact match first, then normalized match (strips emoji, lowercases).
    """
    if not product_id:
        return None
    # Exact match
    for row in CATALOG:
        if row.get("product_id") == product_id:
            return row
    # Normalized match (handles Gemini returning slightly different IDs)
    norm_query = _normalize_id(product_id)
    if not norm_query:
        return None
    best_match = None
    best_score = 0
    for row in CATALOG:
        norm_catalog = _normalize_id(row.get("product_id", ""))
        if norm_catalog == norm_query:
            return row  # Exact normalized match
        # Partial match: check if one contains the other
        if norm_query in norm_catalog or norm_catalog in norm_query:
            score = len(norm_catalog)
            if score > best_score:
                best_score = score
                best_match = row
    return best_match


# =========================================================
# Visual matching prompt builder
# =========================================================

def build_visual_catalog_prompt() -> str:
    """
    Build a text listing of all catalog products with image URLs
    for Gemini visual matching.
    """
    lines = []
    for i, row in enumerate(CATALOG, 1):
        pid = row.get("product_id", "")
        name = row.get("product_name", "")
        price = row.get("price_text", "")
        img = row.get("primary_image_url", "")
        if not pid or not name:
            continue
        lines.append(f"{i}. product_id: {pid}")
        lines.append(f"   name: {name}")
        lines.append(f"   price: {price}")
        lines.append(f"   image_url: {img}")
        lines.append("")
    return "\n".join(lines)


# Pre-build the visual catalog prompt at import time
VISUAL_CATALOG_PROMPT = build_visual_catalog_prompt()


# =========================================================
# Intent detection helpers
# =========================================================

def wants_pictures(text: str) -> bool:
    t = normalize_text(text)
    triggers = [
        "picture", "pictures", "pic", "pics",
        "photo", "photos",
        "chobi", "chobi dekhan", "picture dekhan",
        "pics dekhan", "pic dekhan",
        "ছবি", "ছবি দেখান", "কিছু ছবি দেখান",
        "পিকচার", "পিকচার দেখান",
        "show picture", "show me pictures",
        "kichu dekhan",
    ]
    return any(x in t for x in triggers)


def parse_phone_from_text(text: str) -> str:
    raw = text or ""
    m = re.search(r"(?:\+?88)?(01[3-9]\d{8})", raw)
    return m.group(1) if m else ""


def likely_address(text: str) -> bool:
    t = normalize_text(text)
    if len(t) < 10:
        return False
    markers = [
        "road", "rd", "house", "flat", "area", "district", "thana", "upazila", "village",
        "road no", "sector", "ঢাকা", "জেলা", "থানা", "রোড", "বাড়ি", "বাসা", "ফ্ল্যাট",
        "গ্রাম", "উপজেলা", "পোস্ট", "area", "mirpur", "uttara", "dhaka", "cumilla",
        "chattogram", "sylhet", "khulna", "rajshahi", "barishal", "rangpur", "mymensingh",
    ]
    return any(m in t for m in markers)


def infer_delivery_zone(address: str) -> str:
    a = normalize_text(address)
    if not a:
        return ""
    dhaka_markers = [
        "dhaka", "ঢাকা", "mirpur", "মিরপুর", "uttara", "উত্তরা", "mohammadpur",
        "মোহাম্মদপুর", "banasree", "banani", "ধানমন্ডি", "dhanmondi", "bashundhara",
        "bashabo", "badda", "ramna", "motijheel", "jatrabari", "tejgaon", "gulshan",
    ]
    return "inside_dhaka" if any(m in a for m in dhaka_markers) else "outside_dhaka"
