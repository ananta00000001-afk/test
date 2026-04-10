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
    # Prefer sale price when present, otherwise last number.
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
        row.get("description", ""),
        row.get("price_text", ""),
        row.get("primary_image_url", ""),
        row.get("product_page_url", ""),
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
            clean_row["sale_price_bdt"] = extract_price_bdt(clean_row.get("price_text", ""))
            clean_row["search_text"] = build_search_text(clean_row)
            items.append(clean_row)
    logger.info("Loaded %d catalog items", len(items))
    return items


# Global catalog loaded once at import time
CATALOG = load_catalog()


# =========================================================
# Catalog search (token-based scoring)
# =========================================================

def score_catalog(query_text: str, catalog: List[Dict[str, Any]], top_k: int = 5, min_score: int = 6) -> List[Dict[str, Any]]:
    """
    Score catalog items against a search query.
    min_score: minimum score required to consider a match valid.
    Returns empty list if no product scores above the threshold (avoids wrong matches).
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
        desc = normalize_text(row.get("description", ""))
        for token in query_tokens:
            if token in search_text:
                score += 2
            if token in name:
                score += 3
            if token in desc:
                score += 4  # Description keywords are most specific
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
