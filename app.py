import csv
import io
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from google import genai
from google.genai import types

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
FACEBOOK_PAGE_ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN", "")
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-3.1-pro-preview")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.1-pro-preview")
CATALOG_CSV_PATH = os.getenv("CATALOG_CSV_PATH", "./catalog.csv")
SQLITE_PATH = os.getenv("SQLITE_PATH", "./state.db")
ORDER_WEBHOOK_URL = os.getenv("ORDER_WEBHOOK_URL", "")
BOT_METADATA = "BOT_AUTO_V3"
GRAPH_VERSION = "v25.0"

app = FastAPI(title="Baby Catalog Bot Gemini v3")

genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SIZE_GUIDE = [
    {"min_month": 0, "max_month": 3, "size": "0-3M"},
    {"min_month": 3, "max_month": 6, "size": "3-6M"},
    {"min_month": 6, "max_month": 9, "size": "6-9M"},
    {"min_month": 9, "max_month": 12, "size": "9-12M"},
    {"min_month": 12, "max_month": 18, "size": "12-18M"},
    {"min_month": 18, "max_month": 24, "size": "18-24M"},
]

REPLY_SYSTEM_PROMPT = """
You are writing the FINAL customer-facing reply for a Bangla Messenger sales chat for a baby clothing store.

Rules:
- Return ONLY the final message text that will be sent to the customer.
- Do NOT return JSON.
- Do NOT return YAML.
- Do NOT return markdown.
- Do NOT return bullet points.
- Do NOT return labels like Ask, Reply, delivery_eta, empty, 2 short sentences, or placeholders.
- Do NOT show internal reasoning.
- If any field is missing, ignore it and write the best natural reply from the available facts.
- Write in natural conversational Bengali using Bangla script.
- Sound warm, human, and consultative.
- Maximum 2 short sentences.
- Ask at most one simple question.
- If the customer has corrected gender/addressing, respect that.
- If the customer has corrected delivery zone or other details, respect the corrected value.
- If matched product, price, size, delivery charge, or ETA are available, use them naturally.
- If the order qualifies for free delivery, mention it naturally when useful.
- If the order is below the threshold, delivery charge is exactly ৳100.
- If message_count is 4 or more, you may softly encourage order confirmation once.
- Never invent product names, prices, sizes, or delivery times.
- Never send internal planning text.

Return only the final customer-facing message.
"""

VISION_ANALYSIS_PROMPT = """
Analyze the customer message and optional product/baby image for a Bangladeshi babywear sales chat.

Return ONLY valid JSON with this shape:
{
  "user_intent": "image_only|ask_price|ask_size|ask_both|buy|availability|general",
  "product_keywords": ["..."],
  "product_category": "",
  "color_or_theme": "",
  "matched_product_hint": "",
  "estimated_age_range": "",
  "estimated_age_months_midpoint": 0,
  "recommended_size_hint": "",
  "confidence": "low|medium|high",
  "needs_followup": true,
  "followup_question": "",
  "image_clear_enough": true,
  "reasoning_summary": ""
}

Rules:
- If the image appears to be a baby photo for sizing, estimate age range and size hint if possible.
- If the image appears to be a product photo, extract product cues.
- If the user sent only a picture and no text, set user_intent to image_only.
- If uncertain, set needs_followup=true and ask one short question.
- Keep reasoning_summary short and factual.
"""

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_customer_reply(text: str) -> str:
    if not text:
        return "আপনি চাইলে আমি অর্ডারটা এগিয়ে নিতে পারি 🌸 সাইজটা কনফার্ম করবেন?"

    keep = []
    for line in text.splitlines():
        l = line.strip()
        low = l.lower()
        if not l:
            continue
        banned_prefixes = (
            'delivery_eta:', 'delivery_charge:', 'free_delivery:', 'matched_product_name:',
            'matched_price:', 'recommended_size:', 'confidence:', 'order_stage:', 'ask:', 'reply:'
        )
        if low in {'(empty)', 'empty', '* ask', 'ask', 'reply'}:
            continue
        if '2 short sentences' in low or 'return only' in low or 'backend facts' in low:
            continue
        if low.startswith(banned_prefixes):
            continue
        keep.append(l)

    cleaned = re.sub(r'\s+', ' ', ' '.join(keep)).strip()
    if not cleaned:
        return "আপনি চাইলে আমি অর্ডারটা এগিয়ে নিতে পারি 🌸 সাইজটা কনফার্ম করবেন?"
    return cleaned

def ensure_db():
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS customer_state (
            psid TEXT PRIMARY KEY,
            customer_name TEXT,
            addressing_style TEXT DEFAULT 'আপু',
            human_operator_engaged INTEGER DEFAULT 0,
            message_count INTEGER DEFAULT 0,
            matched_product_name TEXT,
            matched_price TEXT,
            matched_image_url TEXT,
            matched_product_page_url TEXT,
            recommended_size TEXT,
            size_finalized INTEGER DEFAULT 0,
            confidence TEXT,
            phone TEXT,
            address TEXT,
            delivery_zone_inferred TEXT,
            delivery_zone_final TEXT,
            delivery_zone_source TEXT,
            delivery_eta TEXT,
            delivery_charge INTEGER DEFAULT 100,
            free_delivery INTEGER DEFAULT 0,
            order_stage TEXT DEFAULT 'browsing',
            gender_note TEXT,
            last_user_intent TEXT,
            last_user_message TEXT,
            operator_locked_at TEXT,
            order_sent INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def db_get_state(psid: str) -> Dict[str, Any]:
    ensure_db()
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM customer_state WHERE psid = ?", (psid,))
    row = cur.fetchone()
    if row:
        conn.close()
        return dict(row)
    state = {
        "psid": psid,
        "customer_name": "",
        "addressing_style": "আপু",
        "human_operator_engaged": 0,
        "message_count": 0,
        "matched_product_name": "",
        "matched_price": "",
        "matched_image_url": "",
        "matched_product_page_url": "",
        "recommended_size": "",
        "size_finalized": 0,
        "confidence": "",
        "phone": "",
        "address": "",
        "delivery_zone_inferred": "",
        "delivery_zone_final": "",
        "delivery_zone_source": "",
        "delivery_eta": "",
        "delivery_charge": 100,
        "free_delivery": 0,
        "order_stage": "browsing",
        "gender_note": "",
        "last_user_intent": "",
        "last_user_message": "",
        "operator_locked_at": "",
        "order_sent": 0,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    db_upsert_state(state)
    conn.close()
    return state

def db_upsert_state(state: Dict[str, Any]):
    ensure_db()
    state["updated_at"] = now_iso()
    if not state.get("created_at"):
        state["created_at"] = now_iso()
    keys = list(state.keys())
    values = [state[k] for k in keys]
    placeholders = ", ".join(["?"] * len(keys))
    columns = ", ".join(keys)
    updates = ", ".join([f"{k}=excluded.{k}" for k in keys if k != "psid"])
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO customer_state ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT(psid) DO UPDATE SET {updates}",
        values,
    )
    conn.commit()
    conn.close()

def load_catalog() -> List[Dict[str, Any]]:
    items = []
    if not os.path.exists(CATALOG_CSV_PATH):
        return items
    with open(CATALOG_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k: (v or "").strip() for k, v in row.items()}
            row["sale_price_bdt"] = extract_price_bdt(row.get("price_text", ""))
            row["search_text"] = build_search_text(row)
            items.append(row)
    return items


def build_search_text(row: Dict[str, Any]) -> str:
    parts = [
        row.get("product_name", ""),
        row.get("price_text", ""),
        row.get("primary_image_url", ""),
        row.get("product_page_url", ""),
    ]
    txt = " ".join(parts).lower()
    txt = re.sub(r"[^a-z0-9\u0980-\u09ff\s]+", " ", txt)
    return txt

def extract_price_bdt(price_text: str) -> Optional[int]:
    nums = re.findall(r"Tk\s*([0-9,]+(?:\.\d+)?)", price_text)
    if not nums:
        nums = re.findall(r"([0-9,]+)", price_text)
    if not nums:
        return None
    try:
        cleaned = nums[-1].replace(",", "")
        return int(float(cleaned))
    except Exception:
        return None


CATALOG = load_catalog()
def normalize_bengali_digits(text: str) -> str:
    bengali_to_english = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
    return text.translate(bengali_to_english)

def infer_zone_from_address(address: str) -> Tuple[str, str]:
    a = address.lower()
    dhaka_markers = ["dhaka", "ঢাকা", "mirpur", "uttara", "mohammadpur", "banani", "badda", "gulshan", "dhanmondi", "jatrabari", "rampura", "khilgaon"]
    if any(m in a for m in dhaka_markers):
        return "inside_dhaka", "২৪ ঘণ্টা"
    return "outside_dhaka", "৪৮–৭২ ঘণ্টা"

def update_delivery_from_state(state: Dict[str, Any]):
    zone = state.get("delivery_zone_final") or state.get("delivery_zone_inferred")
    if zone == "inside_dhaka":
        state["delivery_eta"] = "২৪ ঘণ্টা"
    elif zone == "outside_dhaka":
        state["delivery_eta"] = "৪৮–৭২ ঘণ্টা"
    price = extract_price_bdt(state.get("matched_price") or "")
    if price is None:
        price = 0
    if price >= 999:
        state["free_delivery"] = 1
        state["delivery_charge"] = 0
    else:
        state["free_delivery"] = 0
        state["delivery_charge"] = 100

def is_complete_order(state: Dict[str, Any]) -> bool:
    return bool(state.get("recommended_size") and state.get("phone") and state.get("address"))

def maybe_apply_corrections(text: str, state: Dict[str, Any]):
    t = (text or "").lower()

    if any(p in t for p in ["not female", "আমি মেয়ে না", "আমি female না", "আপু না", "ভাই বলেন", "i am male", "আমি ছেলে"]):
        if "ভাই" in t or "male" in t or "ছেলে" in t:
            state["addressing_style"] = "ভাই"
            state["gender_note"] = "male_corrected"
        else:
            state["addressing_style"] = "আপনি"
            state["gender_note"] = "not_female_corrected"

    if any(p in t for p in ["inside dhaka", "ঢাকার ভিতরে", "আমি ঢাকায়", "i am inside dhaka"]):
        state["delivery_zone_final"] = "inside_dhaka"
        state["delivery_zone_source"] = "customer_corrected"
    elif any(p in t for p in ["outside dhaka", "ঢাকার বাইরে", "আমি ঢাকার বাইরে", "i am outside dhaka"]):
        state["delivery_zone_final"] = "outside_dhaka"
        state["delivery_zone_source"] = "customer_corrected"

    size_match = re.search(r"\b(\d{1,2}\s*-\s*\d{1,2}\s*m)\b", normalize_bengali_digits(t))
    if size_match:
        state["recommended_size"] = size_match.group(1).replace(" ", "").upper()
        state["size_finalized"] = 1

    phone = extract_phone(text or "")
    if phone:
        state["phone"] = phone

def extract_phone(text: str) -> str:
    t = normalize_bengali_digits(text)
    m = re.search(r'(?:\+?88)?(01[3-9]\d{8})', t)
    return m.group(1) if m else ""

def maybe_extract_address(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if len(t) < 8:
        return ""
    cues = ["road", "street", "house", "বাসা", "রোড", "থানা", "উপজেলা", "জেলা", "ঢাকা", "chittagong", "khulna", "rajshahi", "barisal", "sylhet", "rangpur", "mymensingh", ","]
    if any(c in t.lower() for c in cues):
        return t
    return ""

def age_months_to_size(months: int) -> str:
    for item in SIZE_GUIDE:
        if item["min_month"] <= months < item["max_month"]:
            return item["size"]
    return "12-18M"

async def gemini_json_from_parts(parts: List[types.Part], model_name: str, prompt: str) -> Dict[str, Any]:
    if not genai_client:
        return {}
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)] + parts)
    ]
    response = genai_client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )
    txt = (response.text or "").strip()
    try:
        return json.loads(txt)
    except Exception:
        try:
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}

async def analyze_customer_message_and_image(message_text: str, image_bytes: Optional[bytes], image_mime: str) -> Dict[str, Any]:
    parts: List[types.Part] = [types.Part.from_text(text=f"Customer message: {message_text or ''}")]
    if image_bytes:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime or "image/jpeg"))
    data = await gemini_json_from_parts(parts, GEMINI_VISION_MODEL, VISION_ANALYSIS_PROMPT)
    if not data:
        data = {
            "user_intent": "general" if message_text else "image_only",
            "product_keywords": [],
            "product_category": "",
            "color_or_theme": "",
            "matched_product_hint": "",
            "estimated_age_range": "",
            "estimated_age_months_midpoint": 0,
            "recommended_size_hint": "",
            "confidence": "low",
            "needs_followup": True,
            "followup_question": "বেবুর বয়সটা বলবেন আপু?",
            "image_clear_enough": bool(image_bytes),
            "reasoning_summary": ""
        }
    return data

def score_catalog_item(item: Dict[str, Any], analysis: Dict[str, Any], user_text: str) -> int:
    score = 0
    txt = item.get("search_text", "")
    for kw in analysis.get("product_keywords", []) or []:
        if str(kw).lower() in txt:
            score += 3
    for kw in re.findall(r"[a-zA-Z\u0980-\u09ff0-9]+", (user_text or "").lower()):
        if len(kw) > 2 and kw in txt:
            score += 1
    hint = (analysis.get("matched_product_hint") or "").lower()
    if hint and hint in txt:
        score += 4
    category = (analysis.get("product_category") or "").lower()
    if category and category in txt:
        score += 2
    theme = (analysis.get("color_or_theme") or "").lower()
    if theme and theme in txt:
        score += 2
    return score

def match_best_product(analysis: Dict[str, Any], user_text: str) -> Optional[Dict[str, Any]]:
    if not CATALOG:
        return None
    scored = []
    for item in CATALOG:
        scored.append((score_catalog_item(item, analysis, user_text), item))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_item = scored[0]
    return best_item if best_score > 0 else None

async def generate_reply_text(customer_message: str, state: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    if not genai_client:
        return "জি, একটু পরে আবার মেসেজ দিন।"
    facts = {
        "customer_name": state.get("customer_name") or "",
        "addressing_style": state.get("addressing_style") or "আপু",
        "user_intent": analysis.get("user_intent") or state.get("last_user_intent") or "general",
        "has_image": bool(analysis.get("image_clear_enough")),
        "image_only": bool((not customer_message.strip()) and analysis.get("image_clear_enough")),
        "matched_product_name": state.get("matched_product_name") or "",
        "matched_price": state.get("matched_price") or "",
        "matched_image_url": state.get("matched_image_url") or "",
        "recommended_size": state.get("recommended_size") or "",
        "confidence": state.get("confidence") or analysis.get("confidence") or "",
        "size_finalized": bool(state.get("size_finalized")),
        "free_delivery": bool(state.get("free_delivery")),
        "delivery_charge": state.get("delivery_charge"),
        "delivery_eta": state.get("delivery_eta") or "",
        "order_stage": state.get("order_stage") or "browsing",
        "message_count": state.get("message_count", 0),
        "phone_present": bool(state.get("phone")),
        "address_present": bool(state.get("address")),
    }
    backend_facts = "Backend facts:\n" + "\n".join([f"- {k}: {v}" for k, v in facts.items()])
    user_block = f"{backend_facts}\n\nCustomer message:\n{customer_message or '[image only]'}"
    response = genai_client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=user_block,
        config=types.GenerateContentConfig(
            system_instruction=REPLY_SYSTEM_PROMPT,
            temperature=0.6,
            max_output_tokens=220,
        ),
    )
    text = clean_customer_reply((response.text or "").strip())
    return text or "জি, একটু বুঝিয়ে বলবেন?"

async def fetch_binary(url: str) -> Tuple[Optional[bytes], str]:
    if not url:
        return None, ""
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        r = await client.get(url)
        if r.status_code >= 400:
            return None, ""
        mime = r.headers.get("content-type", "image/jpeg").split(";")[0]
        return r.content, mime

async def send_facebook_text(psid: str, text: str):
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{FACEBOOK_PAGE_ID}/messages"
    payload = {
        "recipient": {"id": psid},
        "messaging_type": "RESPONSE",
        "message": {"text": text, "metadata": BOT_METADATA},
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            params={"access_token": FACEBOOK_PAGE_ACCESS_TOKEN},
            json=payload,
            timeout=30.0,
        )
        print("Facebook text status:", response.status_code)
        print("Facebook text body:", response.text)
        response.raise_for_status()

async def send_facebook_image(psid: str, image_url: str):
    if not image_url:
        return
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{FACEBOOK_PAGE_ID}/messages"
    payload = {
        "recipient": {"id": psid},
        "messaging_type": "RESPONSE",
        "message": {
            "attachment": {
                "type": "image",
                "payload": {"url": image_url, "is_reusable": False}
            },
            "metadata": BOT_METADATA,
        },
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            params={"access_token": FACEBOOK_PAGE_ACCESS_TOKEN},
            json=payload,
            timeout=30.0,
        )
        print("Facebook image status:", response.status_code)
        print("Facebook image body:", response.text)
        response.raise_for_status()

def maybe_capture_profile_name(state: Dict[str, Any], event: Dict[str, Any]):
    if state.get("customer_name"):
        return
    # Messenger webhook payload usually doesn't include profile name directly.
    # Keep graceful fallback; can be extended with Graph profile lookup if enabled.
    return

async def post_order_to_webhook(state: Dict[str, Any]):
    if not ORDER_WEBHOOK_URL:
        return
    payload = {
        "customer_name": state.get("customer_name") or "",
        "facebook_psid": state.get("psid"),
        "addressing_style": state.get("addressing_style") or "",
        "product_name": state.get("matched_product_name") or "",
        "product_price": state.get("matched_price") or "",
        "product_image_url": state.get("matched_image_url") or "",
        "recommended_size": state.get("recommended_size") or "",
        "phone": state.get("phone") or "",
        "address": state.get("address") or "",
        "delivery_zone": state.get("delivery_zone_final") or state.get("delivery_zone_inferred") or "",
        "delivery_eta": state.get("delivery_eta") or "",
        "delivery_charge": state.get("delivery_charge") or 100,
        "free_delivery": bool(state.get("free_delivery")),
        "order_stage": state.get("order_stage") or "confirmed",
        "timestamp": now_iso(),
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(ORDER_WEBHOOK_URL, json=payload)
        print("Order webhook status:", r.status_code)
        print("Order webhook body:", r.text)
        r.raise_for_status()

def maybe_lock_for_human_operator(event: Dict[str, Any], state: Dict[str, Any]) -> bool:
    message = event.get("message") or {}
    if not message.get("is_echo"):
        return False
    metadata = message.get("metadata") or ""
    # If page echoes a message not marked as bot metadata, treat it as human takeover.
    if metadata != BOT_METADATA:
        state["human_operator_engaged"] = 1
        state["operator_locked_at"] = now_iso()
        state["order_stage"] = "human_takeover"
        db_upsert_state(state)
        return True
    return False

def next_stage(state: Dict[str, Any]):
    if not state.get("matched_product_name"):
        state["order_stage"] = "browsing"
    elif not state.get("recommended_size"):
        state["order_stage"] = "awaiting_size"
    elif not state.get("address"):
        state["order_stage"] = "awaiting_address"
    elif not state.get("phone"):
        state["order_stage"] = "awaiting_phone"
    else:
        state["order_stage"] = "confirmed"

async def process_event(event: Dict[str, Any]):
    sender = event.get("sender", {}).get("id")
    if not sender:
        return
    state = db_get_state(sender)

    if maybe_lock_for_human_operator(event, state):
        print(f"Human operator lock enabled for {sender}")
        return

    if state.get("human_operator_engaged"):
        print(f"Skipping AI for {sender}; human operator lock active")
        return

    maybe_capture_profile_name(state, event)

    message = event.get("message") or {}
    customer_text = message.get("text", "") or ""
    state["message_count"] = int(state.get("message_count") or 0) + 1
    state["last_user_message"] = customer_text

    maybe_apply_corrections(customer_text, state)

    attachment_url = ""
    image_bytes = None
    image_mime = ""
    attachments = message.get("attachments") or []
    for att in attachments:
        if (att.get("type") or "").lower() == "image":
            attachment_url = ((att.get("payload") or {}).get("url") or "")
            break

    if attachment_url:
        image_bytes, image_mime = await fetch_binary(attachment_url)

    analysis = await analyze_customer_message_and_image(customer_text, image_bytes, image_mime)
    state["last_user_intent"] = analysis.get("user_intent") or state.get("last_user_intent")

    # Product match
    best_product = match_best_product(analysis, customer_text)
    if best_product:
        state["matched_product_name"] = best_product.get("product_name", "")
        price_num = best_product.get("sale_price_bdt")
        state["matched_price"] = f"৳{price_num}" if price_num else (best_product.get("price_text") or "")
        state["matched_image_url"] = best_product.get("primary_image_url", "")
        state["matched_product_page_url"] = best_product.get("product_page_url", "")

    # Size inference / explicit extraction
    if not state.get("recommended_size"):
        hint = analysis.get("recommended_size_hint") or ""
        if hint:
            state["recommended_size"] = hint
        months = int(analysis.get("estimated_age_months_midpoint") or 0)
        if months > 0 and not state.get("recommended_size"):
            state["recommended_size"] = age_months_to_size(months)
        if state.get("recommended_size"):
            state["confidence"] = analysis.get("confidence") or "medium"

    # Address + phone capture
    if not state.get("phone"):
        phone = extract_phone(customer_text)
        if phone:
            state["phone"] = phone

    extracted_address = maybe_extract_address(customer_text)
    if extracted_address:
        state["address"] = extracted_address
        if not state.get("delivery_zone_final"):
            zone, eta = infer_zone_from_address(extracted_address)
            state["delivery_zone_inferred"] = zone
            state["delivery_eta"] = eta
            if not state.get("delivery_zone_source"):
                state["delivery_zone_source"] = "inferred_from_address"

    update_delivery_from_state(state)
    next_stage(state)
    db_upsert_state(state)

    reply_text = await generate_reply_text(customer_text, state, analysis)
    if reply_text:
        await send_facebook_text(sender, reply_text)

    # Send matched product image if relevant and available.
    if state.get("matched_image_url") and (attachment_url or analysis.get("user_intent") in {"ask_price", "ask_size", "ask_both", "buy", "availability"}):
        await send_facebook_image(sender, state["matched_image_url"])

    # Completed order handoff
    if is_complete_order(state) and not int(state.get("order_sent") or 0):
        state["order_stage"] = "confirmed"
        db_upsert_state(state)
        if ORDER_WEBHOOK_URL:
            await post_order_to_webhook(state)
        state["order_sent"] = 1
        db_upsert_state(state)

@app.get("/")
async def root():
    return {"status": "ok", "catalog_items": len(CATALOG)}

@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(challenge or "")
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    print("Incoming webhook:", json.dumps(body)[:2000])

    for entry in body.get("entry", []):
        for event in entry.get("messaging", []):
            try:
                await process_event(event)
            except Exception as e:
                print("process_event error:", repr(e))

    return JSONResponse({"status": "ok"})

ensure_db()
print(f"Loaded {len(CATALOG)} catalog items")