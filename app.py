import csv
import io
import json
import os
import re
import random
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from google import genai
from google.genai import types

# =========================================================
# Environment
# =========================================================

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "").strip()
FACEBOOK_PAGE_ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN", "").strip()
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.1-pro-preview").strip()
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", GEMINI_TEXT_MODEL).strip()
CATALOG_CSV_PATH = os.getenv("CATALOG_CSV_PATH", "./catalog.csv").strip()
SQLITE_PATH = os.getenv("SQLITE_PATH", "./state.db").strip()
ORDER_WEBHOOK_URL = os.getenv("ORDER_WEBHOOK_URL", "").strip()

AI_MESSAGE_METADATA = "ai_bot_v4"

FREE_DELIVERY_THRESHOLD = 999
FLAT_DELIVERY_CHARGE = 100
DHAKA_ETA = "24 hours"
OUTSIDE_DHAKA_ETA = "48-72 hours"

# =========================================================
# App + clients
# =========================================================

app = FastAPI(title="Baby Catalog Bot v4")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# =========================================================
# SQLite
# =========================================================

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_state (
                psid TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def load_state(psid: str) -> Dict[str, Any]:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT state_json FROM customer_state WHERE psid = ?",
            (psid,),
        ).fetchone()
        if not row:
            return default_state(psid)
        data = json.loads(row["state_json"])
        return merge_state(default_state(psid), data)
    finally:
        conn.close()


def save_state(psid: str, state: Dict[str, Any]) -> None:
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO customer_state (psid, state_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(psid) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (psid, json.dumps(state, ensure_ascii=False), time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def default_state(psid: str) -> Dict[str, Any]:
    return {
        "psid": psid,
        "profile_name": "",
        "addressing_style": "apu",  # apu | vai | neutral
        "message_count": 0,
        "human_takeover": False,
        "last_user_message": "",
        "last_bot_reply": "",
        "conversation_summary": "",
        "user_intent": "",
        "matched_product_name": "",
        "matched_price_text": "",
        "matched_price_bdt": None,
        "matched_image_url": "",
        "matched_product_page_url": "",
        "recommended_size": "",
        "size_finalized": False,
        "size_confidence": "",
        "phone": "",
        "address": "",
        "delivery_zone_inferred": "",
        "delivery_zone_final": "",
        "delivery_zone_source": "",
        "delivery_eta": "",
        "delivery_charge": FLAT_DELIVERY_CHARGE,
        "free_delivery": False,
        "order_stage": "browsing",
        "order_exported": False,
        "notes": "",
    }


def merge_state(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in incoming.items():
        merged[k] = v
    return merged


# =========================================================
# Catalog
# =========================================================

def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\u0980-\u09ff\s&-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def build_search_text(row: Dict[str, Any]) -> str:
    parts = [
        row.get("product_name", ""),
        row.get("price_text", ""),
        row.get("primary_image_url", ""),
        row.get("product_page_url", ""),
    ]
    return normalize_text(" ".join(parts))


def load_catalog() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.exists(CATALOG_CSV_PATH):
        return items
    with open(CATALOG_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {k: (v or "").strip() for k, v in row.items()}
            clean_row["sale_price_bdt"] = extract_price_bdt(clean_row.get("price_text", ""))
            clean_row["search_text"] = build_search_text(clean_row)
            items.append(clean_row)
    return items


CATALOG = load_catalog()


def score_catalog(query_text: str, catalog: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    q = normalize_text(query_text)
    if not q:
        return catalog[:top_k]

    query_tokens = set(q.split())
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for row in catalog:
        score = 0
        search_text = row.get("search_text", "")
        name = normalize_text(row.get("product_name", ""))
        for token in query_tokens:
            if token in search_text:
                score += 2
            if token in name:
                score += 3
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_k]] if scored else catalog[:top_k]

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


def wants_pictures(text: str) -> bool:
    t = normalize_text(text)
    triggers = [
        "picture",
        "pictures",
        "pic",
        "pics",
        "photo",
        "photos",
        "chobi",
        "chobi dekhan",
        "picture dekhan",
        "pics dekhan",
        "pic dekhan",
        "ছবি",
        "ছবি দেখান",
        "কিছু ছবি দেখান",
        "পিকচার",
        "পিকচার দেখান",
        "picture dekhan",
        "show picture",
        "show me pictures",
        "kichu dekhan",
    ]
    return any(x in t for x in triggers)



# =========================================================
# Utility
# =========================================================

def addressing_prefix(state: Dict[str, Any]) -> str:
    style = (state.get("addressing_style") or "apu").lower()
    name = (state.get("profile_name") or "").strip()

    if style == "vai":
        return f"{name}," if name else "ভাই,"
    if style == "neutral":
        return f"{name}," if name else ""
    return f"{name}," if name else "আপু,"


def safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)


def maybe_get_profile_name(psid: str) -> str:
    """
    Best-effort only. If unavailable, fail quietly.
    """
    if not FACEBOOK_PAGE_ACCESS_TOKEN:
        return ""
    url = f"https://graph.facebook.com/v25.0/{psid}"
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(
                url,
                params={
                    "fields": "first_name,last_name,name",
                    "access_token": FACEBOOK_PAGE_ACCESS_TOKEN,
                },
            )
            if r.status_code != 200:
                return ""
            data = r.json()
            return (data.get("first_name") or data.get("name") or "").strip()
    except Exception:
        return ""


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


def compute_delivery_facts(state: Dict[str, Any]) -> None:
    price = state.get("matched_price_bdt")
    free_delivery = bool(price is not None and price >= FREE_DELIVERY_THRESHOLD)

    state["free_delivery"] = free_delivery
    state["delivery_charge"] = 0 if free_delivery else FLAT_DELIVERY_CHARGE

    zone = state.get("delivery_zone_final") or state.get("delivery_zone_inferred") or ""
    if zone == "inside_dhaka":
        state["delivery_eta"] = DHAKA_ETA
    elif zone == "outside_dhaka":
        state["delivery_eta"] = OUTSIDE_DHAKA_ETA
    else:
        state["delivery_eta"] = ""


def order_complete(state: Dict[str, Any]) -> bool:
    return bool(
        state.get("recommended_size")
        and state.get("phone")
        and state.get("address")
    )


def clean_customer_reply(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "জ্বি, আমি সাহায্য করতে পারি 🌸 একটু বলবেন কীভাবে এগোতে চান?"
    bad_prefixes = [
        "delivery_eta:",
        "delivery_charge:",
        "free_delivery:",
        "matched_product_name:",
        "matched_price:",
        "recommended_size:",
        "order_stage:",
    ]
    keep: List[str] = []
    for line in text.splitlines():
        l = line.strip()
        low = l.lower()
        if not l:
            continue
        if low in {"(empty)", "empty", "* ask", "ask"}:
            continue
        if "2 short sentences" in low:
            continue
        if any(low.startswith(p) for p in bad_prefixes):
            continue
        keep.append(l)
    cleaned = re.sub(r"\s+", " ", " ".join(keep)).strip()
    return cleaned or "জ্বি, আমি সাহায্য করতে পারি 🌸 একটু বলবেন কীভাবে এগোতে চান?"


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
        "road no", "sector", "ঢাকা", "জেলা", "থানা", "রোড", "বাড়ি", "বাসা", "ফ্ল্যাট",
        "গ্রাম", "উপজেলা", "পোস্ট", "area", "mirpur", "uttara", "dhaka", "cumilla",
        "chattogram", "sylhet", "khulna", "rajshahi", "barishal", "rangpur", "mymensingh",
    ]
    return any(m in t for m in markers)


# =========================================================
# Gemini prompts
# =========================================================

V4_SYSTEM_PROMPT = """
# Role & Objective
You are a highly engaging, friendly, chatty, and precise sales assistant for a premium baby clothing store in Bangladesh.

Your goal is to chat naturally with customers, understand their intent from the latest message plus the memory provided, help them choose the right babywear bundle, and collect order details step-by-step to close the sale without sounding robotic or pushy.

# Core Source-of-Truth Rule
The backend may provide exact facts such as:
- matched product name
- price
- size suggestion
- delivery charge
- free delivery eligibility
- delivery ETA
- customer memory/state

These facts are true. Use them naturally.
Do not invent product names, prices, sizes, delivery charges, or delivery timelines beyond those facts.

# Image Intent Behavior (Inspired by prior prompt)
- If the customer sent ONLY a picture without context, do not assume their intent and do not pitch aggressively.
- Simply acknowledge the image and ask one short clarifying question.
- If they sent a picture WITH context, answer that exact context first before pushing the sales flow.
- Never pretend the image was not received.

# Tone & Style
- Speak only in natural conversational Bengali using Bangla script.
- Sound like a real Bangladeshi Messenger seller.
- Keep replies warm, helpful, and human.
- Never sound corporate, robotic, or like a form.
- Usually keep replies to 1 or 2 short sentences, but clarity matters more than extreme shortness.

# Conversation Priorities
1. Understand the exact user intent from the latest message and memory.
2. If the customer is browsing, respond naturally and guide gently.
3. If they want to order, smoothly collect only the next missing piece of information.
4. If price/product/size facts are already known from backend, use them naturally.
5. If the customer corrects anything, trust the customer and adapt immediately.
6. If the customer says they are male / not female / prefer another form of address, remember it and address them accordingly.
7. If the customer corrects Dhaka vs outside Dhaka, trust the correction.

# Product Context
- Premium babywear / baby gift bundles
- Main products: bodysuits, rompers, pajama sets
- Soft, baby-safe, premium feel
- Free delivery applies on orders 999 BDT and above
- Otherwise delivery charge is 100 BDT flat
- Delivery ETA:
  - Inside Dhaka: 24 hours
  - Outside Dhaka: 48-72 hours

# Sales Approach
- Be consultative, not salesy.
- Validate choices naturally.
- After 4 or 5 chats, you may softly move them toward the order once.
- Mention free delivery naturally if it applies.
- Mention delivery ETA naturally when helpful.

# Collection Order
Collect information one step at a time in this order:
1. Product choice (if still unclear)
2. Size or baby age (if still unclear)
3. Full address
4. Phone number

# Important Wording Rule
If you need age for size recommendation, ALWAYS ask for the BABY'S age, never the customer's age.

Good:
- "বেবুর বয়সটা বলবেন?"
- "বেবুর বয়সটা জানালে আমি সাইজটা কনফার্ম করে দিচ্ছি।"

Bad:
- "আপনি কোন বয়সের?"
- "আপনার বয়স কত?"

# Browse / Greeting Behavior
If the customer is just greeting, browsing, or asking what is available, do not jump into a random specific product.
Give a short natural overview of the main types available, then ask one simple browsing question.
If the customer asks to see pictures or options, it is okay to say you are showing a few nice options naturally.

# Output Rule
Return ONLY the final customer-facing message text.
No JSON.
No labels.
No placeholders.
No internal notes.
""".strip()


MEMORY_UPDATE_PROMPT = """
You extract structured customer memory updates from a Bangla sales chat.

Return valid JSON only with these keys:
{
  "user_intent": "",
  "addressing_style": "", 
  "phone": "",
  "address": "",
  "delivery_zone_override": "",
  "recommended_size": "",
  "size_finalized": false,
  "order_stage": "",
  "notes": ""
}

Rules:
- If the customer says they are male / not female / says "ভাই বলেন" or similar, set addressing_style to "vai".
- If the customer prefers neutral, set addressing_style to "neutral".
- If nothing changes, return empty strings and false.
- If the customer clearly provides a phone number, extract it.
- If the customer clearly provides a full address, extract it.
- If the customer explicitly says inside Dhaka, set delivery_zone_override to "inside_dhaka".
- If the customer explicitly says outside Dhaka, set delivery_zone_override to "outside_dhaka".
- If they clearly confirm a size, set recommended_size and size_finalized = true.
- Keep notes short.
""".strip()


VISION_ANALYSIS_PROMPT = """
Analyze the attached customer image plus any text context for a Bangla babywear sales assistant.

Return valid JSON only:
{
  "intent_hint": "",
  "product_query_text": "",
  "baby_age_hint_months": 0,
  "recommended_size": "",
  "confidence": "low|medium|high",
  "notes": ""
}

Rules:
- If the image looks like a baby and the user is asking about size, estimate the likely baby age in months if possible.
- If you can suggest a baby clothing size, do it.
- If the image looks like a product, produce a short product_query_text that helps search the catalog.
- Be practical, not overcautious.
""".strip()


def gemini_generate_text(system_prompt: str, user_text: str, model: str) -> str:
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    response = gemini_client.models.generate_content(
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
        ),
    )
    return (response.text or "").strip()


def gemini_generate_json(system_prompt: str, user_text: str, model: str) -> Dict[str, Any]:
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    response = gemini_client.models.generate_content(
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    text = (response.text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        return {}


def analyze_image(image_bytes: bytes, mime_type: str, latest_user_text: str) -> Dict[str, Any]:
    if not gemini_client:
        return {}

    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    response = gemini_client.models.generate_content(
        model=GEMINI_VISION_MODEL,
        contents=[
            part,
            f"Latest user text: {latest_user_text or ''}",
        ],
        config=types.GenerateContentConfig(
            system_instruction=VISION_ANALYSIS_PROMPT,
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    text = (response.text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        return {}


# =========================================================
# Facebook send
# =========================================================

async def send_facebook_text(psid: str, text: str) -> None:
    url = f"https://graph.facebook.com/v25.0/{FACEBOOK_PAGE_ID}/messages"
    payload = {
        "recipient": {"id": psid},
        "messaging_type": "RESPONSE",
        "message": {
            "text": text,
            "metadata": AI_MESSAGE_METADATA,
        },
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            params={"access_token": FACEBOOK_PAGE_ACCESS_TOKEN},
            json=payload,
        )
        print("Facebook text status:", response.status_code)
        print("Facebook text body:", response.text)
        response.raise_for_status()


async def send_facebook_image(psid: str, image_url: str) -> None:
    if not image_url:
        return
    url = f"https://graph.facebook.com/v25.0/{FACEBOOK_PAGE_ID}/messages"
    payload = {
        "recipient": {"id": psid},
        "messaging_type": "RESPONSE",
        "message": {
            "attachment": {
                "type": "image",
                "payload": {
                    "url": image_url,
                    "is_reusable": False,
                },
            },
            "metadata": AI_MESSAGE_METADATA,
        },
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            params={"access_token": FACEBOOK_PAGE_ACCESS_TOKEN},
            json=payload,
        )
        print("Facebook image status:", response.status_code)
        print("Facebook image body:", response.text)
        response.raise_for_status()

async def send_random_product_pictures(psid: str, count: int = 3) -> None:
    products = get_random_catalog_products(count)
    if not products:
        await send_facebook_text(psid, "জ্বি, এই মুহূর্তে ছবি দেখাতে একটু সমস্যা হচ্ছে আপু 🌸")
        return

    await send_facebook_text(psid, "জ্বি, কিছু সুন্দর অপশন দেখাচ্ছি 🌸")

    for product in products:
        name = (product.get("product_name") or "").strip()
        price = (product.get("matched_price_text") or product.get("price_text") or "").strip()
        image_url = (product.get("primary_image_url") or "").strip()

        if name or price:
            if name and price:
                caption = f"{name}\n{price}"
            else:
                caption = name or price
            try:
                await send_facebook_text(psid, caption)
            except Exception as e:
                print("Random image caption send failed:", repr(e))

        if image_url:
            try:
                await send_facebook_image(psid, image_url)
            except Exception as e:
                print("Random image send failed:", repr(e))



# =========================================================
# Order export
# =========================================================

async def export_completed_order(state: Dict[str, Any]) -> None:
    if not ORDER_WEBHOOK_URL or state.get("order_exported") or not order_complete(state):
        return

    payload = {
        "timestamp": int(time.time()),
        "customer_name": state.get("profile_name", ""),
        "facebook_psid": state.get("psid", ""),
        "addressing_style": state.get("addressing_style", ""),
        "product_name": state.get("matched_product_name", ""),
        "product_price": state.get("matched_price_text", ""),
        "product_image_url": state.get("matched_image_url", ""),
        "recommended_size": state.get("recommended_size", ""),
        "phone": state.get("phone", ""),
        "address": state.get("address", ""),
        "delivery_zone": state.get("delivery_zone_final") or state.get("delivery_zone_inferred") or "",
        "delivery_eta": state.get("delivery_eta", ""),
        "delivery_charge": state.get("delivery_charge", ""),
        "free_delivery": state.get("free_delivery", False),
        "order_stage": state.get("order_stage", ""),
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(ORDER_WEBHOOK_URL, json=payload)
            print("Order export status:", r.status_code)
            print("Order export body:", r.text)
            r.raise_for_status()
        state["order_exported"] = True
        save_state(state["psid"], state)
    except Exception as e:
        print("Order export failed:", repr(e))


# =========================================================
# Messenger event parsing
# =========================================================

def get_first_message_event(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        return body["entry"][0]["messaging"][0]
    except Exception:
        return None


async def download_image_from_attachment(url: str) -> Tuple[Optional[bytes], str]:
    if not url:
        return None, "image/jpeg"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                url,
                params={"access_token": FACEBOOK_PAGE_ACCESS_TOKEN},
                follow_redirects=True,
            )
            if r.status_code != 200:
                # Retry without token.
                r = await client.get(url, follow_redirects=True)
            if r.status_code != 200:
                return None, "image/jpeg"
            mime = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
            return r.content, mime
    except Exception:
        return None, "image/jpeg"


def extract_message_content(event: Dict[str, Any]) -> Dict[str, Any]:
    sender_id = event.get("sender", {}).get("id", "")
    message = event.get("message", {}) or {}
    text = (message.get("text") or "").strip()

    attachments = message.get("attachments") or []
    image_url = ""
    if attachments:
        for att in attachments:
            if att.get("type") == "image":
                image_url = ((att.get("payload") or {}).get("url") or "").strip()
                if image_url:
                    break

    return {
        "psid": sender_id,
        "text": text,
        "image_url": image_url,
        "is_echo": safe_bool(message.get("is_echo")),
        "metadata": (message.get("metadata") or "").strip(),
    }


# =========================================================
# Conversation engine
# =========================================================

def build_backend_facts(state: Dict[str, Any], latest_user_text: str, vision: Dict[str, Any]) -> str:
    product_name = state.get("matched_product_name") or ""
    price_text = state.get("matched_price_text") or ""
    rec_size = state.get("recommended_size") or ""
    delivery_eta = state.get("delivery_eta") or ""
    delivery_charge = state.get("delivery_charge")
    free_delivery = state.get("free_delivery", False)
    order_stage = state.get("order_stage") or ""
    prefix = addressing_prefix(state)

    return f"""
Latest user message:
{latest_user_text or ""}

Conversation memory:
- profile_name: {state.get("profile_name", "")}
- addressing_prefix: {prefix}
- message_count: {state.get("message_count", 0)}
- human_takeover: {state.get("human_takeover", False)}
- conversation_summary: {state.get("conversation_summary", "")}
- user_intent: {state.get("user_intent", "")}
- matched_product_name: {product_name}
- matched_price_text: {price_text}
- matched_image_url: {state.get("matched_image_url", "")}
- recommended_size: {rec_size}
- size_finalized: {state.get("size_finalized", False)}
- size_confidence: {state.get("size_confidence", "")}
- address: {state.get("address", "")}
- phone: {state.get("phone", "")}
- delivery_zone_final: {state.get("delivery_zone_final", "")}
- delivery_zone_inferred: {state.get("delivery_zone_inferred", "")}
- delivery_eta: {delivery_eta}
- delivery_charge: {delivery_charge}
- free_delivery: {free_delivery}
- order_stage: {order_stage}

Latest image analysis:
- intent_hint: {vision.get("intent_hint", "")}
- product_query_text: {vision.get("product_query_text", "")}
- baby_age_hint_months: {vision.get("baby_age_hint_months", "")}
- recommended_size: {vision.get("recommended_size", "")}
- confidence: {vision.get("confidence", "")}
- notes: {vision.get("notes", "")}

Write the next customer-facing reply only.
""".strip()


def summarize_state_for_memory(state: Dict[str, Any], latest_user_text: str) -> str:
    fields = []
    if state.get("matched_product_name"):
        fields.append(f"product: {state['matched_product_name']}")
    if state.get("recommended_size"):
        fields.append(f"size: {state['recommended_size']}")
    if state.get("address"):
        fields.append("address collected")
    if state.get("phone"):
        fields.append("phone collected")
    if state.get("delivery_eta"):
        fields.append(f"eta: {state['delivery_eta']}")
    tail = ", ".join(fields)
    base = state.get("conversation_summary", "")
    extra = f"Latest user said: {latest_user_text}."
    if tail:
        extra += f" Known facts: {tail}."
    combined = (base + " " + extra).strip()
    return combined[-1000:]


def merge_memory_updates(state: Dict[str, Any], updates: Dict[str, Any]) -> None:
    if not updates:
        return

    user_intent = (updates.get("user_intent") or "").strip()
    if user_intent:
        state["user_intent"] = user_intent

    style = (updates.get("addressing_style") or "").strip().lower()
    if style in {"apu", "vai", "neutral"}:
        state["addressing_style"] = style

    phone = (updates.get("phone") or "").strip()
    if phone:
        state["phone"] = phone

    address = (updates.get("address") or "").strip()
    if address:
        state["address"] = address
        state["delivery_zone_inferred"] = infer_delivery_zone(address)

    override = (updates.get("delivery_zone_override") or "").strip().lower()
    if override in {"inside_dhaka", "outside_dhaka"}:
        state["delivery_zone_final"] = override
        state["delivery_zone_source"] = "customer_corrected"

    rec_size = (updates.get("recommended_size") or "").strip()
    if rec_size:
        state["recommended_size"] = rec_size

    if "size_finalized" in updates and isinstance(updates.get("size_finalized"), bool):
        state["size_finalized"] = updates["size_finalized"]

    order_stage = (updates.get("order_stage") or "").strip()
    if order_stage:
        state["order_stage"] = order_stage

    notes = (updates.get("notes") or "").strip()
    if notes:
        state["notes"] = notes[-500:]


def apply_vision_to_state(state: Dict[str, Any], vision: Dict[str, Any]) -> None:
    if not vision:
        return

    if vision.get("recommended_size") and not state.get("size_finalized"):
        state["recommended_size"] = (vision.get("recommended_size") or "").strip()
        state["size_confidence"] = (vision.get("confidence") or "").strip()

    query_text = (vision.get("product_query_text") or "").strip()
    if query_text and CATALOG:
        candidates = score_catalog(query_text, CATALOG, top_k=3)
        if candidates:
            best = candidates[0]
            state["matched_product_name"] = best.get("product_name", "")
            state["matched_price_text"] = best.get("price_text", "")
            state["matched_price_bdt"] = best.get("sale_price_bdt")
            state["matched_image_url"] = best.get("primary_image_url", "")
            state["matched_product_page_url"] = best.get("product_page_url", "")


def deterministic_updates_from_text(state: Dict[str, Any], text: str) -> None:
    t = text or ""

    if parse_phone_from_text(t):
        state["phone"] = parse_phone_from_text(t)

    low = normalize_text(t)
    if any(x in low for x in ["i am male", "not female", "ami male", "ami chele", "apu na", "ভাই বলেন", "আমি ছেলে"]):
        state["addressing_style"] = "vai"
    elif any(x in low for x in ["neutral"]):
        state["addressing_style"] = "neutral"

    if likely_address(t):
        state["address"] = t.strip()
        state["delivery_zone_inferred"] = infer_delivery_zone(t)

    if any(x in low for x in ["inside dhaka", "dhakar vitore", "ঢাকার ভিতরে"]):
        state["delivery_zone_final"] = "inside_dhaka"
        state["delivery_zone_source"] = "customer_corrected"
    elif any(x in low for x in ["outside dhaka", "dhakar baire", "ঢাকার বাইরে"]):
        state["delivery_zone_final"] = "outside_dhaka"
        state["delivery_zone_source"] = "customer_corrected"

    if any(x in low for x in ["order korte", "purchase korbo", "order korbo", "অর্ডার করতে", "অর্ডার করবো", "নিতে চাই"]):
        if state.get("size_finalized"):
            state["order_stage"] = "awaiting_address"
        else:
            state["order_stage"] = "awaiting_size"


async def generate_final_reply(state: Dict[str, Any], latest_user_text: str, vision: Dict[str, Any]) -> str:
    # Important clarity override: if customer wants to order and size is still unknown,
    # ask explicitly for BABY age in Bengali, warm but clear.
    low = normalize_text(latest_user_text)
    if (
        any(x in low for x in ["order korte", "purchase korbo", "order korbo", "অর্ডার", "নিতে চাই"])
        and not state.get("recommended_size")
    ):
        prefix = addressing_prefix(state)
        return f"{prefix} জ্বি, আমি অর্ডারটা এগিয়ে নিতে পারি 🌸 বেবুর বয়সটা বলবেন, আমি সঠিক সাইজটা কনফার্ম করে দিচ্ছি।".strip()

    prompt_input = build_backend_facts(state, latest_user_text, vision)
    raw = gemini_generate_text(V4_SYSTEM_PROMPT, prompt_input, GEMINI_TEXT_MODEL)
    return clean_customer_reply(raw)


# =========================================================
# Main processor
# =========================================================

async def process_event(event: Dict[str, Any]) -> None:
    content = extract_message_content(event)
    psid = content["psid"]
    if not psid:
        return

    # Echo handling: if a human operator replies manually, lock AI for this customer.
    if content["is_echo"]:
        if content["metadata"] != AI_MESSAGE_METADATA:
            state = load_state(psid)
            state["human_takeover"] = True
            state["notes"] = "Human operator has taken over"
            save_state(psid, state)
        return

    state = load_state(psid)

    if state.get("human_takeover"):
        return

    if not state.get("profile_name"):
        state["profile_name"] = maybe_get_profile_name(psid)

    text = content["text"]
    image_url = content["image_url"]

    state["message_count"] = int(state.get("message_count", 0)) + 1
    state["last_user_message"] = text

    deterministic_updates_from_text(state, text)

    if wants_pictures(text):
        state["user_intent"] = "show_pictures"
        state["conversation_summary"] = summarize_state_for_memory(state, text)
        save_state(psid, state)
        await send_random_product_pictures(psid, count=3)
        return


    # Best-effort structured memory extraction from LLM.
    updates_input = f"""
Current state:
{json.dumps(state, ensure_ascii=False)}

Latest user message:
{text}
""".strip()
    llm_updates = gemini_generate_json(MEMORY_UPDATE_PROMPT, updates_input, GEMINI_TEXT_MODEL) if gemini_client and text else {}
    merge_memory_updates(state, llm_updates)

    vision: Dict[str, Any] = {}
    if image_url:
        image_bytes, mime = await download_image_from_attachment(image_url)
        if image_bytes:
            vision = analyze_image(image_bytes, mime, text)
            apply_vision_to_state(state, vision)

    # Delivery facts from address or customer correction.
    if not state.get("delivery_zone_final") and state.get("delivery_zone_inferred"):
        state["delivery_zone_final"] = state["delivery_zone_inferred"]
        state["delivery_zone_source"] = "inferred"
    compute_delivery_facts(state)

    # Progress order stages based on known fields.
    if state.get("recommended_size") and not state.get("address"):
        state["order_stage"] = "awaiting_address" if state["message_count"] >= 1 and any(
            x in normalize_text(text) for x in ["order", "purchase", "অর্ডার", "নিতে চাই"]
        ) else state.get("order_stage", "browsing")
    if state.get("recommended_size") and state.get("address") and not state.get("phone"):
        state["order_stage"] = "awaiting_phone"
    if order_complete(state):
        state["order_stage"] = "confirmed"

    # Final reply
    reply = await generate_final_reply(state, text, vision)
    state["last_bot_reply"] = reply
    state["conversation_summary"] = summarize_state_for_memory(state, text)
    save_state(psid, state)

    await send_facebook_text(psid, reply)

    # Send matched product image when helpful.
    if state.get("matched_image_url") and state.get("matched_product_name"):
        low = normalize_text(text)
        should_send_image = bool(
            image_url
            or any(x in low for x in ["price", "dam", "দাম", "eta", "এটা", "same", "same color", "design"])
        )
        if should_send_image:
            try:
                await send_facebook_image(psid, state["matched_image_url"])
            except Exception as e:
                print("Image send failed:", repr(e))

    if order_complete(state):
        await export_completed_order(state)


# =========================================================
# FastAPI routes
# =========================================================

@app.on_event("startup")
async def startup_event() -> None:
    init_db()
    print(f"Loaded {len(CATALOG)} catalog items")


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"ok": True, "service": "baby-catalog-bot-v4"}


@app.get("/webhook")
async def verify_webhook(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "")
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    event = get_first_message_event(body)
    if not event:
        return JSONResponse({"status": "ignored"})
    background_tasks.add_task(process_event, event)
    return JSONResponse({"status": "ok"})
