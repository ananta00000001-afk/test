import base64
import csv
import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baby_catalog_bot")

BASE_DIR = Path(__file__).resolve().parent
CATALOG_CSV_PATH = Path(os.getenv("CATALOG_CSV_PATH", BASE_DIR / "catalog.csv"))
SQLITE_PATH = Path(os.getenv("SQLITE_PATH", BASE_DIR / "state.db"))

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
PAGE_ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN", "")
PAGE_ID = os.getenv("FACEBOOK_PAGE_ID", "")

# Provider config: prefers Gemini if available, otherwise OpenAI if later added.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")

app = FastAPI(title="Baby Catalog Bot", version="0.2.0")


class AIConfigError(RuntimeError):
    pass


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_state (
                session_id TEXT PRIMARY KEY,
                selected_product_id TEXT,
                selected_product_name TEXT,
                selected_price_bdt INTEGER,
                last_image_summary TEXT,
                last_reply TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def parse_sale_price_bdt(price_text: str) -> Optional[int]:
    if not price_text:
        return None
    sale_match = re.search(r"Sale price\s*Tk\s*([\d,]+(?:\.\d+)?)", price_text, re.IGNORECASE)
    if sale_match:
        return int(float(sale_match.group(1).replace(",", "")))
    prices = re.findall(r"Tk\s*([\d,]+(?:\.\d+)?)", price_text, re.IGNORECASE)
    if prices:
        return int(float(prices[0].replace(",", "")))
    return None


def slug_keywords(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s\-&]+", " ", text.lower())
    parts = re.split(r"[\s\-_/&]+", cleaned)
    stop = {"the", "of", "set", "organic", "cotton", "baby", "premium", "and", "for", "pack"}
    return sorted({p for p in parts if p and p not in stop and len(p) > 2})


def build_catalog_row(row: Dict[str, str]) -> Dict[str, Any]:
    name = row.get("product_name", "")
    price_text = row.get("price_text", "")
    page_url = row.get("product_page_url", "")
    image_url = row.get("primary_image_url", "")
    slug_text = " ".join([name, page_url, image_url])
    keywords = slug_keywords(slug_text)
    return {
        "product_id": row.get("product_id", ""),
        "sku": row.get("sku", ""),
        "product_name": name,
        "price_text": price_text,
        "sale_price_bdt": parse_sale_price_bdt(price_text),
        "primary_image_url": image_url,
        "product_page_url": page_url,
        "keywords": keywords,
    }


def load_catalog() -> List[Dict[str, Any]]:
    if not CATALOG_CSV_PATH.exists():
        raise FileNotFoundError(f"Catalog file not found: {CATALOG_CSV_PATH}")
    with open(CATALOG_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [build_catalog_row(row) for row in rows]


CATALOG = load_catalog()


def get_state(session_id: str) -> Dict[str, Any]:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM customer_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def save_state(session_id: str, updates: Dict[str, Any]) -> None:
    state = get_state(session_id)
    state.update(updates)
    state["session_id"] = session_id
    state["updated_at"] = datetime.utcnow().isoformat()
    conn = get_db()
    try:
        conn.execute(
            """
            INSERT INTO customer_state (
                session_id, selected_product_id, selected_product_name, selected_price_bdt,
                last_image_summary, last_reply, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                selected_product_id = excluded.selected_product_id,
                selected_product_name = excluded.selected_product_name,
                selected_price_bdt = excluded.selected_price_bdt,
                last_image_summary = excluded.last_image_summary,
                last_reply = excluded.last_reply,
                updated_at = excluded.updated_at
            """,
            (
                state.get("session_id"),
                state.get("selected_product_id"),
                state.get("selected_product_name"),
                state.get("selected_price_bdt"),
                state.get("last_image_summary"),
                state.get("last_reply"),
                state.get("updated_at"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def extract_events(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for entry in payload.get("entry", []):
        for messaging in entry.get("messaging", []):
            sender_id = messaging.get("sender", {}).get("id")
            message = messaging.get("message", {}) or {}
            text = message.get("text", "")
            attachments = message.get("attachments", []) or []
            image_url = None
            for attachment in attachments:
                if attachment.get("type") == "image":
                    image_url = attachment.get("payload", {}).get("url")
                    break
            if sender_id:
                events.append(
                    {
                        "sender_id": sender_id,
                        "text": text,
                        "image_url": image_url,
                        "raw_message": message,
                    }
                )
    return events


async def download_image_bytes(image_url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as http:
        response = await http.get(image_url)
        response.raise_for_status()
        return response.content


def extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return JSON: {text[:500]}")
    return json.loads(match.group(0))


async def gemini_generate_text(prompt: str, model: str) -> str:
    if not GEMINI_API_KEY:
        raise AIConfigError("GEMINI_API_KEY missing")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3}}
    async with httpx.AsyncClient(timeout=60.0) as http:
        response = await http.post(url, params={"key": GEMINI_API_KEY}, json=payload)
        response.raise_for_status()
        data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        raise ValueError(f"Unexpected Gemini response: {data}") from e


async def analyze_image_with_vision(image_bytes: bytes) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise AIConfigError("GEMINI_API_KEY missing")

    prompt = (
        'Analyze the uploaded babywear product image and return ONLY valid JSON with keys '\
        'category, dominant_color, print_theme, style, keywords, summary. '\
        'keywords must be an array of short strings. Focus on the product itself, not the baby.'
    )
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_VISION_MODEL}:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"temperature": 0.1},
    }
    async with httpx.AsyncClient(timeout=90.0) as http:
        response = await http.post(url, params={"key": GEMINI_API_KEY}, json=payload)
        response.raise_for_status()
        data = response.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = extract_json_block(text)
    except Exception as e:
        raise ValueError(f"Unexpected Gemini vision response: {data}") from e

    parsed.setdefault("category", "")
    parsed.setdefault("dominant_color", "")
    parsed.setdefault("print_theme", "")
    parsed.setdefault("style", "")
    parsed.setdefault("keywords", [])
    parsed.setdefault("summary", "")
    return parsed


def score_product(product: Dict[str, Any], vision: Dict[str, Any]) -> int:
    searchable = " ".join(
        [
            product.get("product_name", "").lower(),
            product.get("product_page_url", "").lower(),
            product.get("primary_image_url", "").lower(),
            " ".join(product.get("keywords", [])),
        ]
    )
    score = 0
    for kw in vision.get("keywords", []):
        kw = str(kw).lower().strip()
        if kw and kw in searchable:
            score += 3
    for key in ["category", "dominant_color", "print_theme", "style"]:
        value = str(vision.get(key, "")).lower().strip()
        if value and value in searchable:
            score += 4
    return score


def find_top_matches(vision: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    scored = []
    for product in CATALOG:
        scored.append((score_product(product, vision), product))
    scored.sort(key=lambda item: item[0], reverse=True)
    matches = []
    for score, product in scored[:limit]:
        enriched = dict(product)
        enriched["match_score"] = score
        matches.append(enriched)
    return matches


async def craft_price_reply(top_matches: List[Dict[str, Any]], vision: Dict[str, Any]) -> str:
    if not top_matches or top_matches[0].get("match_score", 0) <= 0:
        return "আপু, ছবিটার সাথে একদম নিশ্চিতভাবে মিলাতে পারিনি 💖 প্রোডাক্টের নামটা বা আরেকটা ক্লিয়ার ছবি দিলে আমি দামটা বলে দিচ্ছি আপু।"

    best = top_matches[0]
    price = best.get("sale_price_bdt")
    if not GEMINI_API_KEY:
        return (
            f"আপু, ছবিটার সাথে সবচেয়ে বেশি মিলছে {best['product_name']} 💖 "
            f"এর প্রাইস ৳{price} আপু। নিতে চাইলে আমি অর্ডারটা এগিয়ে দেই?"
        )

    prompt = f"""
You are a warm Bangla-speaking babywear sales assistant.
Write exactly 2 short sentences in Bangla script.
Say the uploaded product image most likely matches this catalog product:
- product_name: {best['product_name']}
- sale_price_bdt: {price}
- summary: {vision.get('summary', '')}
Rules:
- Address the customer as আপু
- Mention the exact catalog price as ৳{price}
- Ask one short forward-moving question in the second sentence
- Do not mention technical issues
"""
    text = await gemini_generate_text(prompt, GEMINI_TEXT_MODEL)
    return text.strip()


def build_text_only_reply(state: Dict[str, Any], text: str) -> str:
    text_l = text.lower().strip()
    if state.get("selected_product_name") and state.get("selected_price_bdt") and ("price" in text_l or "দাম" in text_l):
        return f"আপু, {state['selected_product_name']} এর প্রাইস ৳{state['selected_price_bdt']} 💖 নিতে চাইলে আমি অর্ডারটা এগিয়ে দেই আপু?"
    return "আপু, প্রোডাক্টের একটা ছবি দিলে আমি মিলিয়ে সঠিক দামটা বলে দিতে পারব 💖 চাইলে ছবিটা পাঠান আপু।"


async def send_facebook_text(recipient_id: str, text: str) -> None:
    if not PAGE_ACCESS_TOKEN or not PAGE_ID:
        raise RuntimeError("FACEBOOK_PAGE_ACCESS_TOKEN or FACEBOOK_PAGE_ID missing")

    url = f"https://graph.facebook.com/v25.0/{PAGE_ID}/messages"
    payload = {
        "recipient": {"id": recipient_id},
        "messaging_type": "RESPONSE",
        "message": {"text": text},
    }
    async with httpx.AsyncClient(timeout=30.0) as http:
        response = await http.post(
            url,
            params={"access_token": PAGE_ACCESS_TOKEN},
            json=payload,
        )
        response.raise_for_status()
        logger.info("Sent reply to %s: %s", recipient_id, response.text)


async def process_event(event: Dict[str, Any]) -> None:
    session_id = event["sender_id"]
    state = get_state(session_id)
    reply = ""
    updates: Dict[str, Any] = {}

    if event.get("image_url"):
        image_bytes = await download_image_bytes(event["image_url"])
        vision = await analyze_image_with_vision(image_bytes)
        top_matches = find_top_matches(vision)
        reply = await craft_price_reply(top_matches, vision)
        if top_matches and top_matches[0].get("match_score", 0) > 0:
            best = top_matches[0]
            updates.update(
                {
                    "selected_product_id": best.get("product_id"),
                    "selected_product_name": best.get("product_name"),
                    "selected_price_bdt": best.get("sale_price_bdt"),
                    "last_image_summary": vision.get("summary", ""),
                }
            )
    else:
        reply = build_text_only_reply(state, event.get("text", ""))

    updates["last_reply"] = reply
    save_state(session_id, updates)
    await send_facebook_text(session_id, reply)


@app.on_event("startup")
def startup_event() -> None:
    init_db()
    logger.info("Loaded %d catalog items", len(CATALOG))


@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "catalog_items": len(CATALOG), "provider": "gemini" if GEMINI_API_KEY else "none"}


@app.get("/webhook")
async def verify_webhook(request: Request) -> PlainTextResponse:
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN and challenge:
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    payload = await request.json()
    events = extract_events(payload)
    for event in events:
        background_tasks.add_task(process_event, event)
    return JSONResponse({"status": "ok"})


@app.get("/debug/catalog")
def debug_catalog() -> Dict[str, Any]:
    return {"count": len(CATALOG), "items": CATALOG[:5]}
