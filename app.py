"""
Baby Catalog Bot V5 — Main FastAPI Application.

This is the slim orchestrator that ties together all modules:
- config.py   — environment vars & constants
- db.py       — SQLite state management
- catalog.py  — product catalog & search
- gemini_ai.py — Gemini AI prompts & generation
- messenger.py — Facebook Messenger send/receive
- orders.py   — order export & delivery logic
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from config import (
    VERIFY_TOKEN,
    AI_MESSAGE_METADATA,
    RATE_LIMIT_SECONDS,
    logger,
)
from db import init_db, load_state, save_state, default_state
from catalog import (
    CATALOG,
    normalize_text,
    score_catalog,
    get_random_catalog_products,
    wants_pictures,
    parse_phone_from_text,
    likely_address,
    infer_delivery_zone,
)
from gemini_ai import (
    SYSTEM_PROMPT,
    MEMORY_UPDATE_PROMPT,
    gemini_generate_text_safe,
    gemini_generate_json,
    analyze_image,
    gemini_client,
)
from messenger import (
    send_facebook_text,
    send_facebook_image,
    send_random_product_pictures,
    send_typing_indicator,
    download_image_from_attachment,
    extract_message_content,
    maybe_get_profile_name,
)
from orders import (
    compute_delivery_facts,
    order_complete,
    export_completed_order,
)


# =========================================================
# App
# =========================================================

app = FastAPI(title="Baby Catalog Bot V5")


# =========================================================
# Utility helpers
# =========================================================

def addressing_prefix(state: Dict[str, Any]) -> str:
    style = (state.get("addressing_style") or "apu").lower()
    name = (state.get("profile_name") or "").strip()

    if style == "vai":
        return f"{name}," if name else "ভাই,"
    if style == "neutral":
        return f"{name}," if name else ""
    return f"{name}," if name else "আপু,"


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


# =========================================================
# Deterministic text parsing
# =========================================================

def deterministic_updates_from_text(state: Dict[str, Any], text: str) -> None:
    """Fast, regex-based extraction — runs before any LLM calls."""
    t = text or ""

    # Phone
    if parse_phone_from_text(t):
        state["phone_pending"] = parse_phone_from_text(t)

    # Addressing style
    low = normalize_text(t)
    if any(x in low for x in ["i am male", "not female", "ami male", "ami chele", "apu na", "ভাই বলেন", "আমি ছেলে"]):
        state["addressing_style"] = "vai"
    elif any(x in low for x in ["neutral"]):
        state["addressing_style"] = "neutral"

    # Address
    if likely_address(t):
        state["address_pending"] = t.strip()

    # Delivery zone override
    if any(x in low for x in ["inside dhaka", "dhakar vitore", "ঢাকার ভিতরে"]):
        state["delivery_zone_final"] = "inside_dhaka"
        state["delivery_zone_source"] = "customer_corrected"
    elif any(x in low for x in ["outside dhaka", "dhakar baire", "ঢাকার বাইরে"]):
        state["delivery_zone_final"] = "outside_dhaka"
        state["delivery_zone_source"] = "customer_corrected"

    # Order intent
    if any(x in low for x in ["order korte", "purchase korbo", "order korbo", "অর্ডার করতে", "অর্ডার করবো", "নিতে চাই"]):
        if state.get("size_finalized"):
            state["order_stage"] = "awaiting_address"
        else:
            state["order_stage"] = "awaiting_size"


# =========================================================
# Confirmation flow
# =========================================================

def handle_confirmation_response(state: Dict[str, Any], text: str) -> Optional[str]:
    """
    If we're awaiting confirmation for address or phone, check if customer confirmed.
    Returns a reply string if handled, None if not in confirmation mode.
    """
    awaiting = state.get("awaiting_confirmation", "")
    if not awaiting:
        return None

    low = normalize_text(text)
    prefix = addressing_prefix(state)

    # Positive confirmation words
    yes_words = ["হ্যাঁ", "হ্যা", "ha", "yes", "ji", "jee", "ঠিক", "ঠিক আছে", "correct", "right", "ok", "okay", "সঠিক"]
    no_words = ["না", "no", "nah", "wrong", "ভুল", "change", "পরিবর্তন"]

    is_yes = any(w in low for w in yes_words)
    is_no = any(w in low for w in no_words)

    if awaiting == "address":
        if is_yes:
            state["address"] = state.get("address_pending", "")
            state["address_pending"] = ""
            state["awaiting_confirmation"] = ""
            state["delivery_zone_inferred"] = infer_delivery_zone(state["address"])
            if not state.get("delivery_zone_final"):
                state["delivery_zone_final"] = state["delivery_zone_inferred"]
                state["delivery_zone_source"] = "inferred"
            return f"{prefix} ঠিকানা নোট করলাম 🌸 এখন একটা কন্টাক্ট নাম্বার দিন?"
        elif is_no:
            state["address_pending"] = ""
            state["awaiting_confirmation"] = ""
            return f"{prefix} কোন সমস্যা নেই 🌸 সঠিক ঠিকানাটা দিন?"
        else:
            # They might have given a NEW address instead of yes/no
            if likely_address(text):
                state["address_pending"] = text.strip()
                return f"{prefix} আপনার ঠিকানাটা কি এটা: {text.strip()}?"
            # Still waiting
            return f"{prefix} ঠিকানাটা কি ঠিক আছে? হ্যাঁ বা না বলুন 🌸"

    elif awaiting == "phone":
        if is_yes:
            state["phone"] = state.get("phone_pending", "")
            state["phone_pending"] = ""
            state["awaiting_confirmation"] = ""
            return None  # Let the normal flow continue (order might be complete now)
        elif is_no:
            state["phone_pending"] = ""
            state["awaiting_confirmation"] = ""
            return f"{prefix} কোন সমস্যা নেই 🌸 সঠিক নাম্বারটা দিন?"
        else:
            phone = parse_phone_from_text(text)
            if phone:
                state["phone_pending"] = phone
                return f"{prefix} এই নাম্বারটা ঠিক আছে: {phone}?"
            return f"{prefix} নাম্বারটা কি ঠিক আছে? হ্যাঁ বা না বলুন 🌸"

    return None


# =========================================================
# State update helpers
# =========================================================

def summarize_state_for_memory(state: Dict[str, Any], latest_user_text: str) -> str:
    fields = []
    if state.get("matched_product_name"):
        fields.append(f"product: {state['matched_product_name']}")
    cart = state.get("cart") or []
    if cart:
        fields.append(f"cart items: {len(cart)}")
    if state.get("recommended_size"):
        fields.append(f"size: {state['recommended_size']}")
    if state.get("address"):
        fields.append("address collected")
    if state.get("phone"):
        fields.append("phone collected")
    if state.get("delivery_eta"):
        fields.append(f"eta: {state['delivery_eta']}")
    key_facts = state.get("key_facts", {})
    if key_facts.get("is_gift"):
        fields.append("gifting")
    if key_facts.get("preferred_product_type"):
        fields.append(f"prefers: {key_facts['preferred_product_type']}")
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
    if phone and not state.get("phone"):
        # Don't overwrite confirmed phone, go through confirmation
        state["phone_pending"] = phone

    address = (updates.get("address") or "").strip()
    if address and not state.get("address"):
        # Don't overwrite confirmed address, go through confirmation
        state["address_pending"] = address

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

    # Key facts update (permanent structured memory)
    kf_update = updates.get("key_facts_update") or {}
    if kf_update and isinstance(kf_update, dict):
        key_facts = state.get("key_facts", {})
        if not isinstance(key_facts, dict):
            key_facts = {}

        ppt = (kf_update.get("preferred_product_type") or "").strip()
        if ppt:
            key_facts["preferred_product_type"] = ppt

        age = kf_update.get("baby_age_months")
        if age is not None and isinstance(age, (int, float)) and age > 0:
            key_facts["baby_age_months"] = int(age)

        if kf_update.get("is_gift") is True:
            key_facts["is_gift"] = True

        colors = kf_update.get("preferred_colors")
        if colors and isinstance(colors, list):
            existing = key_facts.get("preferred_colors", [])
            combined = list(set(existing + colors))
            key_facts["preferred_colors"] = combined[:10]

        budget = (kf_update.get("budget_range") or "").strip()
        if budget:
            key_facts["budget_range"] = budget

        state["key_facts"] = key_facts


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


# =========================================================
# Build context for Gemini
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
    cart = state.get("cart") or []
    key_facts = state.get("key_facts") or {}
    awaiting_confirmation = state.get("awaiting_confirmation") or ""

    cart_summary = ""
    if cart:
        cart_lines = []
        for i, item in enumerate(cart, 1):
            cart_lines.append(f"  {i}. {item.get('product_name', '')} — {item.get('price_text', '')}")
        cart_summary = "\n".join(cart_lines)

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
- address_pending: {state.get("address_pending", "")}
- phone: {state.get("phone", "")}
- phone_pending: {state.get("phone_pending", "")}
- awaiting_confirmation: {awaiting_confirmation}
- delivery_zone_final: {state.get("delivery_zone_final", "")}
- delivery_zone_inferred: {state.get("delivery_zone_inferred", "")}
- delivery_eta: {delivery_eta}
- delivery_charge: {delivery_charge}
- free_delivery: {free_delivery}
- order_stage: {order_stage}

Cart ({len(cart)} items):
{cart_summary or "(empty)"}

Key facts (permanent memory):
- preferred_product_type: {key_facts.get("preferred_product_type", "")}
- baby_age_months: {key_facts.get("baby_age_months", "")}
- is_gift: {key_facts.get("is_gift", False)}
- preferred_colors: {key_facts.get("preferred_colors", [])}
- budget_range: {key_facts.get("budget_range", "")}

Latest image analysis:
- intent_hint: {vision.get("intent_hint", "")}
- product_query_text: {vision.get("product_query_text", "")}
- baby_age_hint_months: {vision.get("baby_age_hint_months", "")}
- recommended_size: {vision.get("recommended_size", "")}
- confidence: {vision.get("confidence", "")}
- notes: {vision.get("notes", "")}

Write the next customer-facing reply only.
""".strip()


# =========================================================
# Reply generation
# =========================================================

async def generate_final_reply(state: Dict[str, Any], latest_user_text: str, vision: Dict[str, Any]) -> str:
    # Hard override: if customer wants to order and size is still unknown
    low = normalize_text(latest_user_text)
    if (
        any(x in low for x in ["order korte", "purchase korbo", "order korbo", "অর্ডার", "নিতে চাই"])
        and not state.get("recommended_size")
    ):
        prefix = addressing_prefix(state)
        return f"{prefix} জ্বি, আমি অর্ডারটা এগিয়ে নিতে পারি 🌸 বেবুর বয়সটা বলবেন, আমি সঠিক সাইজটা কনফার্ম করে দিচ্ছি।".strip()

    prompt_input = build_backend_facts(state, latest_user_text, vision)
    style = (state.get("addressing_style") or "apu").lower()
    raw = gemini_generate_text_safe(SYSTEM_PROMPT, prompt_input, addressing_style=style)
    return clean_customer_reply(raw)


# =========================================================
# Main event processor
# =========================================================

async def process_event(event: Dict[str, Any]) -> None:
    content = extract_message_content(event)
    psid = content["psid"]
    if not psid:
        return

    # -------------------------------------------------------
    # Echo handling
    # -------------------------------------------------------
    if content["is_echo"]:
        echo_text = content.get("text", "").strip()

        # Human takeover toggle commands
        if content["metadata"] != AI_MESSAGE_METADATA:
            if echo_text == "/ai_resume":
                # Operator wants to re-enable AI for this customer
                recipient_id = event.get("recipient", {}).get("id", "")
                if recipient_id:
                    state = load_state(recipient_id)
                    state["human_takeover"] = False
                    state["notes"] = "AI resumed by operator"
                    save_state(recipient_id, state)
                    logger.info("[%s] AI resumed by operator", recipient_id)
                return

            # Normal human message — lock AI
            recipient_id = event.get("recipient", {}).get("id", "")
            if recipient_id:
                state = load_state(recipient_id)
                state["human_takeover"] = True
                state["notes"] = "Human operator has taken over"
                save_state(recipient_id, state)
                logger.info("[%s] Human takeover activated", recipient_id)
        return

    # -------------------------------------------------------
    # Load customer state
    # -------------------------------------------------------
    state = load_state(psid)

    if state.get("human_takeover"):
        logger.info("[%s] Skipping — human takeover active", psid)
        return

    # -------------------------------------------------------
    # Rate limiting
    # -------------------------------------------------------
    now = time.time()
    last_ts = state.get("last_message_timestamp", 0)
    if now - last_ts < RATE_LIMIT_SECONDS:
        logger.info("[%s] Rate limited — %.1fs since last message", psid, now - last_ts)
        return
    state["last_message_timestamp"] = now

    # -------------------------------------------------------
    # Profile name
    # -------------------------------------------------------
    if not state.get("profile_name"):
        state["profile_name"] = maybe_get_profile_name(psid)

    text = content["text"]
    image_url = content["image_url"]

    state["message_count"] = int(state.get("message_count", 0)) + 1
    state["last_user_message"] = text

    # -------------------------------------------------------
    # First-message product hook (attention grabber)
    # -------------------------------------------------------
    if state["message_count"] == 1:
        prefix = addressing_prefix(state)
        hook_products = get_random_catalog_products(count=1)
        if hook_products:
            product = hook_products[0]
            product_name = (product.get("product_name") or "").strip()
            price_text = (product.get("price_text") or "").strip()
            image_url_hook = (product.get("primary_image_url") or "").strip()

            # Send just the image
            if image_url_hook:
                await send_typing_indicator(psid)
                try:
                    await send_facebook_image(psid, image_url_hook)
                except Exception as e:
                    logger.warning("[%s] Hook image send failed: %s", psid, repr(e))

            # One simple line
            await send_facebook_text(
                psid,
                f"{prefix} আসসালামু আলাইকুম 🌸 এটা আমাদের বেস্ট সেলার! এটা নিবেন নাকি অন্য কিছু দেখবেন?"
            )

            # Save state and return
            state["user_intent"] = "first_contact"
            state["matched_product_name"] = product_name
            state["matched_price_text"] = price_text
            state["matched_price_bdt"] = product.get("sale_price_bdt")
            state["matched_image_url"] = image_url_hook
            state["matched_product_page_url"] = (product.get("product_page_url") or "").strip()
            state["conversation_summary"] = summarize_state_for_memory(state, text)
            save_state(psid, state)
            return

    # -------------------------------------------------------
    # Deterministic text parsing
    # -------------------------------------------------------
    deterministic_updates_from_text(state, text)

    # -------------------------------------------------------
    # Confirmation flow (address / phone)
    # -------------------------------------------------------
    confirmation_reply = handle_confirmation_response(state, text)
    if confirmation_reply is not None:
        state["conversation_summary"] = summarize_state_for_memory(state, text)
        compute_delivery_facts(state)
        save_state(psid, state)
        await send_typing_indicator(psid)
        await send_facebook_text(psid, confirmation_reply)

        # Check if order just completed after confirmation
        if order_complete(state):
            state["order_stage"] = "confirmed"
            save_state(psid, state)
            await export_completed_order(state)
        return

    # -------------------------------------------------------
    # Trigger pending confirmations
    # -------------------------------------------------------
    if state.get("address_pending") and not state.get("address") and not state.get("awaiting_confirmation"):
        pending = state["address_pending"]
        state["awaiting_confirmation"] = "address"
        state["conversation_summary"] = summarize_state_for_memory(state, text)
        save_state(psid, state)
        prefix = addressing_prefix(state)
        await send_typing_indicator(psid)
        await send_facebook_text(psid, f"{prefix} আপনার ঠিকানাটা কি এটা: {pending}?")
        return

    if state.get("phone_pending") and not state.get("phone") and not state.get("awaiting_confirmation"):
        pending = state["phone_pending"]
        state["awaiting_confirmation"] = "phone"
        state["conversation_summary"] = summarize_state_for_memory(state, text)
        save_state(psid, state)
        prefix = addressing_prefix(state)
        await send_typing_indicator(psid)
        await send_facebook_text(psid, f"{prefix} এই নাম্বারটা ঠিক আছে: {pending}?")
        return

    # -------------------------------------------------------
    # Picture request
    # -------------------------------------------------------
    if wants_pictures(text):
        state["user_intent"] = "show_pictures"
        state["conversation_summary"] = summarize_state_for_memory(state, text)
        save_state(psid, state)
        products = get_random_catalog_products(count=3)
        await send_typing_indicator(psid)
        await send_random_product_pictures(psid, products)
        return

    # -------------------------------------------------------
    # LLM memory extraction
    # -------------------------------------------------------
    if gemini_client and text:
        updates_input = f"""
Current state:
{json.dumps(state, ensure_ascii=False)}

Latest user message:
{text}
""".strip()
        llm_updates = gemini_generate_json(MEMORY_UPDATE_PROMPT, updates_input)
        merge_memory_updates(state, llm_updates)

    # -------------------------------------------------------
    # Vision (image analysis)
    # -------------------------------------------------------
    vision: Dict[str, Any] = {}
    if image_url:
        image_bytes, mime = await download_image_from_attachment(image_url)
        if image_bytes:
            vision = analyze_image(image_bytes, mime, text)
            apply_vision_to_state(state, vision)

    # -------------------------------------------------------
    # Check pending confirmations after LLM updates
    # -------------------------------------------------------
    if state.get("address_pending") and not state.get("address") and not state.get("awaiting_confirmation"):
        pending = state["address_pending"]
        state["awaiting_confirmation"] = "address"
        state["conversation_summary"] = summarize_state_for_memory(state, text)
        save_state(psid, state)
        prefix = addressing_prefix(state)
        await send_typing_indicator(psid)
        await send_facebook_text(psid, f"{prefix} আপনার ঠিকানাটা কি এটা: {pending}?")
        return

    if state.get("phone_pending") and not state.get("phone") and not state.get("awaiting_confirmation"):
        pending = state["phone_pending"]
        state["awaiting_confirmation"] = "phone"
        state["conversation_summary"] = summarize_state_for_memory(state, text)
        save_state(psid, state)
        prefix = addressing_prefix(state)
        await send_typing_indicator(psid)
        await send_facebook_text(psid, f"{prefix} এই নাম্বারটা ঠিক আছে: {pending}?")
        return

    # -------------------------------------------------------
    # Delivery facts
    # ------ -------------------------------------------------
    if not state.get("delivery_zone_final") and state.get("delivery_zone_inferred"):
        state["delivery_zone_final"] = state["delivery_zone_inferred"]
        state["delivery_zone_source"] = "inferred"
    compute_delivery_facts(state)

    # -------------------------------------------------------
    # Progress order stages
    # -------------------------------------------------------
    if state.get("recommended_size") and not state.get("address"):
        state["order_stage"] = "awaiting_address" if state["message_count"] >= 1 and any(
            x in normalize_text(text) for x in ["order", "purchase", "অর্ডার", "নিতে চাই"]
        ) else state.get("order_stage", "browsing")
    if state.get("recommended_size") and state.get("address") and not state.get("phone"):
        state["order_stage"] = "awaiting_phone"
    if order_complete(state):
        state["order_stage"] = "confirmed"

    # -------------------------------------------------------
    # Generate final reply
    # -------------------------------------------------------
    await send_typing_indicator(psid)
    reply = await generate_final_reply(state, text, vision)
    state["last_bot_reply"] = reply
    state["conversation_summary"] = summarize_state_for_memory(state, text)
    save_state(psid, state)

    await send_facebook_text(psid, reply)

    # -------------------------------------------------------
    # Send matched product image when relevant
    # -------------------------------------------------------
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
                logger.warning("[%s] Image send failed: %s", psid, repr(e))

    # -------------------------------------------------------
    # Order export
    # -------------------------------------------------------
    if order_complete(state):
        await export_completed_order(state)


# =========================================================
# Webhook event parser
# =========================================================

def get_first_message_event(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        return body["entry"][0]["messaging"][0]
    except Exception:
        return None


# =========================================================
# FastAPI routes
# =========================================================

@app.on_event("startup")
async def startup_event() -> None:
    init_db()
    logger.info("Loaded %d catalog items", len(CATALOG))


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"ok": True, "service": "baby-catalog-bot-v5"}


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
