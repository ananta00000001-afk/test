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
import asyncio
from collections import deque
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
    VISUAL_CATALOG_PROMPT,
    normalize_text,
    score_catalog,
    find_product_by_id,
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
    visual_match_product,
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

# Message deduplication: track last 200 processed message IDs
_processed_mids: set = set()
_processed_mids_order: deque = deque(maxlen=200)


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
            
            if not state.get("phone") and not state.get("phone_pending"):
                return f"{prefix} আমি আপনার ঠিকানা হিসেবে নোট করেছি 🌸 এখন শেষ ধাপে একটা কন্টাক্ট নাম্বার দিন?"
            return None
        elif is_no:
            state["address_pending"] = ""
            state["awaiting_confirmation"] = ""
            return f"{prefix} কোন সমস্যা নেই 🌸 সঠিক ঠিকানাটা দিন?"
        else:
            # They might have given a NEW address instead of yes/no
            if likely_address(text):
                state["address_pending"] = text.strip()
                return f"{prefix} আমি আপনার ঠিকানা হিসেবে নোট করেছি: {text.strip()}।"
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
                return f"{prefix} আপনার কন্টাক্ট নাম্বার নোট করেছি: {phone}।"
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


def apply_visual_match_to_state(state: Dict[str, Any], match_result: Dict[str, Any]) -> bool:
    """
    Apply Gemini visual match result to state.
    Returns True if a product was matched, False otherwise.
    """
    if not match_result:
        return False

    product_id = (match_result.get("matched_product_id") or "").strip()
    confidence = (match_result.get("confidence") or "low").strip().lower()

    # Skip non-matches
    if not product_id or product_id in ("NONE", "BABY_IMAGE"):
        logger.info("Visual match returned: %s (confidence: %s)", product_id, confidence)
        return False

    # Only accept medium or high confidence matches
    if confidence == "low":
        logger.info("Visual match '%s' rejected (low confidence)", product_id)
        return False

    # Look up the matched product in catalog
    product = find_product_by_id(product_id)
    if not product:
        logger.warning("Visual match returned unknown product_id: %s", product_id)
        return False

    # Apply match to state
    state["matched_product_name"] = product.get("product_name", "")
    state["matched_price_text"] = product.get("price_text", "")
    state["matched_price_bdt"] = product.get("sale_price_bdt")
    state["matched_image_url"] = product.get("primary_image_url", "")
    state["matched_product_page_url"] = product.get("product_page_url", "")
    logger.info("Visual match: %s -> %s (confidence: %s)", product_id, product.get("product_name", ""), confidence)
    return True


def apply_vision_to_state(state: Dict[str, Any], vision: Dict[str, Any]) -> None:
    """
    Apply secondary vision analysis (baby size/age hints) to state.
    Product matching is now done by visual_match_product, not here.
    """
    if not vision:
        return
    if vision.get("recommended_size") and not state.get("size_finalized"):
        state["recommended_size"] = (vision.get("recommended_size") or "").strip()
        state["size_confidence"] = (vision.get("confidence") or "").strip()


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
- no_catalog_match: {vision.get("no_catalog_match", False)}

IMPORTANT: If no_catalog_match is True, it means the customer sent an image of a product that does NOT exist in our catalog. Do NOT pretend we have it. Acknowledge the image, honestly say we don't have this exact product, and gently offer to show similar options we do have.

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
    # Message deduplication (prevents double processing)
    # -------------------------------------------------------
    mid = event.get("message", {}).get("mid", "")
    if mid and mid in _processed_mids:
        logger.info("[%s] Skipping duplicate message mid=%s", psid, mid)
        return
    if mid:
        _processed_mids.add(mid)
        _processed_mids_order.append(mid)
        # Evict oldest if over capacity
        if len(_processed_mids) > 200:
            oldest = _processed_mids_order[0]
            _processed_mids.discard(oldest)

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

    text = content.get("text", "")
    image_url = content.get("image_url", "")
    image_urls = content.get("image_urls") or []
    if not image_urls and image_url:
        image_urls = [image_url]

    state["message_count"] = int(state.get("message_count", 0)) + 1
    state["last_user_message"] = text

    # -------------------------------------------------------
    # First-message product hook (attention grabber)
    # -------------------------------------------------------
    if state["message_count"] == 1 and not image_urls:
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
    if state.get("address_pending") and not state.get("address"):
        pending = state.pop("address_pending", "")
        state["address"] = pending
        state["delivery_zone_inferred"] = infer_delivery_zone(pending)
        if not state.get("delivery_zone_final"):
            state["delivery_zone_final"] = state["delivery_zone_inferred"]
            state["delivery_zone_source"] = "inferred"
        state["conversation_summary"] = summarize_state_for_memory(state, text)
        save_state(psid, state)

    if state.get("phone_pending") and not state.get("phone"):
        pending = state.pop("phone_pending", "")
        state["phone"] = pending
        state["conversation_summary"] = summarize_state_for_memory(state, text)
        save_state(psid, state)

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
    # Vision: Visual product matching + size analysis
    # -------------------------------------------------------
    vision: Dict[str, Any] = {}
    vision_matched = False
    vision_matched_any = False

    if image_urls:
        if "cart" not in state:
            state["cart"] = []

        for idx, img_url in enumerate(image_urls):
            image_bytes, mime = await download_image_from_attachment(img_url)
            if not image_bytes:
                continue

            # Step 1: Visual matching — Gemini sees customer image + catalog
            try:
                match_result = visual_match_product(
                    image_bytes, mime, VISUAL_CATALOG_PROMPT, text
                )
                vision_matched = apply_visual_match_to_state(state, match_result)
            except Exception as e:
                logger.error("[%s] Visual match failed for img %d: %s", psid, idx, repr(e))
                match_result = {}
                vision_matched = False

            # Step 2: Secondary analysis (only for first image to save time/tokens)
            if idx == 0:
                vision = analyze_image(image_bytes, mime, text)
                apply_vision_to_state(state, vision)

            # Step 3: FALLBACK — if visual match failed, try token-based search (on first image only)
            if not vision_matched and idx == 0 and vision.get("product_query_text"):
                query = vision["product_query_text"]
                candidates = score_catalog(query, CATALOG, top_k=3, min_score=4)
                if candidates:
                    best = candidates[0]
                    state["matched_product_name"] = best.get("product_name", "")
                    state["matched_price_text"] = best.get("price_text", "")
                    state["matched_price_bdt"] = best.get("sale_price_bdt")
                    state["matched_image_url"] = best.get("primary_image_url", "")
                    state["matched_product_page_url"] = best.get("product_page_url", "")
                    vision_matched = True
                    logger.info("[%s] Fallback text match: '%s' -> %s", psid, query, best.get("product_name", ""))

            # Cart addition
            if vision_matched:
                vision_matched_any = True
                p_name = state.get("matched_product_name")
                if p_name:
                    existing = next((x for x in state["cart"] if x.get("product_name") == p_name), None)
                    if existing:
                        existing["quantity"] = existing.get("quantity", 1) + 1
                    else:
                        state["cart"].append({
                            "product_name": p_name,
                            "price_text": state.get("matched_price_text", ""),
                            "price_bdt": state.get("matched_price_bdt", 0),
                            "image_url": state.get("matched_image_url", ""),
                            "quantity": 1
                        })

            # Flag for Gemini context (on first image)
            if idx == 0:
                matched_id = (match_result.get("matched_product_id") or "NONE").strip()
                if matched_id == "BABY_IMAGE":
                    vision["is_baby_image"] = True
                elif not vision_matched:
                    vision["no_catalog_match"] = True
                    vision["match_reasoning"] = match_result.get("reasoning", "")
                    
        vision_matched = vision_matched_any
    else:
        # Text-only message — keep existing matched product if any
        vision_matched = bool(state.get("matched_product_name") or state.get("cart"))
    # -------------------------------------------------------
    # Check pending confirmations after LLM updates
    # -------------------------------------------------------
    if state.get("address_pending") and not state.get("address"):
        pending = state.pop("address_pending", "")
        state["address"] = pending
        state["delivery_zone_inferred"] = infer_delivery_zone(pending)
        if not state.get("delivery_zone_final"):
            state["delivery_zone_final"] = state["delivery_zone_inferred"]
            state["delivery_zone_source"] = "inferred"
        state["conversation_summary"] = summarize_state_for_memory(state, text)
        save_state(psid, state)

    if state.get("phone_pending") and not state.get("phone"):
        pending = state.pop("phone_pending", "")
        state["phone"] = pending
        state["conversation_summary"] = summarize_state_for_memory(state, text)
        save_state(psid, state)

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
    # Setup Cart Images extraction
    # -------------------------------------------------------
    cart_images = []
    if state.get("cart"):
        cart_images = [item["image_url"] for item in state["cart"] if item.get("image_url")]
    elif state.get("matched_image_url"):
        cart_images = [state["matched_image_url"]]

    # -------------------------------------------------------
    # Send order receipt if newly confirmed
    # -------------------------------------------------------
    if order_complete(state) and not state.get("receipt_sent"):
        size = state.get("recommended_size", "-")
        phone = state.get("phone", "-")
        address = state.get("address", "-")
        
        # Build price text from cart if available, else matched product
        if state.get("cart"):
            price = sum((item.get("price_bdt") or 0) * item.get("quantity", 1) for item in state["cart"])
            
            # --- Mix & Match Bundle Logic ---
            # Every 3 individual bodysuits get the 1399 Tk bundle price (499*3=1497, so 98 Tk discount per 3)
            # Only count items that are individually ~499 Tk to avoid discounting an already discounted Set
            single_bodysuit_count = sum(
                item.get("quantity", 1) 
                for item in state["cart"] 
                if "bodysuit" in item.get("product_name", "").lower() and (item.get("price_bdt") or 0) < 600
            )
            bundle_discounts = (single_bodysuit_count // 3) * 98
            price -= bundle_discounts
            # --------------------------------

            price_str = f"{price} BDT"
            # Add delivery charge text context if free delivery is known
            if price >= FREE_DELIVERY_THRESHOLD: 
                price_str += " (Free Delivery)"
            elif price_str != "0 BDT":
                price_str += " + 100 TK Delivery"
        else:
            price_str = state.get("matched_price_text", "-")
            
        receipt_text = f"অর্ডার সামারি:\nSize: {size}\nPhone: {phone}\nAddress: {address}\nPrice: {price_str}"
        await send_facebook_text(psid, receipt_text)
        
        for img_url in cart_images:
            try:
                await send_facebook_image(psid, img_url)
            except Exception as e:
                logger.warning("[%s] Receipt image send failed: %s", psid, repr(e))
                
        state["receipt_sent"] = True
        save_state(psid, state)

    # -------------------------------------------------------
    # Send matched product image when relevant AND confident
    # -------------------------------------------------------
    if vision_matched and cart_images:
        low = normalize_text(text)
        should_send_image = any(
            x in low for x in ["price", "dam", "দাম", "eta", "এটা", "same", "same color", "design", "nibo", "ta"]
        )
        # Also send if customer sent an image and we found a confident match
        if image_urls or should_send_image:
            for img_url in cart_images:
                try:
                    await send_facebook_image(psid, img_url)
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


_user_queues: Dict[str, List[Dict[str, Any]]] = {}
_user_tasks: Dict[str, asyncio.Task] = {}

async def process_user_queue(psid: str):
    # Buffer time to catch all concurrent image webhooks
    await asyncio.sleep(1.5)
    
    events = _user_queues.pop(psid, [])
    _user_tasks.pop(psid, None)
    
    if not events:
        return

    # Serve the MEGA payload
    merged_event = dict(events[-1])
    if "message" not in merged_event:
        merged_event["message"] = {}
        
    merged_text = []
    merged_images = []
    
    for ev in events:
        content = extract_message_content(ev)
        if content.get("text"):
            merged_text.append(content["text"])
        if content.get("image_urls"):
            merged_images.extend(content["image_urls"])

    # Reconstruct combined payload
    merged_event["message"]["text"] = " | ".join(merged_text)
    
    attachments = []
    for img_url in set(merged_images):
        attachments.append({"type": "image", "payload": {"url": img_url}})
    merged_event["message"]["attachments"] = attachments
    
    try:
        await process_event(merged_event)
    except Exception as e:
        logger.error("[%s] Error processing aggregated event: %s", psid, repr(e))


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    event = get_first_message_event(body)
    if not event:
        return JSONResponse({"status": "ignored"})
        
    psid = event.get("sender", {}).get("id")
    if not psid:
        return JSONResponse({"status": "ignored"})
        
    if psid not in _user_queues:
        _user_queues[psid] = []
    _user_queues[psid].append(event)
    
    if psid in _user_tasks and not _user_tasks[psid].done():
        _user_tasks[psid].cancel()
        
    _user_tasks[psid] = asyncio.create_task(process_user_queue(psid))
    
    return JSONResponse({"status": "ok"})
