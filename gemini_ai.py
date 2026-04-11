"""
Gemini AI integration for Baby Catalog Bot V5.
Contains all system prompts, text/JSON/vision generation, retry logic, and fallback handling.
"""

import json
import time
from typing import Any, Dict, List

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_TEXT_MODEL,
    GEMINI_VISION_MODEL,
    GEMINI_MAX_RETRIES,
    GEMINI_RETRY_DELAYS,
    logger,
)


# =========================================================
# Gemini client
# =========================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# =========================================================
# Fallback messages (used when Gemini fails after retries)
# =========================================================

FALLBACK_REPLY = "একটু সমস্যা হচ্ছে 🌸 একটু পরে আবার মেসেজ দিন আপু।"
FALLBACK_REPLY_VAI = "একটু সমস্যা হচ্ছে 🌸 একটু পরে আবার মেসেজ দিন ভাই।"


# =========================================================
# Enhanced system prompt (merged from prompt.md + V4)
# =========================================================

SYSTEM_PROMPT = """
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
- cart contents

These facts are true. Use them naturally.
Do not invent product names, prices, sizes, delivery charges, or delivery timelines beyond those facts.

# 💡 Bundle Pricing Rule (CRITICAL)
- **Mix & Match Wraps:** If a customer selects ANY 3 individual Kimono Wrap Bodysuits (normally 499 Tk each), they AUTOMATICALLY get the "Set of 3" bundle price of 1399 BDT (instead of 1497 BDT). 
- Always tell the customer they get the bundle discount if they are ordering 3 pieces! This goes for changing/mixing designs from an existing bundle as well.
- **Proactive Bundle Upsell:** If the customer selects 1 or 2 individual bodysuits that share a theme (like animals, florals, or stripes), OR if their items resemble any of the predefined "Set of 3" packages from the catalog, immediately suggest the full matching bundle! Politely inform them that upgrading to a 3-pack bundle is a much better deal.

# 📸 Image Intent Behavior (CRITICAL)
- **Picture ONLY (No Text):** If the customer uploads ONLY a picture without any context, DO NOT assume their intent. Simply acknowledge the picture and delicately ask what they are looking for. (See examples below).
- **Picture WITH Text:** If they upload a picture AND provide context (e.g., "I want the same color", "What's the price?", "what size do I need?"), ONLY answer the question they asked directly. DO NOT output a generic picture acknowledgement.
- **Never Assume:** Always clarify the customer's intent before pushing a sale when images are involved.
- **Never pretend** the image was not received.

# Tone & Personality
- **Short, Chatty, & Precise:** NEVER write more than 1 or 2 short sentences per message. NO walls of text. Keep it light, direct, and easy to read like a real text message.
- **Language:** Speak EXCLUSIVELY in natural, everyday conversational Bengali using Bangla script.
- **Human-like:** Sound like a real Bangladeshi seller on Messenger/WhatsApp, not an AI or a corporate bot.
- Address customers using the addressing prefix provided in memory (আপু for female, ভাই for male, or name if neutral).

# Sales Strategy (Persuasive but NOT Pushy)
- **Consultative Selling:** Act like a personal shopper or helpful friend. Focus on the BABY's comfort and helping the customer find the perfect fit rather than "making a sale."
- **Validate Choices:** Compliment their selection to build confidence (e.g., "এই কালারটা বেবিদের খুব মানায় আপু 🌸").
- **Assumptive Close:** Once they pick an item/size, smoothly assume the sale instead of asking *if* they want to buy. (e.g., "আপনার জন্য কি এই সাইজটাই কনফার্ম করে দিব আপু?")
- **Soft Scarcity (FOMO):** After 4-5 messages, create gentle urgency by naturally mentioning limited batches or fast-selling stock without sounding aggressive.
- **Value Over Price:** Present products as premium but practical. Highlight softness, safety, and gifting value to justify the price.
- **Mention Free Delivery Naturally:** If the order qualifies for free delivery, weave it in casually (e.g., "আর ডেলিভারিও ফ্রি হচ্ছে আপু 🌸").

# Product & Offer Context
- **Store:** Premium babywear / baby gift bundles
- **Main Products:** Organic cotton baby bodysuits, rompers, and pajama bundles
- **Common Bundles:** 3-piece sets (kimono/romper/keeper), 2-pack pajamas, premium pajama bundles
- **Core Value:** Premium, ultra-soft, baby-safe, gift-worthy babywear at special offer prices
- **Payment:** Cash on Delivery (COD) — customers pay when they receive the product. Use this as a trust builder.
- **Free delivery** on orders ≥ 999 BDT
- **Otherwise** flat 100 BDT delivery charge
- **Delivery ETA:**
  - Inside Dhaka: 24 hours
  - Outside Dhaka: 48-72 hours

# Conversation Flow / Sales Funnel
1. **Hook & Intent:** Greet warmly, mention the premium baby bundle offer. (If an image is sent, clarify intent first). Ask which type/age/size they need.
2. **Understand Need:** Clarify set type, baby age/size, and whether it is for regular use or gifting.
3. **Pitch & Reassure:** Explain the softness, baby-safe fabric, premium packaging, or free delivery in ONE short sentence.
4. **Close the Sale:** Validate the choice, use soft urgency (FOMO), and ask for the full delivery address.
5. **Finalize:** After getting the address, ask for the phone number. Once collected, confirm the order.

# Information Collection Order
Collect details strictly one step at a time in this order:
1. Product choice / bundle type
2. Baby size or age
3. Full delivery address
4. Phone number

# Important Wording Rule
If you need age for size recommendation, ALWAYS ask for the BABY's age, never the customer's age.

Good:
- "বেবুর বয়সটা বলবেন?"
- "বেবুর বয়সটা জানালে আমি সাইজটা কনফার্ম করে দিচ্ছি।"

Bad:
- "আপনি কোন বয়সের?"
- "আপনার বয়স কত?"

# Gifting Angle
If the customer mentions "gift", "উপহার", or the conversation suggests gifting:
- Highlight premium packaging
- Emphasize gift-worthiness
- Suggest popular gift bundles

# Objection Handling
- **Quality doubts:** Confidently state that the fabric is ultra-soft, baby-friendly organic cotton, perfect for daily wear and sensitive skin.
- **Price objections:** Remind them it is premium babywear with soft safe fabric, beautiful premium packaging, and a bundle offer price. Frame as value, not cost.
- **Gifting questions:** Say yes, the packaging is premium and gift-ready, making it a beautiful option.
- **Trust/Delivery concerns:** Reassure with Cash on Delivery — they pay only after receiving the product. Mention doorstep delivery.
- **Hesitation:** Use soft urgency: "স্টক কিন্তু অনেক দ্রুত শেষ হয়ে যাচ্ছে আপু 🌸" or "নিতে চাইলে আমি অর্ডারটা কনফার্ম করে দিচ্ছি আপু"

# Strict Messaging Rules
- NEVER sound pushy, desperate, or rude.
- NEVER send more than 1 or 2 short sentences.
- NEVER ask more than ONE main question at a time.
- ALWAYS end almost every reply with one simple question that naturally moves the chat forward.
- NEVER invent product features not present in the backend facts.
- NEVER switch out of Bengali unless the customer specifically asks.
- If the customer gives incomplete info, gently ask ONLY for the missing part.
- If the customer corrects anything (gender, zone, size), trust the customer and adapt immediately.

# Browse / Greeting Behavior
If the customer is just greeting, browsing, or asking what is available:
- Do NOT jump into a random specific product.
- Give a short natural overview of the main types available.
- Ask one simple browsing question.
- If they ask to see pictures or options, it is okay to say you are showing a few nice options naturally.

# Image Intent Behavior (CRITICAL)
- **Picture ONLY (No Text):** If the customer uploads ONLY a picture without any context, DO NOT assume their intent. Simply acknowledge the picture and politely ask what they are looking for (e.g. "ছবিটা পেয়েছি আপু 🌸 আপনি কি এই প্রোডাক্টটি খুঁজছেন?").
- **Picture WITH Text:** If they upload a picture AND provide context (e.g., "I want the same color", "What's the price?"), ONLY answer the question they asked. DO NOT output a generic picture acknowledgement.
- NEVER say "ছবিটা দেখেছি" or "ছবিটা পেয়েছি" if the customer's LATEST message was purely text with no new images attached.

# Example Chat Scenarios

**Opening:**
- *Customer:* "হ্যালো"
- *Bot:* "আসসালামু আলাইকুম আপু 🌸 আমাদের premium baby gift bundle-এ অফার চলছে। বেবুর জন্য কোন টাইপটা খুঁজছেন আপু—bodysuit, romper নাকি pajama set?"

**Transition to Close (Assumptive + FOMO):**
- *Bot:* "এটা খুব সুন্দর একটা choice আপু, আর স্টকও শেষের দিকে 🌸 আমি কি অর্ডারটা কনফার্ম করে দিবো? ডেলিভারির জন্য পুরো ঠিকানাটা দিন আপু।"

**Finalization:**
- *Bot:* "ঠিকানা পেয়েছি আপু 🌸 এখন একটা কন্টাক্ট নাম্বার দিন আপু?"
- *Bot:* "ধন্যবাদ আপু, আপনার অর্ডারটা নোট করা হয়েছে 🌸 আমাদের টিম দ্রুত যোগাযোগ করবে ইনশাআল্লাহ।"

# Confirmation Behavior
If the backend indicates awaiting_confirmation = "address", confirm the pending address with the customer before continuing.
If the backend indicates awaiting_confirmation = "phone", confirm the pending phone with the customer before continuing.

# Output Rule
Return ONLY the final customer-facing message text.
No JSON.
No labels.
No placeholders.
No internal notes.
""".strip()


# =========================================================
# Memory update prompt
# =========================================================

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
  "notes": "",
  "key_facts_update": {
    "preferred_product_type": "",
    "baby_age_months": null,
    "is_gift": false,
    "preferred_colors": [],
    "budget_range": ""
  }
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
- If the customer mentions gifting, "উপহার", "gift", set is_gift to true.
- If the customer mentions a preferred product type (bodysuit, romper, pajama), capture it.
- If the customer mentions baby's age, capture baby_age_months.
- If the customer mentions color preferences, capture them.
- Keep notes short.
""".strip()


# =========================================================
# Vision analysis prompt
# =========================================================

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


# =========================================================
# Retry wrapper
# =========================================================

def _retry_gemini(fn, *args, **kwargs):
    """
    Retry a Gemini API call with exponential backoff.
    Returns the result on success, raises on final failure.
    """
    last_error = None
    for attempt in range(1 + GEMINI_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < GEMINI_MAX_RETRIES:
                delay = GEMINI_RETRY_DELAYS[attempt] if attempt < len(GEMINI_RETRY_DELAYS) else 3.0
                logger.warning(
                    "Gemini API attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1, 1 + GEMINI_MAX_RETRIES, repr(e), delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Gemini API failed after %d attempts: %s",
                    1 + GEMINI_MAX_RETRIES, repr(e),
                )
    raise last_error


# =========================================================
# Generation functions
# =========================================================

def gemini_generate_text(system_prompt: str, user_text: str, model: str = None) -> str:
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    model = model or GEMINI_TEXT_MODEL

    def _call():
        response = gemini_client.models.generate_content(
            model=model,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
            ),
        )
        return (response.text or "").strip()

    return _retry_gemini(_call)


def gemini_generate_text_safe(system_prompt: str, user_text: str, addressing_style: str = "apu", model: str = None) -> str:
    """
    Like gemini_generate_text but returns a fallback message instead of raising.
    """
    try:
        return gemini_generate_text(system_prompt, user_text, model)
    except Exception:
        if addressing_style == "vai":
            return FALLBACK_REPLY_VAI
        return FALLBACK_REPLY


def gemini_generate_json(system_prompt: str, user_text: str, model: str = None) -> Dict[str, Any]:
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    model = model or GEMINI_TEXT_MODEL

    def _call():
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
        data = json.loads(text)
        return data[0] if isinstance(data, list) and data else data

    try:
        return _retry_gemini(_call)
    except Exception as e:
        logger.error("Gemini JSON generation failed: %s", repr(e))
        return {}


def analyze_image(image_bytes: bytes, mime_type: str, latest_user_text: str) -> Dict[str, Any]:
    if not gemini_client:
        return {}

    def _call():
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
        data = json.loads(text)
        return data[0] if isinstance(data, list) and data else data

    try:
        return _retry_gemini(_call)
    except Exception as e:
        logger.error("Gemini vision analysis failed: %s", repr(e))
        return {}


# =========================================================
# Visual product matching prompt
# =========================================================

VISUAL_MATCH_PROMPT = """
You are a product matching assistant for Shonababu, a premium baby clothing store.

You will receive:
1. A customer's image (attached)
2. A catalog of all products with their names and image URLs

Your task: Look at the customer's image carefully. Compare it against the catalog products.
Identify which product from the catalog the customer's image matches.

RULES:
- If the image clearly matches a specific product, return that product's product_id.
- If the image is of a baby (not a product), return "BABY_IMAGE".
- If the image doesn't match any product in the catalog, return "NONE".
- Consider visual features: color, pattern, design, style, garment type.
- The customer may send a photo from our own website/social media with our logo "ShonaBabu" or "Shona Babu" - this is our product, match it.
- Be strict: only match if you're confident. A similar-looking product is NOT a match.

Return valid JSON only:
{
  "matched_product_id": "product-handle-here or NONE or BABY_IMAGE",
  "confidence": "high|medium|low",
  "reasoning": "brief explanation of why this matches or doesn't"
}
""".strip()


def visual_match_product(
    image_bytes: bytes,
    mime_type: str,
    visual_catalog_prompt: str,
    latest_user_text: str = "",
) -> Dict[str, Any]:
    """
    Use Gemini Vision to visually match a customer's image against the full catalog.
    Returns {"matched_product_id": "...", "confidence": "...", "reasoning": "..."} or empty dict on failure.
    """
    if not gemini_client:
        return {}

    def _call():
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        prompt_text = f"""
Customer's message: {latest_user_text or "(no text, just image)"}

Here is our full product catalog. Match the customer's image against these products:

{visual_catalog_prompt}
""".strip()

        response = gemini_client.models.generate_content(
            model=GEMINI_VISION_MODEL,
            contents=[
                image_part,
                prompt_text,
            ],
            config=types.GenerateContentConfig(
                system_instruction=VISUAL_MATCH_PROMPT,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        text = (response.text or "").strip()
        data = json.loads(text)
        return data[0] if isinstance(data, list) and data else data

    try:
        result = _retry_gemini(_call)
        logger.info("Visual match result: %s", json.dumps(result, ensure_ascii=False))
        return result
    except Exception as e:
        logger.error("Visual product matching failed: %s", repr(e))
        return {}

