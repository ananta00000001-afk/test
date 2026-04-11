"""
Facebook Messenger integration for Baby Catalog Bot V5.
Handles sending text, images, typing indicators, and downloading attachments.
"""

from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import FACEBOOK_PAGE_ACCESS_TOKEN, FACEBOOK_PAGE_ID, AI_MESSAGE_METADATA, logger


# =========================================================
# Typing indicator
# =========================================================

async def send_typing_indicator(psid: str) -> None:
    """Send 'typing_on' action so the customer sees typing dots."""
    if not FACEBOOK_PAGE_ACCESS_TOKEN:
        return
    url = f"https://graph.facebook.com/v25.0/{FACEBOOK_PAGE_ID}/messages"
    payload = {
        "recipient": {"id": psid},
        "sender_action": "typing_on",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                url,
                params={"access_token": FACEBOOK_PAGE_ACCESS_TOKEN},
                json=payload,
            )
    except Exception as e:
        logger.warning("[%s] Typing indicator failed: %s", psid, repr(e))


# =========================================================
# Send text message
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
        logger.info("[%s] Facebook text send — status: %d", psid, response.status_code)
        if response.status_code != 200:
            logger.error("[%s] Facebook text error body: %s", psid, response.text)
        response.raise_for_status()


# =========================================================
# Send image message
# =========================================================

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
        logger.info("[%s] Facebook image send — status: %d", psid, response.status_code)
        if response.status_code != 200:
            logger.error("[%s] Facebook image error body: %s", psid, response.text)
        response.raise_for_status()


# =========================================================
# Send random product pictures (for browsing)
# =========================================================

async def send_random_product_pictures(psid: str, products: List[Dict[str, Any]]) -> None:
    """
    Send a set of product cards (name + price caption + image).
    Products list is passed in from caller (catalog module provides it).
    """
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
                logger.warning("[%s] Random image caption send failed: %s", psid, repr(e))

        if image_url:
            try:
                await send_facebook_image(psid, image_url)
            except Exception as e:
                logger.warning("[%s] Random image send failed: %s", psid, repr(e))


# =========================================================
# Download image from Messenger attachment
# =========================================================

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
                logger.warning("Image download failed — status: %d", r.status_code)
                return None, "image/jpeg"
            mime = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
            return r.content, mime
    except Exception as e:
        logger.error("Image download error: %s", repr(e))
        return None, "image/jpeg"


# =========================================================
# Extract message content from webhook event
# =========================================================

def safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)


def extract_message_content(event: Dict[str, Any]) -> Dict[str, Any]:
    sender_id = event.get("sender", {}).get("id", "")
    message = event.get("message", {}) or {}
    text = (message.get("text") or "").strip()

    attachments = message.get("attachments") or []
    image_urls = []
    if attachments:
        for att in attachments:
            if att.get("type") == "image":
                url = ((att.get("payload") or {}).get("url") or "").strip()
                if url:
                    image_urls.append(url)

    return {
        "psid": sender_id,
        "text": text,
        "image_url": image_urls[0] if image_urls else "",
        "image_urls": image_urls,
        "is_echo": safe_bool(message.get("is_echo")),
        "metadata": (message.get("metadata") or "").strip(),
    }


# =========================================================
# Fetch profile name (best-effort)
# =========================================================

def maybe_get_profile_name(psid: str) -> str:
    """Best-effort only. If unavailable, fail quietly."""
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
