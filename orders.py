"""
Order management for Baby Catalog Bot V5.
Handles delivery computation, order completion checks, and Google Sheets export.
"""

import time
from typing import Any, Dict

import httpx

from config import (
    ORDER_WEBHOOK_URL,
    FREE_DELIVERY_THRESHOLD,
    FLAT_DELIVERY_CHARGE,
    DHAKA_ETA,
    OUTSIDE_DHAKA_ETA,
    logger,
)
from db import save_state


# =========================================================
# Delivery computation
# =========================================================

def compute_delivery_facts(state: Dict[str, Any]) -> None:
    """Compute delivery charge, free delivery flag, and ETA from current state."""
    # Compute total cart value (sum of all items) or matched product price
    cart = state.get("cart") or []
    if cart:
        total_price = sum(item.get("price_bdt", 0) or 0 for item in cart)
    else:
        total_price = state.get("matched_price_bdt") or 0

    free_delivery = bool(total_price >= FREE_DELIVERY_THRESHOLD)
    state["free_delivery"] = free_delivery
    state["delivery_charge"] = 0 if free_delivery else FLAT_DELIVERY_CHARGE

    zone = state.get("delivery_zone_final") or state.get("delivery_zone_inferred") or ""
    if zone == "inside_dhaka":
        state["delivery_eta"] = DHAKA_ETA
    elif zone == "outside_dhaka":
        state["delivery_eta"] = OUTSIDE_DHAKA_ETA
    else:
        state["delivery_eta"] = ""


# =========================================================
# Order completion check
# =========================================================

def order_complete(state: Dict[str, Any]) -> bool:
    return bool(
        state.get("recommended_size")
        and state.get("phone")
        and state.get("address")
    )


# =========================================================
# Order export (Google Sheets via Apps Script webhook)
# =========================================================

def _build_cart_summary(state: Dict[str, Any]) -> str:
    """Build a comma-separated product summary for multi-item orders."""
    cart = state.get("cart") or []
    if not cart:
        # Fallback to single matched product
        name = state.get("matched_product_name", "")
        return name

    parts = []
    for item in cart:
        name = item.get("product_name", "Unknown")
        qty = item.get("quantity", 1)
        if qty > 1:
            parts.append(f"{name} (x{qty})")
        else:
            parts.append(name)
    return ", ".join(parts)


def _build_cart_price_summary(state: Dict[str, Any]) -> str:
    """Build a price summary for multi-item orders."""
    cart = state.get("cart") or []
    if not cart:
        return state.get("matched_price_text", "")

    parts = []
    for item in cart:
        price_text = item.get("price_text", "")
        if price_text:
            parts.append(price_text)
    return " | ".join(parts)


def _build_cart_image_summary(state: Dict[str, Any]) -> str:
    """Build an image URL summary for multi-item orders."""
    cart = state.get("cart") or []
    if not cart:
        return state.get("matched_image_url", "")

    urls = [item.get("image_url", "") for item in cart if item.get("image_url")]
    return " | ".join(urls)


async def export_completed_order(state: Dict[str, Any]) -> None:
    """Export completed order to Google Sheets. One row per order (Option A)."""
    if not ORDER_WEBHOOK_URL or state.get("order_exported") or not order_complete(state):
        return

    payload = {
        "timestamp": int(time.time()),
        "customer_name": state.get("profile_name", ""),
        "facebook_psid": state.get("psid", ""),
        "addressing_style": state.get("addressing_style", ""),
        "product_name": _build_cart_summary(state),
        "product_price": _build_cart_price_summary(state),
        "product_image_url": _build_cart_image_summary(state),
        "recommended_size": state.get("recommended_size", ""),
        "phone": state.get("phone", ""),
        "address": state.get("address", ""),
        "delivery_zone": state.get("delivery_zone_final") or state.get("delivery_zone_inferred") or "",
        "delivery_eta": state.get("delivery_eta", ""),
        "delivery_charge": state.get("delivery_charge", ""),
        "free_delivery": state.get("free_delivery", False),
        "order_stage": state.get("order_stage", ""),
    }

    psid = state.get("psid", "unknown")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(ORDER_WEBHOOK_URL, json=payload)
            logger.info("[%s] Order export — status: %d", psid, r.status_code)
            r.raise_for_status()
        state["order_exported"] = True
        save_state(psid, state)
    except Exception as e:
        logger.error("[%s] Order export failed: %s", psid, repr(e))
