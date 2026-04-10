"""
SQLite state management for Baby Catalog Bot V5.
Handles customer state persistence with WAL mode and safe transactions.
"""

import json
import sqlite3
import time
from typing import Any, Dict

from config import SQLITE_PATH, FLAT_DELIVERY_CHARGE, logger


# =========================================================
# Connection
# =========================================================

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
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


# =========================================================
# Default state
# =========================================================

def default_state(psid: str) -> Dict[str, Any]:
    return {
        "psid": psid,
        "profile_name": "",
        "addressing_style": "apu",        # apu | vai | neutral
        "message_count": 0,
        "human_takeover": False,
        "last_user_message": "",
        "last_bot_reply": "",
        "conversation_summary": "",
        "user_intent": "",

        # Currently-viewed product (buffer before adding to cart)
        "matched_product_name": "",
        "matched_price_text": "",
        "matched_price_bdt": None,
        "matched_image_url": "",
        "matched_product_page_url": "",

        # Multi-product cart
        "cart": [],

        # Sizing
        "recommended_size": "",
        "size_finalized": False,
        "size_confidence": "",

        # Contact & delivery
        "phone": "",
        "phone_pending": "",
        "address": "",
        "address_pending": "",
        "awaiting_confirmation": "",       # "" | "address" | "phone"
        "delivery_zone_inferred": "",
        "delivery_zone_final": "",
        "delivery_zone_source": "",
        "delivery_eta": "",
        "delivery_charge": FLAT_DELIVERY_CHARGE,
        "free_delivery": False,

        # Order
        "order_stage": "browsing",
        "order_exported": False,
        "notes": "",

        # Structured permanent memory (never truncated)
        "key_facts": {
            "preferred_product_type": "",   # bodysuit / romper / pajama
            "baby_age_months": None,
            "is_gift": False,
            "preferred_colors": [],
            "budget_range": "",
            "previous_purchases": [],
        },

        # Rate limiting
        "last_message_timestamp": 0.0,
    }


# =========================================================
# State merge (forward-compatible with new fields)
# =========================================================

def merge_state(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in incoming.items():
        if k == "key_facts" and isinstance(v, dict):
            # Deep merge key_facts so new sub-keys get defaults
            base_kf = merged.get("key_facts", {})
            if isinstance(base_kf, dict):
                merged_kf = dict(base_kf)
                merged_kf.update(v)
                merged["key_facts"] = merged_kf
            else:
                merged["key_facts"] = v
        else:
            merged[k] = v
    return merged


# =========================================================
# Load & Save
# =========================================================

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
    except Exception as e:
        logger.error("[%s] Failed to load state: %s", psid, repr(e))
        return default_state(psid)
    finally:
        conn.close()


def save_state(psid: str, state: Dict[str, Any]) -> None:
    conn = get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
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
    except Exception as e:
        logger.error("[%s] Failed to save state: %s", psid, repr(e))
        conn.rollback()
    finally:
        conn.close()
