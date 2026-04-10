"""
Centralized configuration for Baby Catalog Bot V5.
All environment variables and business constants live here.
"""

import os

# =========================================================
# Facebook / Messenger
# =========================================================

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "").strip()
FACEBOOK_PAGE_ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN", "").strip()
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID", "").strip()

# =========================================================
# Gemini AI
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.1-pro-preview").strip()
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", GEMINI_TEXT_MODEL).strip()

# =========================================================
# Data paths
# =========================================================

CATALOG_CSV_PATH = os.getenv("CATALOG_CSV_PATH", "./catalog.csv").strip()
SQLITE_PATH = os.getenv("SQLITE_PATH", "./state.db").strip()
ORDER_WEBHOOK_URL = os.getenv("ORDER_WEBHOOK_URL", "").strip()

# =========================================================
# Bot identity
# =========================================================

AI_MESSAGE_METADATA = "ai_bot_v5"

# =========================================================
# Delivery business rules
# =========================================================

FREE_DELIVERY_THRESHOLD = 999      # BDT — orders >= this get free delivery
FLAT_DELIVERY_CHARGE = 100         # BDT — flat charge when below threshold
DHAKA_ETA = "24 hours"
OUTSIDE_DHAKA_ETA = "48-72 hours"

# =========================================================
# Rate limiting
# =========================================================

RATE_LIMIT_SECONDS = 2             # Min seconds between messages from same PSID

# =========================================================
# Retry settings (for Gemini API calls)
# =========================================================

GEMINI_MAX_RETRIES = 2
GEMINI_RETRY_DELAYS = [1.0, 3.0]   # Seconds — exponential backoff

# =========================================================
# Logging
# =========================================================

import logging

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()

logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("baby_bot")
