# Baby Catalog Bot (Gemini + Messenger)

This backend:
- verifies Meta webhooks
- receives Messenger text/image messages
- analyzes uploaded product images with Gemini
- matches products against `catalog.csv`
- replies with the matched catalog price
- stores simple customer state in SQLite

## Files
- `app.py` — FastAPI app
- `catalog.csv` — product catalog
- `requirements.txt` — Python dependencies
- `.env.example` — environment variables

## Render settings
### Build command
```bash
pip install -r requirements.txt
```

### Start command
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

## Required environment variables in Render
- `VERIFY_TOKEN`
- `FACEBOOK_PAGE_ACCESS_TOKEN`
- `FACEBOOK_PAGE_ID`
- `GEMINI_API_KEY`
- `GEMINI_VISION_MODEL`
- `GEMINI_TEXT_MODEL`
- `CATALOG_CSV_PATH`
- `SQLITE_PATH`

## Recommended values
```text
VERIFY_TOKEN=ananta
GEMINI_VISION_MODEL=gemini-2.5-flash
GEMINI_TEXT_MODEL=gemini-2.5-flash
CATALOG_CSV_PATH=./catalog.csv
SQLITE_PATH=./state.db
```

## Meta webhook
Set your callback URL to:
```text
https://YOUR-RENDER-APP.onrender.com/webhook
```

Use the same `VERIFY_TOKEN` value in Meta and Render.
