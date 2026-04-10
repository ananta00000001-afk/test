# Baby Catalog Bot Gemini v3

This version does all of these:
- receives Facebook Messenger text and image messages
- matches product from `catalog.csv`
- replies in Bengali with product + price + likely size
- sends the matched product image URL from the CSV
- remembers corrections:
  - gender/addressing
  - inside/outside Dhaka
  - size
  - phone
  - address
- uses delivery logic:
  - inside Dhaka: 24 hours
  - outside Dhaka: 48–72 hours
  - free delivery for orders >= ৳999
  - otherwise flat ৳100 delivery
- becomes a little more persuasive after more chats
- stops AI forever for that customer if a human operator replies manually from the page
- sends completed orders to a Google Sheet via a webhook

## Files
- `app.py` — FastAPI app
- `catalog.csv` — your product catalog
- `requirements.txt`
- `.env.example`
- `google_apps_script_order_sink.js` — paste this into Google Apps Script for the Google Sheets handoff

## GitHub steps
1. Extract this zip on your computer.
2. Open your GitHub repo.
3. Delete the old files in the repo root.
4. Upload these new extracted files.
5. Commit the changes.

Upload the extracted files, not the zip.

## Render steps
Create or update a Web Service with:
- Build Command:
  `pip install -r requirements.txt`
- Start Command:
  `uvicorn app:app --host 0.0.0.0 --port $PORT`

Add these environment variables in Render:
- `VERIFY_TOKEN`
- `FACEBOOK_PAGE_ACCESS_TOKEN`
- `FACEBOOK_PAGE_ID`
- `GEMINI_API_KEY`
- `GEMINI_VISION_MODEL`
- `GEMINI_TEXT_MODEL`
- `CATALOG_CSV_PATH`
- `SQLITE_PATH`
- `ORDER_WEBHOOK_URL`

Then click **Manual Deploy** → **Deploy latest commit**.

## Meta / Messenger steps
1. Callback URL:
   `https://YOUR-RENDER-APP.onrender.com/webhook`
2. Verify token:
   same value as `VERIFY_TOKEN`
3. Turn on these webhook fields:
   - `messages`
   - `message_echoes`
   - `messaging_postbacks` (optional)
   - `message_reads` (optional)
   - `message_deliveries` (optional)

`message_echoes` is important because that is how the bot can detect when a human operator replied manually and then stop AI forever for that customer.

## Google Sheets handoff setup

### A. Create the sheet
1. Open Google Sheets.
2. Create a new sheet.
3. Rename the first tab to `Orders`.
4. In row 1, create these columns:

timestamp | customer_name | facebook_psid | addressing_style | product_name | product_price | product_image_url | recommended_size | phone | address | delivery_zone | delivery_eta | delivery_charge | free_delivery | order_stage

### B. Create the Apps Script
1. In the Google Sheet, click **Extensions** → **Apps Script**
2. Delete the default code.
3. Paste the contents of `google_apps_script_order_sink.js`
4. Save

### C. Deploy the Apps Script as Web App
1. Click **Deploy** → **New deployment**
2. Select type: **Web app**
3. Execute as: **Me**
4. Who has access: **Anyone**
5. Deploy
6. Copy the Web App URL

### D. Put that URL into Render
Set:
`ORDER_WEBHOOK_URL=<your_web_app_url>`

Then redeploy Render.

## How completed orders are sent
When these are all present:
- size
- phone
- address

the app sends a JSON payload to `ORDER_WEBHOOK_URL`, and the Apps Script appends it to the Google Sheet.

## Important safety notes
You exposed Facebook and Gemini keys earlier in screenshots. Rotate both after this setup.

## Troubleshooting
- If the bot receives messages but does not reply, check Render logs for Facebook send errors.
- If replies stop after a human message, that is expected. The customer is permanently locked to human-operator mode.
- If Google Sheets does not update, open the Apps Script **Executions** panel and check errors.
