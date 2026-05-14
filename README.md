
# MagnoAPI 🧠
### Give any AI app persistent memory. Two endpoints. That's it.

---

## What This API Does (Simply Explained)

When a user chats with an AI app, everything is forgotten the moment the
conversation ends. This API solves that permanently.

```
WITHOUT Magno API:              WITH Magno API:
─────────────────              ──────────────────────────────
User: "I like Python"          User: "I like Python"
AI:   "Great!"                 AI:   "Great!"
                                     ↓ Magno API saves this
[next day]                     [next day]
User: "What should I use?"     User: "What should I use?"
AI:   "I don't know your       AI:   "Based on your preference
       preferences..."                for Python, I'd recommend..."
                                     ↑ Magno API retrieves this
```

---

## Endpoints

| Method | Endpoint         | What it does                          |
|--------|------------------|---------------------------------------|
| POST   | /memory/store    | Save a memory (text → vector → DB)    |
| POST   | /memory/search   | Find relevant past memories           |

---

## Setup on Windows 10 (Step by Step)

### Step 1 — Install Python
1. Go to https://python.org/downloads
2. Download Python 3.11
3. Run installer — CHECK "Add Python to PATH" before clicking Install
4. Open Command Prompt and type: `python --version`
   You should see: Python 3.11.x

### Step 2 — Install VS Code
1. Go to https://code.visualstudio.com
2. Download and install
3. Open VS Code → install the "Python" extension

### Step 3 — Create your project folder
Open Command Prompt and run:
```
mkdir C:\memory-api
cd C:\memory-api
```

Copy all files from this project into that folder.

### Step 4 — Create a virtual environment
In Command Prompt (inside C:\memory-api):
```
python -m venv venv
venv\Scripts\activate
```
You'll see (venv) appear in your terminal. Good.

### Step 5 — Install dependencies
```
pip install -r requirements.txt
```
This downloads everything needed (~500MB first time, takes 2-5 minutes).

### Step 6 — Set up Supabase (free)
1. Go to https://supabase.com → Sign up free
2. Click "New Project" → give it a name → choose a region
3. Wait ~2 minutes for it to provision
4. Go to: Settings → API
5. Copy your "Project URL" and "anon public" key
6. Create a file called `.env` in your project folder:
   ```
   SUPABASE_URL=https://yourproject.supabase.co
   SUPABASE_KEY=your-anon-key-here
   ```

### Step 7 — Set up the database
1. In Supabase dashboard → click "SQL Editor" in the left sidebar
2. Click "New Query"
3. Open the file `supabase_setup.sql` from this project
4. Copy ALL the contents and paste into the SQL editor
5. Click "Run"
6. You should see "Success. No rows returned"

### Step 8 — Start the API
In Command Prompt (with venv activated):
```
uvicorn main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Step 9 — Test it
Open a SECOND Command Prompt window:
```
cd C:\memory-api
venv\Scripts\activate
python test_api.py
```

You should see memories being stored and retrieved successfully.

### Step 10 — View the interactive docs
Open your browser and go to:
```
http://localhost:8000/docs
```
You'll see a full interactive API explorer where you can test every endpoint.

---

## How Developers Will Use Your API

```python
import requests

# ── Before calling Claude/GPT/Gemini ─────────────────────────────
# Get relevant memories for the user's current message
memories = requests.post("https://yourapi.com/memory/search", json={
    "user_id": "user_123",
    "query":   "What are my project preferences?",
    "top_k":   3
}).json()["results"]

# Inject into your LLM prompt
context = "\n".join([m["text"] for m in memories])
# → Claude now knows about past conversations

# ── After getting the LLM response ───────────────────────────────
# Save what just happened
requests.post("https://yourapi.com/memory/search", json={
    "user_id": "user_123",
    "text":    "User asked about project preferences and prefers Python + FastAPI",
    "metadata": {"session": "2026-05-08"}
})
```

---

## Deploy for Free (Share With Developers)

1. Go to https://railway.app → Sign up with GitHub
2. Click "New Project" → "Deploy from GitHub repo"
3. Connect your repo and add your .env variables in Railway settings
4. Railway gives you a public URL like: `https://memory-api.railway.app`
5. Share that URL with developers to start using your API

---

## Monetise With Stripe

1. Sign up at https://stripe.com
2. Create a "Usage" product with per-operation pricing
3. Add a middleware to main.py that counts calls per API key
4. Gate requests that exceed the free tier

---

Built with FastAPI + Supabase pgvector + sentence-transformers
=======
# Magno-memory-api
>>>>>>> 47c374ce3e2463adbb18e941a3069e903c2c87a3
