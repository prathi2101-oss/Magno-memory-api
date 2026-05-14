
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

## How You Can Use Magno API For Your Own      Software 

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

=======
# Magno-memory-api
>>>>>>>

# In Development
# New Updates And Changes Coming Soon 