"""
test_api.py — Run this to verify both endpoints are working correctly.

Make sure your API server is running first:
    uvicorn main:app --reload

Then in a second terminal:
    python test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"
USER_ID  = "surya_test_user"


def print_section(title):
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print('═' * 55)


# ── TEST 1: Health check ──────────────────────────────────────
print_section("Health Check")
r = requests.get(f"{BASE_URL}/")
print(json.dumps(r.json(), indent=2))


# ── TEST 2: Store some memories ───────────────────────────────
print_section("Storing 3 Memories")

memories_to_store = [
    {
        "user_id":  USER_ID,
        "text":     "The user prefers Python over JavaScript for backend development.",
        "metadata": {"source": "conversation", "topic": "programming"}
    },
    {
        "user_id":  USER_ID,
        "text":     "The user is building a memory API for AI applications using Supabase and FastAPI.",
        "metadata": {"source": "conversation", "topic": "project"}
    },
    {
        "user_id":  USER_ID,
        "text":     "The user wants to target warehouse and industrial clients as their first customers.",
        "metadata": {"source": "conversation", "topic": "business"}
    }
]

for mem in memories_to_store:
    r = requests.post(f"{BASE_URL}/memory/store", json=mem)
    result = r.json()
    print(f"✅ Stored: '{mem['text'][:60]}...'")
    print(f"   ID: {result['id']}")


# ── TEST 3: Search memories ───────────────────────────────────
print_section("Searching Memories")

queries = [
    "What programming language does the user like?",
    "What is the user building?",
    "Who are the target customers?"
]

for query in queries:
    print(f"\n🔍 Query: '{query}'")
    r = requests.post(f"{BASE_URL}/memory/search", json={
        "user_id": USER_ID,
        "query":   query,
        "top_k":   2
    })
    results = r.json()["results"]
    for i, result in enumerate(results):
        print(f"   [{i+1}] Similarity: {result['similarity']:.4f}")
        print(f"       Text: {result['text']}")


# ── TEST 4: Simulate how a developer uses this ────────────────
print_section("Real-World Usage Example")
print("""
HOW A DEVELOPER USES YOUR API IN THEIR APP:
─────────────────────────────────────────────

# Step 1: User sends a message to their AI app
user_message = "Which programming language should I use for my backend?"

# Step 2: Developer calls YOUR API first (before calling Claude/GPT)
memories = requests.post("http://yourapi.com/memory/search", json={
    "user_id": "user_123",
    "query":   user_message,
    "top_k":   3
}).json()["results"]

# Step 3: Inject memories into the LLM prompt
memory_context = "\\n".join([m["text"] for m in memories])
prompt = f\"\"\"
You are a helpful assistant.

Relevant context from past conversations:
{memory_context}

User: {user_message}
\"\"\"

# Step 4: Call Claude/GPT with enriched prompt
response = anthropic.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": prompt}]
)

# Step 5: Store this new conversation in YOUR API
requests.post("http://yourapi.com/memory/store", json={
    "user_id": "user_123",
    "text":    f"User asked: {user_message}. Assistant said: {response.content[0].text[:200]}"
})

# Result: The AI now "remembers" the user prefers Python
# and gives a personalised answer — powered entirely by YOUR API
""")

print("✅ All tests complete. Your Memory API is working!")
