from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import numpy as np
import secrets
import json
import os

# ── Load environment variables ─────────────────────────────────────────────
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in your .env file")

# ── Load the embedding model ───────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready. API starting...")

app = FastAPI(
    title="MagnoAPI",
    description="""
Give any AI app persistent memory across conversations.

## Authentication
All `/memory` endpoints require an API key passed in the request header:
```
X-API-Key: magno_sk_your_key_here
```

Get your free API key by calling `POST /keys/create` with your email.

## Free Tier
- 1,000 API calls per month
- Includes both store and search operations
""",
    version="2.0.0"
)


# ══════════════════════════════════════════════════════════════════════════
# DATABASE HELPER
# ══════════════════════════════════════════════════════════════════════════

def get_db():
    return psycopg2.connect(DATABASE_URL)


# ══════════════════════════════════════════════════════════════════════════
# API KEY SECURITY
#
# FastAPI's APIKeyHeader reads the "X-API-Key" header from every request.
# If the header is missing, it automatically returns a 403 error.
# Our verify_api_key function then checks if the key exists in the database,
# is active, and hasn't exceeded its usage limit.
# ══════════════════════════════════════════════════════════════════════════

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency injected into every protected endpoint.
    Checks: key exists → key is active → usage is within limit.
    On success: increments usage count and returns the key record.
    On failure: raises 401 Unauthorized.
    """

    # Missing header
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Add 'X-API-Key: your_key' to your request headers. Get a free key at POST /keys/create"
        )

    conn = None
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Look up the key in the database
        cur.execute("""
            SELECT id, email, tier, usage_count, usage_limit, is_active
            FROM api_keys
            WHERE key = %s
        """, (api_key,))

        record = cur.fetchone()

        # Key doesn't exist
        if not record:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key. Check your key or get a new one at POST /keys/create"
            )

        # Key has been deactivated
        if not record["is_active"]:
            raise HTTPException(
                status_code=401,
                detail="This API key has been deactivated. Contact support."
            )

        # Usage limit exceeded
        if record["usage_count"] >= record["usage_limit"]:
            raise HTTPException(
                status_code=429,
                detail=f"Usage limit reached ({record['usage_limit']} calls/month on {record['tier']} tier). Upgrade at magnoapi.com/upgrade"
            )

        # ✅ Key is valid — increment usage count
        cur.execute("""
            UPDATE api_keys
            SET usage_count = usage_count + 1
            WHERE key = %s
        """, (api_key,))
        conn.commit()

        return dict(record)

    except HTTPException:
        raise
    except Exception as e:
        print(f"API KEY VERIFICATION ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════

class CreateKeyRequest(BaseModel):
    email: str

class CreateKeyResponse(BaseModel):
    api_key: str
    email: str
    tier: str
    usage_limit: int
    message: str

class StoreRequest(BaseModel):
    user_id: str
    text: str
    metadata: dict = {}

class StoreResponse(BaseModel):
    id: str
    message: str
    stored_at: str

class SearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 5

class MemoryResult(BaseModel):
    id: str
    text: str
    similarity: float
    metadata: dict
    stored_at: str

class SearchResponse(BaseModel):
    results: list[MemoryResult]
    total_found: int


# ══════════════════════════════════════════════════════════════════════════
# COSINE SIMILARITY HELPER
# ══════════════════════════════════════════════════════════════════════════

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ══════════════════════════════════════════════════════════════════════════
# STARTUP — Create tables if they don't exist
# ══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def create_tables():
    conn = None
    try:
        conn = get_db()
        cur = conn.cursor()

        # Memories table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id    TEXT        NOT NULL,
                text       TEXT        NOT NULL,
                embedding  JSONB       NOT NULL,
                metadata   JSONB       DEFAULT '{}',
                stored_at  TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS memories_user_id_idx ON memories (user_id);
        """)

        # API keys table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                key         TEXT UNIQUE NOT NULL,
                email       TEXT NOT NULL,
                tier        TEXT DEFAULT 'free',
                usage_count INTEGER DEFAULT 0,
                usage_limit INTEGER DEFAULT 1000,
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                is_active   BOOLEAN DEFAULT TRUE
            );
            CREATE INDEX IF NOT EXISTS api_keys_key_idx ON api_keys (key);
        """)

        conn.commit()
        print("Database tables ready.")
    except Exception as e:
        print(f"Startup error: {e}")
    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — POST /keys/create
#
# PUBLIC endpoint — no API key required to call this.
# A developer sends their email and gets back a unique API key.
#
# The key format is: magno_sk_ + 32 random hex characters
# Example: magno_sk_a3f7c821b9d4e2f8c1a05d9e7b3f4c6d
#
# This is the only door into your system. Everything else is locked
# behind the key that this endpoint creates.
# ══════════════════════════════════════════════════════════════════════════

@app.post("/keys/create", response_model=CreateKeyResponse)
async def create_api_key(request: CreateKeyRequest):
    """
    Get a free API key. No authentication required.
    Send your email and receive your key instantly.
    """

    if not request.email or "@" not in request.email:
        raise HTTPException(status_code=400, detail="A valid email address is required.")

    conn = None
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Check if this email already has a key
        cur.execute("SELECT key FROM api_keys WHERE email = %s", (request.email,))
        existing = cur.fetchone()

        if existing:
            return CreateKeyResponse(
                api_key=existing["key"],
                email=request.email,
                tier="free",
                usage_limit=1000,
                message="You already have an API key. Here it is again."
            )

        # Generate a new unique key
        # secrets.token_hex(16) generates 32 cryptographically random hex characters
        new_key = f"magno_sk_{secrets.token_hex(16)}"

        # Save to database
        cur.execute("""
            INSERT INTO api_keys (key, email, tier, usage_count, usage_limit, is_active)
            VALUES (%s, %s, 'free', 0, 1000, TRUE)
        """, (new_key, request.email))

        conn.commit()

        return CreateKeyResponse(
            api_key=new_key,
            email=request.email,
            tier="free",
            usage_limit=1000,
            message="Your API key has been created. Keep it safe — treat it like a password. Add it to all requests as: X-API-Key: " + new_key
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"CREATE KEY ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — POST /memory/store
#
# PROTECTED — requires a valid API key in the X-API-Key header.
# The `key_data = Depends(verify_api_key)` line is what enforces this.
# FastAPI runs verify_api_key before the endpoint function even starts.
# If the key is invalid, the request never reaches this function.
# ══════════════════════════════════════════════════════════════════════════

@app.post("/memory/store", response_model=StoreResponse)
async def store_memory(
    request: StoreRequest,
    key_data: dict = Depends(verify_api_key)   # ← API key check happens here
):
    """
    Save a memory for a user.
    Requires: X-API-Key header with your MagnoAPI key.
    Call this AFTER your LLM responds.
    """

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    conn = None
    try:
        embedding = model.encode(request.text).tolist()

        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            INSERT INTO memories (user_id, text, embedding, metadata, stored_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, stored_at
        """, (
            request.user_id,
            request.text,
            json.dumps(embedding),
            json.dumps(request.metadata),
            datetime.utcnow().isoformat()
        ))

        conn.commit()
        row = cur.fetchone()

        return StoreResponse(
            id=str(row["id"]),
            message="Memory stored successfully.",
            stored_at=str(row["stored_at"])
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"STORE ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — POST /memory/search
#
# PROTECTED — requires a valid API key in the X-API-Key header.
# ══════════════════════════════════════════════════════════════════════════

@app.post("/memory/search", response_model=SearchResponse)
async def search_memories(
    request: SearchRequest,
    key_data: dict = Depends(verify_api_key)   # ← API key check happens here
):
    """
    Find memories relevant to the current query.
    Requires: X-API-Key header with your MagnoAPI key.
    Call this BEFORE your LLM call, then inject results into the prompt.
    """

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    conn = None
    try:
        query_vector = model.encode(request.query).tolist()

        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT id, text, embedding, metadata, stored_at
            FROM memories
            WHERE user_id = %s
        """, (request.user_id,))

        rows = cur.fetchall()

        if not rows:
            return SearchResponse(results=[], total_found=0)

        scored = []
        for row in rows:
            score = cosine_similarity(query_vector, row["embedding"])
            scored.append({
                "id":         str(row["id"]),
                "text":       row["text"],
                "similarity": round(score, 4),
                "metadata":   row["metadata"] or {},
                "stored_at":  str(row["stored_at"])
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = scored[:request.top_k]

        return SearchResponse(
            results=[MemoryResult(**r) for r in top_results],
            total_found=len(top_results)
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"SEARCH ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════
# ROOT — Health check (public, no key required)
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return FileResponse("index.html")
    
