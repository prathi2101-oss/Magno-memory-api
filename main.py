from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import numpy as np
import json
import os

# ── Load the database URL from .env ───────────────────────────────────────
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
    description="Give any AI app persistent memory across conversations.",
    version="1.0.0"
)


# ══════════════════════════════════════════════════════════════════════════
# DATABASE HELPER
#
# Instead of using the supabase-py library (which had endless dependency
# conflicts), we connect directly to the PostgreSQL database.
# psycopg2 is the standard, rock-solid PostgreSQL driver for Python.
# It has been around for 15+ years with zero drama.
# ══════════════════════════════════════════════════════════════════════════

def get_db():
    """Open a fresh database connection. Always close it when done."""
    return psycopg2.connect(DATABASE_URL)


# ══════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════

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
# HELPER — Cosine Similarity
#
# Given two vectors, this tells us how similar they are in meaning.
# Score of 1.0 = identical meaning
# Score of 0.5 = somewhat related
# Score of 0.0 = completely unrelated
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
# STARTUP — Create the memories table if it doesn't exist
#
# This runs automatically when the API starts. So you don't need to
# manually run any SQL setup — the API creates its own table on first boot.
# ══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def create_table():
    """Create the memories table if it doesn't already exist."""
    conn = None
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id    TEXT        NOT NULL,
                text       TEXT        NOT NULL,
                embedding  JSONB       NOT NULL,
                metadata   JSONB       DEFAULT '{}',
                stored_at  TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS memories_user_id_idx
                ON memories (user_id);
        """)
        conn.commit()
        print("Database table ready.")
    except Exception as e:
        print(f"Startup table creation error: {e}")
    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — POST /memory/store
# ══════════════════════════════════════════════════════════════════════════

@app.post("/memory/store", response_model=StoreResponse)
async def store_memory(request: StoreRequest):
    """
    Save a memory for a user.
    Call this AFTER your LLM responds, passing in the conversation text.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    conn = None
    try:
        # Convert text to a 384-number vector representing its meaning
        embedding = model.encode(request.text).tolist()

        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Insert the memory directly into Postgres using standard SQL
        cur.execute("""
            INSERT INTO memories (user_id, text, embedding, metadata, stored_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, stored_at
        """, (
            request.user_id,
            request.text,
            json.dumps(embedding),          # Store vector as JSON
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
# ENDPOINT 2 — POST /memory/search
# ══════════════════════════════════════════════════════════════════════════

@app.post("/memory/search", response_model=SearchResponse)
async def search_memories(request: SearchRequest):
    """
    Find memories relevant to the current query.
    Call this BEFORE your LLM call, then inject results into the prompt.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    conn = None
    try:
        # Convert the search query into a vector
        query_vector = model.encode(request.query).tolist()

        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Fetch all memories belonging to this user
        cur.execute("""
            SELECT id, text, embedding, metadata, stored_at
            FROM memories
            WHERE user_id = %s
        """, (request.user_id,))

        rows = cur.fetchall()

        if not rows:
            return SearchResponse(results=[], total_found=0)

        # Score each memory by cosine similarity with the query vector
        scored = []
        for row in rows:
            stored_embedding = row["embedding"]
            score = cosine_similarity(query_vector, stored_embedding)
            scored.append({
                "id":         str(row["id"]),
                "text":       row["text"],
                "similarity": round(score, 4),
                "metadata":   row["metadata"] or {},
                "stored_at":  str(row["stored_at"])
            })

        # Sort by relevance (highest score first) and return top K
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
# ROOT — Health check
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "api":     "MagnoAPI",
        "version": "1.0.0",
        "status":  "running",
        "endpoints": [
            "POST /memory/store  → Save a memory",
            "POST /memory/search → Find relevant memories"
        ]
    }