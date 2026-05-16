from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.responses import FileResponse
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Load environment variables ─────────────────────────────────────────────
# load_dotenv() loads the variables from the .env file
# os.getenv calls the variables 
load_dotenv()
DATABASE_URL      = os.getenv("DATABASE_URL")
GMAIL_USER        = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

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
Your key will be sent directly to your inbox — it is never shown on the webpage.

## Free Tier
- 1,000 API calls per month
- Includes both store and search operations
""",
    version="3.0.0"
)


# ══════════════════════════════════════════════════════════════════════════
# EMAIL HELPER
#
# Sends the API key directly to the user's email address.
# The key is NEVER returned in the API response — only delivered via email.
# This means only the person who owns that inbox can ever see their key.
# ══════════════════════════════════════════════════════════════════════════

#def defines the function 
def send_api_key_email(to_email: str, api_key: str, is_existing: bool = False):
    """
    Send a beautifully formatted HTML email containing the user's API key.
    Uses Gmail SMTP with an App Password for secure sending.
    """
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        print("WARNING: Gmail credentials not set. Email not sent.")
        return False

    subject = "Your MagnoAPI Key is Ready" if not is_existing else "Your MagnoAPI Key (resent)"

    # ── HTML email body — matches MagnoAPI's bold colorful design ──────────
    html_body = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#050508;font-family:'Helvetica Neue',Arial,sans-serif;">

  <div style="max-width:560px;margin:0 auto;padding:40px 20px;">

    <!-- Header -->
    <div style="margin-bottom:32px;">
      <span style="font-size:24px;font-weight:900;color:#f0f0f5;letter-spacing:-0.02em;">
        Magno<span style="color:#ff2d6b;">API</span>
      </span>
    </div>

    <!-- Hero -->
    <div style="background:#0e0e14;border:1px solid #1e1e2e;border-radius:12px;padding:32px;margin-bottom:24px;">
      <div style="display:inline-block;background:rgba(184,255,60,0.1);border:1px solid rgba(184,255,60,0.3);border-radius:4px;padding:4px 12px;margin-bottom:20px;">
        <span style="font-size:11px;color:#b8ff3c;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;">
          {'✓ Key Retrieved' if is_existing else '✓ Key Created'}
        </span>
      </div>

      <h1 style="color:#f0f0f5;font-size:28px;font-weight:900;margin:0 0 8px;letter-spacing:-0.02em;">
        {'Here is your existing key' if is_existing else 'Your API key is ready.'}
      </h1>
      <p style="color:#888899;font-size:15px;margin:0 0 28px;line-height:1.5;">
        Keep this key private — treat it like a password. 
        It gives full access to your MagnoAPI account.
      </p>

      <!-- The API Key -->
      <div style="background:#080810;border:1px solid rgba(0,245,255,0.2);border-radius:8px;padding:20px;margin-bottom:8px;">
        <div style="font-size:10px;color:#00f5ff;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;margin-bottom:8px;font-family:monospace;">
          Your MagnoAPI Key
        </div>
        <div style="font-family:monospace;font-size:14px;color:#f0f0f5;word-break:break-all;line-height:1.5;">
          {api_key}
        </div>
      </div>
      <p style="color:#585b70;font-size:12px;margin:0;font-family:monospace;">
        Never share this key publicly or commit it to GitHub.
      </p>
    </div>

    <!-- How to use it -->
    <div style="background:#0e0e14;border:1px solid #1e1e2e;border-radius:12px;padding:32px;margin-bottom:24px;">
      <h2 style="color:#f0f0f5;font-size:18px;font-weight:800;margin:0 0 20px;">
        How to use your key
      </h2>

      <div style="margin-bottom:16px;">
        <div style="font-size:11px;color:#ff2d6b;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:8px;font-family:monospace;">
          Step 1 — Add to every request header
        </div>
        <div style="background:#080810;border-radius:6px;padding:14px;font-family:monospace;font-size:13px;color:#b8ff3c;">
          X-API-Key: {api_key}
        </div>
      </div>

      <div style="margin-bottom:16px;">
        <div style="font-size:11px;color:#ff2d6b;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:8px;font-family:monospace;">
          Step 2 — Search memories before your LLM call
        </div>
        <div style="background:#080810;border-radius:6px;padding:14px;font-family:monospace;font-size:12px;color:#cdd6f4;line-height:1.6;">
          POST /memory/search<br>
          {{"user_id": "your_user", "query": "user message", "top_k": 3}}
        </div>
      </div>

      <div>
        <div style="font-size:11px;color:#ff2d6b;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:8px;font-family:monospace;">
          Step 3 — Store memory after your LLM responds
        </div>
        <div style="background:#080810;border-radius:6px;padding:14px;font-family:monospace;font-size:12px;color:#cdd6f4;line-height:1.6;">
          POST /memory/store<br>
          {{"user_id": "your_user", "text": "conversation text"}}
        </div>
      </div>
    </div>

    <!-- Plan details -->
    <div style="background:#0e0e14;border:1px solid #1e1e2e;border-radius:12px;padding:24px;margin-bottom:24px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
        <div>
          <div style="font-size:11px;color:#888899;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:4px;">Current Plan</div>
          <div style="font-size:20px;font-weight:900;color:#b8ff3c;">Free Tier</div>
        </div>
        <div>
          <div style="font-size:11px;color:#888899;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:4px;">Monthly Limit</div>
          <div style="font-size:20px;font-weight:900;color:#f0f0f5;">1,000 calls</div>
        </div>
        <div>
          <div style="font-size:11px;color:#888899;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:4px;">LLMs Supported</div>
          <div style="font-size:20px;font-weight:900;color:#f0f0f5;">All of them</div>
        </div>
      </div>
    </div>

    <!-- CTA -->
    <div style="text-align:center;margin-bottom:32px;">
      <a href="https://magno-memory-api-production.up.railway.app/docs"
         style="display:inline-block;background:#ff2d6b;color:#050508;padding:14px 32px;border-radius:6px;font-weight:800;font-size:15px;text-decoration:none;letter-spacing:-0.01em;">
        View API Documentation →
      </a>
    </div>

    <!-- Footer -->
    <div style="text-align:center;border-top:1px solid #1e1e2e;padding-top:24px;">
      <p style="color:#585b70;font-size:12px;margin:0;font-family:monospace;">
        MagnoAPI · Built different<br>
        Questions? Reply to this email.
      </p>
    </div>

  </div>
</body>
</html>
"""

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"MagnoAPI <{GMAIL_USER}>"
        msg["To"]      = to_email

        # Attach HTML version
        msg.attach(MIMEText(html_body, "html"))

        # Connect to Gmail's SMTP server and send
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, to_email, msg.as_string())

        print(f"EMAIL SENT: API key delivered to {to_email}")
        return True

    except Exception as e:
        print(f"EMAIL ERROR: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════
# DATABASE HELPER
# ══════════════════════════════════════════════════════════════════════════

def get_db():
    return psycopg2.connect(DATABASE_URL)


# ══════════════════════════════════════════════════════════════════════════
# API KEY SECURITY
# ══════════════════════════════════════════════════════════════════════════

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Add 'X-API-Key: your_key' to your request headers. Get a free key at POST /keys/create"
        )

    conn = None
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT id, email, tier, usage_count, usage_limit, is_active
            FROM api_keys WHERE key = %s
        """, (api_key,))

        record = cur.fetchone()

        if not record:
            raise HTTPException(status_code=401, detail="Invalid API key.")

        if not record["is_active"]:
            raise HTTPException(status_code=401, detail="This API key has been deactivated.")

        if record["usage_count"] >= record["usage_limit"]:
            raise HTTPException(
                status_code=429,
                detail=f"Usage limit reached ({record['usage_limit']} calls/month on {record['tier']} tier)."
            )

        cur.execute("UPDATE api_keys SET usage_count = usage_count + 1 WHERE key = %s", (api_key,))
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
# Note: CreateKeyResponse no longer contains the api_key.
# The key is sent privately via email only.
# ══════════════════════════════════════════════════════════════════════════

class CreateKeyRequest(BaseModel):
    email: str

class CreateKeyResponse(BaseModel):
    email: str
    status: str
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
# Creates the key, emails it privately to the user, and returns
# ONLY a success confirmation — never the key itself in the response.
# ══════════════════════════════════════════════════════════════════════════

@app.post("/keys/create", response_model=CreateKeyResponse)
async def create_api_key(request: CreateKeyRequest):
    """
    Get a free API key sent directly to your email.
    The key is never shown on the webpage — only delivered to your inbox.
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
            # Resend the existing key to their email
            email_sent = send_api_key_email(request.email, existing["key"], is_existing=True)
            return CreateKeyResponse(
                email=request.email,
                status="resent",
                message=f"You already have a key. We've resent it to {request.email}. Check your inbox."
                if email_sent else
                f"You already have a key but email delivery failed. Contact support."
            )

        # Generate a brand new key
        new_key = f"magno_sk_{secrets.token_hex(16)}"

        # Save to database
        cur.execute("""
            INSERT INTO api_keys (key, email, tier, usage_count, usage_limit, is_active)
            VALUES (%s, %s, 'free', 0, 1000, TRUE)
        """, (new_key, request.email))
        conn.commit()

        # Send the key privately via email
        # The key is NEVER included in the API response below
        email_sent = send_api_key_email(request.email, new_key, is_existing=False)

        return CreateKeyResponse(
            email=request.email,
            status="created",
            message=f"Your API key has been sent to {request.email}. Check your inbox (and spam folder just in case)."
            if email_sent else
            f"Key created but email delivery failed. Please contact triviacolosseum@gmail.com."
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
# ENDPOINT 2 — POST /memory/store  (protected)
# ══════════════════════════════════════════════════════════════════════════

@app.post("/memory/store", response_model=StoreResponse)
async def store_memory(
    request: StoreRequest,
    key_data: dict = Depends(verify_api_key)
):
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
# ENDPOINT 3 — POST /memory/search  (protected)
# ══════════════════════════════════════════════════════════════════════════

@app.post("/memory/search", response_model=SearchResponse)
async def search_memories(
    request: SearchRequest,
    key_data: dict = Depends(verify_api_key)
):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    conn = None
    try:
        query_vector = model.encode(request.query).tolist()
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT id, text, embedding, metadata, stored_at
            FROM memories WHERE user_id = %s
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
# ROOT — Serves the landing page
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return FileResponse("index.html")