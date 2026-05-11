-- ════════════════════════════════════════════════════════════════
-- Run this entire script in your Supabase SQL Editor (one paste)
-- Dashboard → SQL Editor → New Query → Paste → Run
-- ════════════════════════════════════════════════════════════════


-- Step 1: Enable the pgvector extension
-- This gives Postgres the ability to store and search vectors
create extension if not exists vector;


-- Step 2: Create the memories table
create table if not exists memories (
    id          uuid primary key default gen_random_uuid(),
    user_id     text        not null,     -- who this memory belongs to
    text        text        not null,     -- the original text
    embedding   vector(384) not null,     -- 384-dim vector from all-MiniLM-L6-v2
    metadata    jsonb       default '{}', -- optional extra data
    stored_at   timestamptz default now()
);


-- Step 3: Create an index for fast vector search
-- Without this, every search scans the whole table (slow at scale)
create index if not exists memories_embedding_idx
    on memories
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);


-- Step 4: Index on user_id so searches filter by user quickly
create index if not exists memories_user_id_idx
    on memories (user_id);


-- Step 5: Create the search function
-- This is what your /memory/search endpoint calls via supabase.rpc()
-- It finds the most similar memories for a given user and query vector
create or replace function match_memories(
    query_embedding vector(384),
    match_user_id   text,
    match_count     int default 5
)
returns table (
    id         uuid,
    text       text,
    metadata   jsonb,
    stored_at  timestamptz,
    similarity float
)
language sql stable
as $$
    select
        m.id,
        m.text,
        m.metadata,
        m.stored_at,
        1 - (m.embedding <=> query_embedding) as similarity
    from memories m
    where m.user_id = match_user_id
    order by m.embedding <=> query_embedding   -- cosine distance (lower = more similar)
    limit match_count;
$$;
