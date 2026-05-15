-- ═══════════════════════════════════════════════════════════
-- lndata_db 初始化 Schema
-- ═══════════════════════════════════════════════════════════

-- 啟用 pgvector 擴充
CREATE EXTENSION IF NOT EXISTS vector;

-- ── ① 客戶原始資料 ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customers (
    id              SERIAL PRIMARY KEY,
    trace_id        VARCHAR(32) UNIQUE NOT NULL,
    name            TEXT,
    phone           TEXT,
    address         TEXT,
    credit_card     TEXT,
    category        VARCHAR(50),
    original_text   TEXT,                       -- 原始投訴全文
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── ② PII 對照表（Vault 永久備份）──────────────────────────
CREATE TABLE IF NOT EXISTS pii_vault (
    id              SERIAL PRIMARY KEY,
    trace_id        VARCHAR(32) NOT NULL,
    token           VARCHAR(50) NOT NULL,       -- e.g. [PERSON_1]
    real_value      TEXT NOT NULL,
    entity_type     VARCHAR(20),                -- PERSON, PHONE, ADDRESS, CREDIT_CARD
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(trace_id, token)
);

CREATE INDEX IF NOT EXISTS idx_vault_trace ON pii_vault(trace_id);

-- (已廢棄) ③ 原 complaint_embeddings 表格已移除
-- 向量資料現在改由 LangChain PGVector 套件自動建立的 
-- langchain_pg_collection 與 langchain_pg_embedding 表格來管理。

-- ── ④ LLM 整理後的摘要 ────────────────────────────────────
CREATE TABLE IF NOT EXISTS complaint_summaries (
    id              SERIAL PRIMARY KEY,
    trace_id        VARCHAR(32) NOT NULL REFERENCES customers(trace_id),
    encoded_summary TEXT NOT NULL,               -- LLM 輸出，仍含 [TAG@xxx]
    decoded_summary TEXT,                        -- Decoder 還原後的版本
    model_used      VARCHAR(50),
    query_text      TEXT,                        -- 觸發此摘要的查詢
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
