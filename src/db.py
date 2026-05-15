"""
db.py — PostgreSQL 連線管理 + CRUD 操作
======================================
所有與 PostgreSQL 的互動封裝在此。
"""

import os
import json
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

# ── 連線設定 ─────────────────────────────────────────────────
PG_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB", "lndata_db"),
    "user":     os.getenv("PG_USER", "lndata"),
    "password": os.getenv("PG_PASSWORD", "lndata_secret"),
}


def get_connection():
    """取得一個新的 PostgreSQL 連線（caller 負責 close）"""
    conn = psycopg2.connect(**PG_CONFIG)
    register_vector(conn)
    return conn


def _execute(sql, params=None, fetch=False):
    """執行單一 SQL 語句的便利函式"""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            if fetch:
                rows = cur.fetchall()
            else:
                rows = None
            conn.commit()
            return rows
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════
# CRUD — customers
# ═══════════════════════════════════════════════════════════

def save_customer(trace_id: str, record: dict, original_text: str):
    """存入客戶原始資料"""
    sql = """
        INSERT INTO customers (trace_id, name, phone, address, credit_card, category, original_text)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (trace_id) DO UPDATE SET
            name = EXCLUDED.name,
            phone = EXCLUDED.phone,
            address = EXCLUDED.address,
            credit_card = EXCLUDED.credit_card,
            category = EXCLUDED.category,
            original_text = EXCLUDED.original_text
    """
    _execute(sql, (
        trace_id,
        record.get("name", ""),
        record.get("phone", ""),
        record.get("address", ""),
        record.get("credit_card", ""),
        record.get("category", ""),
        original_text,
    ))


# ═══════════════════════════════════════════════════════════
# CRUD — pii_vault
# ═══════════════════════════════════════════════════════════

def save_vault(trace_id: str, vault_snapshot: dict):
    """
    批次寫入 PII 對照表。
    vault_snapshot: {token: real_value}, e.g. {"[PERSON_1]": "陳志明"}
    """
    if not vault_snapshot:
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # 推斷 entity_type from token
            rows = []
            for token, real_value in vault_snapshot.items():
                # token 格式: [PERSON_1], [PHONE_1], etc.
                entity_type = token.strip("[]").rsplit("_", 1)[0] if "_" in token else "UNKNOWN"
                rows.append((trace_id, token, real_value, entity_type))

            execute_values(
                cur,
                """
                INSERT INTO pii_vault (trace_id, token, real_value, entity_type)
                VALUES %s
                ON CONFLICT (trace_id, token) DO UPDATE SET
                    real_value = EXCLUDED.real_value,
                    entity_type = EXCLUDED.entity_type
                """,
                rows,
            )
            conn.commit()
    finally:
        conn.close()


def load_vault(trace_id: str) -> dict:
    """從 PostgreSQL 載入某 trace 的所有 token→real_value 對照"""
    rows = _execute(
        "SELECT token, real_value FROM pii_vault WHERE trace_id = %s",
        (trace_id,),
        fetch=True,
    )
    return {r["token"]: r["real_value"] for r in rows} if rows else {}


def load_vaults_batch(trace_ids: list) -> dict:
    """
    批次載入多個 trace 的 vault。
    回傳: {trace_id: {token: real_value}}
    """
    if not trace_ids:
        return {}
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT trace_id, token, real_value FROM pii_vault WHERE trace_id = ANY(%s)",
                (trace_ids,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    result = {}
    for r in rows:
        tid = r["trace_id"]
        if tid not in result:
            result[tid] = {}
        result[tid][r["token"]] = r["real_value"]
    return result


# ═══════════════════════════════════════════════════════════
# LangChain PGVector 向量資料庫
# ═══════════════════════════════════════════════════════════

_vectorstore = None

def get_vectorstore():
    """取得 LangChain PGVector 實例"""
    global _vectorstore
    if _vectorstore is None:
        try:
            from langchain_postgres import PGVector
            from src.embedding import CustomBGEEmbeddings
            
            # 使用 psycopg3 驅動
            CONNECTION_STRING = f"postgresql+psycopg://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
            
            _vectorstore = PGVector(
                embeddings=CustomBGEEmbeddings(),
                collection_name="complaints",
                connection=CONNECTION_STRING,
                use_jsonb=True,
            )
        except ImportError as e:
            print(f"[DB] 初始化 LangChain PGVector 失敗，請確認套件已安裝: {e}")
            return None
    return _vectorstore


# ═══════════════════════════════════════════════════════════
# CRUD — complaint_summaries
# ═══════════════════════════════════════════════════════════

def save_summary(trace_id: str, encoded_summary: str, decoded_summary: str,
                 model_used: str = "", query_text: str = ""):
    """存入 LLM 摘要"""
    _execute(
        """
        INSERT INTO complaint_summaries
            (trace_id, encoded_summary, decoded_summary, model_used, query_text)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (trace_id, encoded_summary, decoded_summary, model_used, query_text),
    )


# ═══════════════════════════════════════════════════════════
# 工具函式
# ═══════════════════════════════════════════════════════════

def check_connection() -> bool:
    """測試 PostgreSQL 連線"""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return True
    except Exception as e:
        print(f"[DB] 連線失敗: {e}")
        return False


def get_row_counts() -> dict:
    """回傳各表的行數"""
    tables = ["customers", "pii_vault", "langchain_pg_collection", "langchain_pg_embedding", "complaint_summaries"]
    counts = {}
    for t in tables:
        try:
            rows = _execute(f"SELECT COUNT(*) as cnt FROM {t}", fetch=True)
            counts[t] = rows[0]["cnt"] if rows else 0
        except Exception as e:
            counts[t] = f"Error (table not exists?)"
    return counts
