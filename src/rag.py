"""
rag.py — RAG 查詢引擎
====================
完整流程：
  1. 使用者查詢 → embed
  2. pgvector 搜尋相似投訴 → top-k
  3. Re-tag：[PERSON_1] → [PERSON_1@001]，建立 merged vault
  4. 組裝 context → 送 LLM
  5. LLM 回答（含 @tag）→ Decoder 還原
  6. 存摘要到 PostgreSQL
"""

import os
import re
import time
from typing import Optional

from dotenv import load_dotenv

from src.db import save_summary, get_vectorstore
from src.embedding import embed_query
from src.decoder import (
    PrivacyDecoder,
    retag_text,
    build_merged_vault,
    extract_tags,
)
from src.llm import call_llm

load_dotenv()

decoder = PrivacyDecoder()


# ═══════════════════════════════════════════════════════════
# RAG Context 組裝
# ═══════════════════════════════════════════════════════════

def _build_context(results: list) -> tuple:
    """
    將搜尋結果組裝為 LLM context。
    做 re-tag 使每個 tag 全域唯一。

    回傳: (context_str, merged_vault)
    """
    trace_ids = [r["trace_id"] for r in results]
    merged_vault = build_merged_vault(trace_ids)

    context_parts = []
    for i, r in enumerate(results, 1):
        tid = r["trace_id"]
        encoded = r["encoded_text"]
        category = r.get("category", "")
        similarity = r.get("similarity", 0)

        # Re-tag：[PERSON_1] → [PERSON_1@xxx]
        retagged = retag_text(encoded, tid)

        context_parts.append(
            f"[案件 {tid}] (類別: {category}, 相似度: {similarity:.3f})\n{retagged}"
        )

    context_str = "\n\n".join(context_parts)
    return context_str, merged_vault


# ═══════════════════════════════════════════════════════════
# LLM Prompt
# ═══════════════════════════════════════════════════════════

RAG_PROMPT_TEMPLATE = """你是一個專業的客服投訴分析助手。根據以下檢索到的客戶投訴記錄，回答使用者的問題。

重要規則：
1. 你的回答中必須保留所有方括號標記，例如 [PERSON_1@001], [PHONE_1@003] 等。
   這些是隱私保護佔位符，絕對不能移除、修改、或自行展開。
2. 當引用特定案件時，請註明案件編號（如 TRACE_001）。
3. 回答應結構化、簡潔，聚焦於問題本身。
4. 如果檢索結果不足以回答問題，請誠實說明。

=== 檢索到的投訴記錄 ===
{context}

=== 使用者問題 ===
{query}

請根據以上記錄回答：
"""

RAG_GUEST_PROMPT_TEMPLATE = """你是一個專業的客服投訴分析助手。根據以下檢索到的客戶投訴記錄，回答使用者的問題。

重要規則：
1. 你的回答必須是一個高層次的總結（Summary）。
2. 絕對不要在回答中使用或保留任何方括號標記（如 [PERSON_1@001], [PHONE_1@003] 等）。請將這些佔位符轉換為通用的代名詞，例如「某位客戶」、「某個地址」、「某張信用卡」等。
3. 你的目標是讓讀者了解發生了什麼事（例如投訴的內容、情況、處理進度），但不能透露任何具體的個資或內部代碼。
4. 如果檢索結果不足以回答問題，請誠實說明。

=== 檢索到的投訴記錄 ===
{context}

=== 使用者問題 ===
{query}

請根據以上記錄，提供不含任何標籤的總結報告：
"""


# ═══════════════════════════════════════════════════════════
# LLM 呼叫（透過 llm.py fallback chain）
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
# 主 RAG 函式
# ═══════════════════════════════════════════════════════════

def retrieve(query: str, top_k: int = 5) -> list:
    """
    步驟 1-2：LangChain PGVector 搜尋
    回傳 top-k 相似結果
    """
    vectorstore = get_vectorstore()
    if vectorstore is None:
        print("[RAG] ❌ 找不到向量資料庫")
        return []

    docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    
    results = []
    for doc, score in docs_and_scores:
        results.append({
            "trace_id": doc.metadata.get("trace_id", "UNKNOWN"),
            "category": doc.metadata.get("category", ""),
            "encoded_text": doc.page_content,
            "similarity": 1.0 - score  # assuming L2 or cosine distance
        })
    return results


def ask(query: str, top_k: int = 5, verbose: bool = True) -> dict:
    """
    完整 RAG 管線：
      query → embed → retrieve → re-tag → LLM → decode → 存摘要

    回傳:
      {
        "query": str,
        "encoded_answer": str,      # LLM 原始回答（含 @tag）
        "decoded_answer": str,      # 還原後的回答
        "retrieved_count": int,
        "retrieved_traces": list,
        "tag_validation": dict,
        "model_used": str,
      }
    """
    # ── Step 1-2: Retrieve ────────────────────────────────────
    results = retrieve(query, top_k=top_k)

    if not results:
        return {
            "query": query,
            "encoded_answer": "沒有找到相關的投訴記錄。",
            "decoded_answer": "沒有找到相關的投訴記錄。",
            "retrieved_count": 0,
            "retrieved_traces": [],
            "tag_validation": {"valid": True, "missing": [], "extra": []},
            "model_used": "",
        }

    if verbose:
        print(f"[RAG] 檢索到 {len(results)} 筆相似投訴")
        for r in results:
            print(f"  - {r['trace_id']} ({r['category']}) sim={r['similarity']:.3f}")

    # ── Step 3: Re-tag + Build merged vault ───────────────────
    context_str, merged_vault = _build_context(results)

    if verbose:
        print(f"[RAG] Merged vault 共 {len(merged_vault)} 個 tag")

    # ── Step 4: LLM ──────────────────────────────────────────
    role = os.getenv("USER_ROLE", "admin").lower()
    if role == "guest":
        prompt = RAG_GUEST_PROMPT_TEMPLATE.format(context=context_str, query=query)
    else:
        prompt = RAG_PROMPT_TEMPLATE.format(context=context_str, query=query)

    if verbose:
        print("[RAG] 呼叫 LLM ...")

    encoded_answer, model_used = call_llm(prompt, verbose=verbose)

    # ── Step 5: Validate tags ────────────────────────────────
    tag_check = decoder.validate_tags(context_str, encoded_answer)
    if verbose:
        if tag_check["extra"]:
            print(f"[RAG] ⚠️  LLM 產生了未知 tag: {tag_check['extra']}")
        if tag_check["missing"]:
            print(f"[RAG] ℹ️  LLM 省略了部分 tag: {tag_check['missing']}")

    # ── Step 6: Decode ───────────────────────────────────────
    if role == "guest":
        decoded_answer = encoded_answer
    else:
        decoded_answer = decoder.decode_rag(encoded_answer, merged_vault)

    # ── Step 7: Save summary ─────────────────────────────────
    trace_ids = [r["trace_id"] for r in results]
    for tid in trace_ids:
        save_summary(
            trace_id=tid,
            encoded_summary=encoded_answer,
            decoded_summary=decoded_answer,
            model_used=model_used,
            query_text=query,
        )

    if verbose:
        print("[RAG] ✅ 完成")

    return {
        "query": query,
        "encoded_answer": encoded_answer,
        "decoded_answer": decoded_answer,
        "retrieved_count": len(results),
        "retrieved_traces": trace_ids,
        "tag_validation": {
            "valid": tag_check["valid"],
            "missing": list(tag_check["missing"]),
            "extra": list(tag_check["extra"]),
        },
        "model_used": model_used,
    }


# ═══════════════════════════════════════════════════════════
# 便利函式：互動式查詢
# ═══════════════════════════════════════════════════════════

def interactive():
    """互動式 RAG 查詢（終端使用）"""
    print("=" * 60)
    print("📋 客戶投訴 RAG 查詢系統")
    print("輸入問題查詢相關投訴，輸入 'quit' 離開")
    print("=" * 60)

    while True:
        query = input("\n🔍 查詢: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("👋 再見！")
            break
        if not query:
            continue

        result = ask(query)
        print("\n" + "─" * 50)
        print("📝 LLM 回答（已還原個資）：")
        print(result["decoded_answer"])
        print("─" * 50)


if __name__ == "__main__":
    interactive()
