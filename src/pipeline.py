"""
pipeline.py — 資料入庫主流程
============================
讀取 CSV → Encode → 存 PostgreSQL + Redis → Embed → 存 pgvector

用法:
    python pipeline.py                     # 處理所有資料
    python pipeline.py --check             # 只檢查連線
    python pipeline.py --stats             # 顯示各表行數
"""

import csv
import sys
import time
import argparse
from pathlib import Path

from src.encoder import PrivacyEncoder
from src.embedding import embed_batch, embed_text
from src.db import (
    check_connection,
    save_customer,
    save_vault,
    get_row_counts,
    get_vectorstore,
)

# ── 設定 ─────────────────────────────────────────────────────
CSV_PATH = Path(__file__).parent.parent / "data" / "個資範例資料.csv"


def load_csv(path: str = None) -> list:
    """讀取 CSV，回傳 list of dict"""
    p = Path(path) if path else CSV_PATH
    with open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def run_pipeline(csv_path: str = None, verbose: bool = True):
    """
    完整入庫流程：
      1. 讀取 CSV
      2. PrivacyEncoder 去識別化
      3. 存 PostgreSQL (customers + pii_vault)
      4. 批次 embedding
      5. 存 pgvector (complaint_embeddings)
    """
    # ── 連線檢查 ─────────────────────────────────────────────
    print("[Pipeline] 檢查 PostgreSQL 連線 ...")
    if not check_connection():
        print("[Pipeline] ❌ PostgreSQL 連線失敗，請確認 Docker 容器是否啟動")
        sys.exit(1)
    print("[Pipeline] ✅ PostgreSQL 連線正常")

    # ── 讀取資料 ─────────────────────────────────────────────
    records = load_csv(csv_path)
    print(f"[Pipeline] 讀取 {len(records)} 筆資料")

    # ── Encode + 存資料庫 ────────────────────────────────────
    encoder = PrivacyEncoder(use_presidio=False)
    encoded_texts = []
    trace_ids = []
    categories = []

    t0 = time.time()
    for i, record in enumerate(records, 1):
        result = encoder.encode_record(record)
        tid = result.trace_id

        # 存客戶原始資料
        save_customer(tid, record, result.original_text)

        # 存 PII 對照表
        save_vault(tid, result.vault_snapshot)

        # 收集 encoded text 待批次 embedding
        encoded_texts.append(result.encoded_text)
        trace_ids.append(tid)
        categories.append(record.get("category", ""))

        if verbose and i % 20 == 0:
            print(f"  [Encode] {i}/{len(records)} 完成 ...")

    t_encode = time.time() - t0
    print(f"[Pipeline] Encode + 入庫完成 ({t_encode:.1f}s)")

    # ── 存入 LangChain PGVector ──────────────────────────────
    print(f"[Pipeline] 開始批次 Embedding 與存入 PGVector ({len(encoded_texts)} 筆) ...")
    t1 = time.time()
    vectorstore = get_vectorstore()
    if vectorstore is None:
        print("[Pipeline] ❌ 無法初始化 PGVector")
        sys.exit(1)
        
    from langchain_core.documents import Document
    docs = [
        Document(page_content=txt, metadata={"trace_id": tid, "category": cat}) 
        for tid, txt, cat in zip(trace_ids, encoded_texts, categories)
    ]
    # 傳入 ids 確保有重複 trace_id 時能覆寫 (Upsert)
    vectorstore.add_documents(docs, ids=trace_ids)
    t_embed = time.time() - t1
    print(f"[Pipeline] 向量入庫完成 ({t_embed:.1f}s)")

    # ── 完成 ─────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f"\n[Pipeline] ✅ 全部完成！耗時 {total_time:.1f}s")
    print(f"[Pipeline] 各表行數: {get_row_counts()}")


def main():
    parser = argparse.ArgumentParser(description="客戶投訴資料入庫管線")
    parser.add_argument("--check", action="store_true", help="只檢查資料庫連線")
    parser.add_argument("--stats", action="store_true", help="顯示各表行數")
    parser.add_argument("--csv", type=str, default=None, help="CSV 檔案路徑")
    args = parser.parse_args()

    if args.check:
        ok = check_connection()
        print("✅ 連線正常" if ok else "❌ 連線失敗")
        if ok:
            print(f"各表行數: {get_row_counts()}")
        sys.exit(0 if ok else 1)

    if args.stats:
        if check_connection():
            counts = get_row_counts()
            for table, cnt in counts.items():
                print(f"  {table}: {cnt} rows")
        else:
            print("❌ 連線失敗")
        sys.exit(0)

    run_pipeline(csv_path=args.csv)


if __name__ == "__main__":
    main()
