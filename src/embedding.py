"""
embedding.py — 本地中文 Embedding 模型
=====================================
使用 BAAI/bge-base-zh-v1.5，768 維度，中文語義搜尋表現優秀。
"""

import os
import numpy as np
from typing import Union

# ── 延遲載入模型（避免 import 就占 GPU 記憶體）───────────────
_model = None
_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")


def _get_model():
    """懶載入 sentence-transformers 模型"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[Embedding] 載入模型 {_MODEL_NAME} ...")
        _model = SentenceTransformer(_MODEL_NAME)
        print(f"[Embedding] 模型就緒 (dim={_model.get_sentence_embedding_dimension()})")
    return _model


def embed_text(text: str) -> list:
    """
    對單一文本做 embedding。
    BGE 模型建議對 query 加上 "为这个句子生成表示以用于检索相关文章：" 前綴，
    但對 document 端不加。這裡用於 document 端，不加前綴。
    """
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_query(query: str) -> list:
    """
    對查詢文本做 embedding（加 BGE 專用 query 前綴）。
    """
    model = _get_model()
    # BGE 官方建議的 query 前綴
    prefixed = f"为这个句子生成表示以用于检索相关文章：{query}"
    vec = model.encode(prefixed, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: list, show_progress: bool = True) -> list:
    """
    批次 embedding（用於一次性入庫 100 筆）。
    回傳: list of list[float]
    """
    model = _get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        batch_size=32,
    )
    return vecs.tolist()


def get_dimension() -> int:
    """回傳模型向量維度"""
    model = _get_model()
    return model.get_sentence_embedding_dimension()


# ═══════════════════════════════════════════════════════════
# LangChain 介面橋接
# ═══════════════════════════════════════════════════════════
try:
    from langchain_core.embeddings import Embeddings

    class CustomBGEEmbeddings(Embeddings):
        """
        將本地的 BGE 模型封裝成 LangChain 標準 Embeddings 介面。
        這樣 LangChain 的 PGVector 等元件就可以無縫呼叫。
        """
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            # Document 不加前綴，可批次處理
            return embed_batch(texts, show_progress=False)

        def embed_query(self, text: str) -> list[float]:
            # Query 會自動加上 BGE 專屬的前綴
            return embed_query(text)

except ImportError:
    pass
