"""
decoder.py — 去識別化還原引擎
============================
兩種模式：
  1. 單筆還原 decode()   — 用 trace_id 查 vault 還原
  2. RAG 還原 decode_rag() — 用 merged_vault 還原（處理跨 trace tag 碰撞）
"""

import re
from typing import Optional

from src.db import load_vault, load_vaults_batch
from src.encoder import vault as redis_vault   # Redis VaultBackend 實例


# ═══════════════════════════════════════════════════════════
# Tag 解析工具
# ═══════════════════════════════════════════════════════════

# 匹配 [TYPE_N] 或 [TYPE_N@xxx] 格式
_RE_TAG = re.compile(r"\[([A-Z_]+_\d+)(?:@(\w+))?\]")


def extract_tags(text: str) -> set:
    """提取文本中所有 [TAG_N] 或 [TAG_N@xxx] 標記"""
    return {m.group(0) for m in _RE_TAG.finditer(text)}


# ═══════════════════════════════════════════════════════════
# Re-tag：為 RAG Context 做全域唯一標記
# ═══════════════════════════════════════════════════════════

def retag_text(encoded_text: str, trace_id: str) -> str:
    """
    將 [PERSON_1] 轉為 [PERSON_1@001]，使 tag 在跨 trace 場景下全域唯一。
    trace_id 格式: TRACE_001 → 取末三碼 001 作為後綴
    """
    suffix = trace_id.replace("TRACE_", "")  # "001", "005", etc.

    def _add_suffix(match):
        tag_name = match.group(1)   # e.g. "PERSON_1"
        return f"[{tag_name}@{suffix}]"

    return _RE_TAG.sub(_add_suffix, encoded_text)


def build_merged_vault(trace_ids: list) -> dict:
    """
    為一組 trace_id 建立 merged vault（全域唯一 tag → real_value）。

    例：
      trace_ids = ["TRACE_001", "TRACE_005"]
      結果:
        {"[PERSON_1@001]": "陳志明", "[PHONE_1@001]": "0912...",
         "[PERSON_1@005]": "李美玲", "[ADDRESS_1@005]": "桃園市..."}
    """
    # 1. 先嘗試 Redis（快），再 fallback PostgreSQL
    merged = {}

    # 批次從 PostgreSQL 載入
    pg_vaults = load_vaults_batch(trace_ids)

    for tid in trace_ids:
        suffix = tid.replace("TRACE_", "")

        # 優先用 Redis vault
        vault_map = redis_vault.get_all(tid)

        # Fallback: PostgreSQL
        if not vault_map:
            vault_map = pg_vaults.get(tid, {})

        for token, real_value in vault_map.items():
            # token: "[PERSON_1]" → "[PERSON_1@001]"
            tag_name = token.strip("[]")
            global_tag = f"[{tag_name}@{suffix}]"
            merged[global_tag] = real_value

    return merged


# ═══════════════════════════════════════════════════════════
# Decoder
# ═══════════════════════════════════════════════════════════

class PrivacyDecoder:
    """去識別化文本還原引擎"""

    def decode(self, encoded_text: str, trace_id: str) -> str:
        """
        單筆還原：將含 [PERSON_1] 的文本還原為真實個資。
        查詢順序：Redis → PostgreSQL
        """
        # 1. Redis
        vault_map = redis_vault.get_all(trace_id)

        # 2. PostgreSQL fallback
        if not vault_map:
            vault_map = load_vault(trace_id)

        if not vault_map:
            return encoded_text   # 找不到 vault，原樣返回

        return self._replace(encoded_text, vault_map)

    def decode_rag(self, llm_output: str, merged_vault: dict) -> str:
        """
        RAG 還原：處理含 [TAG@xxx] 的 LLM 回答。
        merged_vault 由 build_merged_vault() 產生。
        """
        if not merged_vault:
            return llm_output
        return self._replace(llm_output, merged_vault)

    def validate_tags(self, input_text: str, output_text: str) -> dict:
        """
        驗證 LLM output 的 tag 完整性。
        回傳: {"valid": bool, "missing": set, "extra": set}
        """
        input_tags = extract_tags(input_text)
        output_tags = extract_tags(output_text)
        missing = input_tags - output_tags
        extra = output_tags - input_tags

        return {
            "valid": len(extra) == 0,   # 不允許出現 input 沒有的 tag
            "missing": missing,          # input 有但 output 沒有（可能被摘要省略）
            "extra": extra,              # output 多出來的（危險！可能是幻覺）
        }

    @staticmethod
    def _replace(text: str, vault_map: dict) -> str:
        """最長 token 優先替換，避免子串問題"""
        decoded = text
        for token, real_value in sorted(vault_map.items(), key=lambda x: -len(x[0])):
            decoded = decoded.replace(token, real_value)
        return decoded
