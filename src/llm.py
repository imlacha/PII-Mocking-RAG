"""
llm.py — LLM Fallback Chain
============================
自動切換策略：
  Gemini Key1 → Gemini Key2 → OpenAI (gpt-4o-mini)

每個 provider 失敗時自動 fallback 到下一個，並記錄使用了哪個。
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

# ── 載入所有 API Keys ────────────────────────────────────────
_GEMINI_KEYS = [
    os.getenv("GOOGLE_API_KEY", ""),
    os.getenv("GOOGLE_API_KEY2", ""),
]
_GEMINI_KEYS = [k for k in _GEMINI_KEYS if k]  # 過濾空值

_OPENAI_KEY = os.getenv("OPENAI_KEY", "")


# ═══════════════════════════════════════════════════════════
# Gemini Provider
# ═══════════════════════════════════════════════════════════

def _call_gemini(prompt: str, api_key: str, model_name: str = "gemini-2.0-flash") -> str:
    """用指定 API Key 呼叫 Gemini"""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


# ═══════════════════════════════════════════════════════════
# OpenAI Provider
# ═══════════════════════════════════════════════════════════

def _call_openai(prompt: str, model_name: str = "gpt-4o-mini") -> str:
    """用 OpenAI API 呼叫（便宜的 gpt-4o-mini）"""
    from openai import OpenAI
    client = OpenAI(api_key=_OPENAI_KEY)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000,
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════
# Fallback Chain
# ═══════════════════════════════════════════════════════════

def _is_quota_error(e: Exception) -> bool:
    """判斷是否為 quota/rate limit 錯誤"""
    msg = str(e).lower()
    return any(kw in msg for kw in ["429", "quota", "resourceexhausted", "rate_limit", "rate limit"])


def call_llm(prompt: str, verbose: bool = True) -> tuple:
    """
    LLM Fallback Chain：依序嘗試所有可用的 provider。

    回傳: (response_text, model_used)

    嘗試順序：
      1. Gemini Key 1 (gemini-2.0-flash)
      2. Gemini Key 2 (gemini-2.0-flash)
      3. OpenAI (gpt-4o-mini)
    """
    errors = []

    # ── 嘗試 Gemini Keys ──────────────────────────────────────
    for i, key in enumerate(_GEMINI_KEYS, 1):
        key_preview = f"{key[:8]}...{key[-4:]}"
        try:
            if verbose:
                print(f"[LLM] 嘗試 Gemini Key {i} ({key_preview})...")
            text = _call_gemini(prompt, api_key=key)
            if verbose:
                print(f"[LLM] ✅ Gemini Key {i} 成功")
            return text, f"gemini-2.0-flash (key{i})"
        except Exception as e:
            if _is_quota_error(e):
                if verbose:
                    print(f"[LLM] ⚠️  Gemini Key {i} 額度耗盡，切換下一個...")
                errors.append(f"Gemini Key {i}: quota exhausted")
            else:
                if verbose:
                    print(f"[LLM] ❌ Gemini Key {i} 錯誤: {str(e)[:100]}")
                errors.append(f"Gemini Key {i}: {str(e)[:100]}")

    # ── 嘗試 OpenAI ──────────────────────────────────────────
    if _OPENAI_KEY:
        try:
            if verbose:
                print(f"[LLM] 嘗試 OpenAI (gpt-4o-mini)...")
            text = _call_openai(prompt)
            if verbose:
                print(f"[LLM] ✅ OpenAI gpt-4o-mini 成功")
            return text, "gpt-4o-mini"
        except Exception as e:
            if verbose:
                print(f"[LLM] ❌ OpenAI 錯誤: {str(e)[:100]}")
            errors.append(f"OpenAI: {str(e)[:100]}")

    # ── 全部失敗 ──────────────────────────────────────────────
    error_summary = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(errors))
    raise RuntimeError(
        f"[LLM] 所有 provider 都失敗了：\n{error_summary}"
    )
