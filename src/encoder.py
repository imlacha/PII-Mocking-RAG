"""
encoder.py — 去識別化引擎（雙次掃描版）
=========================================

掃描架構
--------
第一次掃描（Pass 1）— 正則盲掃
  不依賴任何先驗知識，純粹用正則偵測文本中「看起來像個資」的片段。
  涵蓋：手機、市話、信用卡號、台灣地址格式、姓名上下文。

第二次掃描（Pass 2）— 結構化欄位比對（本次新增）
  利用資料庫記錄中的已知欄位值（name / phone / address / credit_card）
  對第一次掃描後的「剩餘文本」再跑一遍。
  重點處理四類 Pass-1 容易漏掉的情境：

  [A] 姓名直接出現（無「我是」等上下文觸發詞）
  [B] 電話格式變體（無分隔符 / 不同分隔符）
  [C] 地址縮寫（省略縣市，只寫區以後）
  [D] 信用卡號無分隔符

最後做一次「殘留掃描」，確保所有來源的 PII 變體都被替換掉。
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

# ─── 選用：Presidio ──────────────────────────────────────────────────────────
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

# ─── 選用：Redis ─────────────────────────────────────────────────────────────
try:
    import os as _os
    import redis as _redis_lib
    _redis_host = _os.getenv("REDIS_HOST", "localhost")
    _redis_port = int(_os.getenv("REDIS_PORT", 6379))
    _redis_client = _redis_lib.Redis(host=_redis_host, port=_redis_port, db=0, decode_responses=True)
    _redis_client.ping()
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False
    _redis_client = None


# ═══════════════════════════════════════════════════════════
# 資料結構
# ═══════════════════════════════════════════════════════════

@dataclass
class PIIEntity:
    token:       str
    entity_type: str
    real_value:  str
    start:       int
    end:         int
    source:      str = "pass1"  # pass1 | pass2_name | pass2_phone | pass2_addr | pass2_card


@dataclass
class EncodeResult:
    trace_id:       str
    original_text:  str
    encoded_text:   str
    entities:       list = field(default_factory=list)      # list[PIIEntity]
    vault_snapshot: dict = field(default_factory=dict)      # token -> real_value
    pass2_findings: list = field(default_factory=list)      # list[dict]
    residual_hits:  list = field(default_factory=list)      # list[str]
    tag_list:       set  = field(default_factory=set)       # 所有 unique tag，如 {"[PERSON_1]", "[PHONE_1]"}


# ═══════════════════════════════════════════════════════════
# Vault
# ═══════════════════════════════════════════════════════════

class VaultBackend:
    DEFAULT_TTL = 1800

    def __init__(self):
        self._mem = {}

    def _key(self, trace_id, token):
        return f"vault:{trace_id}:{token}"

    def set(self, trace_id, token, real_value, ttl=DEFAULT_TTL):
        key = self._key(trace_id, token)
        if REDIS_AVAILABLE:
            _redis_client.setex(key, ttl, real_value)
        else:
            self._mem[key] = real_value

    def get(self, trace_id, token):
        key = self._key(trace_id, token)
        if REDIS_AVAILABLE:
            return _redis_client.get(key)
        return self._mem.get(key)

    def get_all(self, trace_id):
        if REDIS_AVAILABLE:
            keys = _redis_client.keys(f"vault:{trace_id}:*")
            result = {}
            for k in keys:
                token = k.split(":", 2)[2]
                val = _redis_client.get(k)
                if val:
                    result[token] = val
            return result
        prefix = f"vault:{trace_id}:"
        return {k[len(prefix):]: v for k, v in self._mem.items() if k.startswith(prefix)}

    def delete_session(self, trace_id):
        if REDIS_AVAILABLE:
            keys = _redis_client.keys(f"vault:{trace_id}:*")
            if keys:
                _redis_client.delete(*keys)
        else:
            for k in [k for k in self._mem if k.startswith(f"vault:{trace_id}:")]:
                del self._mem[k]


vault = VaultBackend()


# ═══════════════════════════════════════════════════════════
# Pass-1：正則盲掃
# ═══════════════════════════════════════════════════════════

_RE_PHONE_MOBILE   = re.compile(r"09\d{2}[\s\-]?\d{3}[\s\-]?\d{3}")
_RE_PHONE_LANDLINE = re.compile(r"0[2-8]\d[\s\-]?\d{4}[\s\-]?\d{4}")
_RE_CREDIT_CARD    = re.compile(r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b")
_RE_ADDRESS_TW     = re.compile(
    r"(?:台[北中南]市|新[北竹]市|桃園市|高雄市|基隆市|嘉義[市縣]"
    r"|屏東縣|彰化縣|[南北]投縣|雲林縣|花蓮縣|台東縣|宜蘭縣|澎湖縣|金門縣|連江縣)"
    r"[\u4e00-\u9fff\w\-號樓之]+"
)
_RE_NAME_CTX = re.compile(
    r"(?:我(?:是|叫)|客戶|帳號|姓名[是：:]?\s*)([^\s，。！？、]{2,4})"
)
_COMMON_SURNAMES = set("陳林黃張李王吳劉蔡楊許鄭謝洪葉邱廖賴徐周葛施蘇盧蔣何韓傅胡江高朱鍾羅柯孫曾游錢曹彭汪石田馬")


# ═══════════════════════════════════════════════════════════
# Pass-2：結構化欄位變體生成
# ═══════════════════════════════════════════════════════════

def _phone_variants(phone: str) -> list:
    clean = re.sub(r"[\s\-()]", "", phone)
    variants = {phone, clean}
    if re.match(r"^09\d{8}$", clean):
        variants.update([
            f"{clean[:4]}-{clean[4:7]}-{clean[7:]}",
            f"{clean[:4]} {clean[4:7]} {clean[7:]}",
            f"{clean[:4]}-{clean[4:]}",
        ])
    elif re.match(r"^0[2-8]\d{8}$", clean):
        variants.update([
            f"{clean[:2]}-{clean[2:]}",
            f"{clean[:2]}-{clean[2:6]}-{clean[6:]}",
        ])
    return [v for v in variants if v]


# 縣市全稱 → 常用縮稱
_CITY_ABBREV = {
    "台北市":"台北","新北市":"新北","台中市":"台中","台南市":"台南",
    "高雄市":"高雄","桃園市":"桃園","新竹市":"新竹","新竹縣":"新竹",
    "基隆市":"基隆","嘉義市":"嘉義","嘉義縣":"嘉義","屏東縣":"屏東",
    "彰化縣":"彰化","南投縣":"南投","雲林縣":"雲林","花蓮縣":"花蓮",
    "台東縣":"台東","宜蘭縣":"宜蘭","澎湖縣":"澎湖","苗栗縣":"苗栗",
    "金門縣":"金門","連江縣":"連江","台北縣":"台北",
}


def _address_variants(address: str) -> list:
    """
    產生地址的所有常見縮寫/變體。
    涵蓋：去縣市名、縣市縮稱、去縣市+去區鄉鎮、縮稱+區鄉鎮等組合。
    例：
      「新北市板橋區縣民大道二段7號」→ 板橋區縣民大道...、新北板橋區...、縣民大道...
      「台南市東區」→ 東區、台南東區
      「台東縣卑南鄉」→ 卑南鄉、台東卑南鄉
    """
    variants = {address}

    matched_city = None
    matched_abbr = None
    for full, abbr in _CITY_ABBREV.items():
        if address.startswith(full):
            matched_city = full
            matched_abbr = abbr
            break

    if matched_city:
        after_city = address[len(matched_city):]
        if after_city and len(after_city) >= 2:
            variants.add(after_city)
        abbr_full = matched_abbr + after_city
        if abbr_full != address:
            variants.add(abbr_full)
        m2 = re.match(r"^([一-鿿]{2,4}[區市鄉鎮])(.*)$", after_city)
        if m2:
            district = m2.group(1)
            rest     = m2.group(2)
            variants.add(district)
            variants.add(matched_abbr + district)
            if rest:
                variants.add(rest)
                variants.add(district + rest)
                variants.add(matched_abbr + district + rest)

    return sorted(
        [v for v in variants if v and len(v) >= 2],
        key=lambda x: -len(x)
    )


def _credit_card_variants(card: str) -> list:
    clean = re.sub(r"[\s\-]", "", card)
    return list({card, clean})


def _pass2_detect(text: str, record: dict, value_to_token: dict) -> list:
    """
    對文本做結構化欄位二次掃描。
    只回傳 Pass-1 尚未偵測到的新發現。
    """
    findings = []
    already_covered = set(value_to_token.keys())

    # [A] 姓名直接出現
    name = record.get("name", "").strip()
    if name and name not in already_covered and name in text:
        positions = [(m.start(), m.end()) for m in re.finditer(re.escape(name), text)]
        findings.append({
            "entity_type": "PERSON",
            "real_value":  name,
            "canonical":   name,
            "source":      "pass2_name",
            "positions":   positions,
        })

    # [B] 電話格式變體
    phone = record.get("phone", "").strip()
    if phone:
        for variant in _phone_variants(phone):
            if variant in already_covered or variant == phone:
                continue
            if variant in text:
                positions = [(m.start(), m.end()) for m in re.finditer(re.escape(variant), text)]
                if positions:
                    findings.append({
                        "entity_type": "PHONE",
                        "real_value":  variant,
                        "canonical":   phone,
                        "source":      "pass2_phone",
                        "positions":   positions,
                    })

    # [C] 地址縮寫
    address = record.get("address", "").strip()
    if address:
        for variant in _address_variants(address):
            if variant == address or variant in already_covered:
                continue
            if variant in text:
                positions = [(m.start(), m.end()) for m in re.finditer(re.escape(variant), text)]
                if positions:
                    findings.append({
                        "entity_type": "ADDRESS",
                        "real_value":  variant,
                        "canonical":   address,
                        "source":      "pass2_addr",
                        "positions":   positions,
                    })
                    break  # 取最長縮寫即停

    # [D] 信用卡無分隔符
    card = record.get("credit_card", "").strip()
    if card:
        for variant in _credit_card_variants(card):
            if variant == card or variant in already_covered:
                continue
            if variant in text:
                positions = [(m.start(), m.end()) for m in re.finditer(re.escape(variant), text)]
                if positions:
                    findings.append({
                        "entity_type": "CREDIT_CARD",
                        "real_value":  variant,
                        "canonical":   card,
                        "source":      "pass2_card",
                        "positions":   positions,
                    })

    return findings


def _residual_scan(encoded_text: str, record: dict, value_to_token: dict) -> list:
    """殘留掃描：在最終 encoded_text 中確認所有欄位變體都已被替換"""
    residual = []
    checks = []

    name = record.get("name", "")
    if name:
        checks.append((name, "name"))

    phone = record.get("phone", "")
    if phone:
        for v in _phone_variants(phone):
            checks.append((v, "phone_variant"))

    address = record.get("address", "")
    if address:
        for v in _address_variants(address):
            checks.append((v, "addr_variant"))

    card = record.get("credit_card", "")
    if card:
        for v in _credit_card_variants(card):
            checks.append((v, "card_variant"))

    for variant, label in checks:
        if variant and len(variant) >= 2 and variant in encoded_text:
            residual.append(f"{label}:{variant!r}")

    return residual


# ═══════════════════════════════════════════════════════════
# Pass-1 含 known_entities
# ═══════════════════════════════════════════════════════════

def _pass1_with_known(text: str, known: dict) -> list:
    """回傳 (entity_type, start, end, value, source)"""
    hits = []

    def _add(s, e, etype, v, src):
        for ps, pe, *_ in hits:
            if not (e <= ps or s >= pe):
                return
        hits.append((s, e, etype, v, src))

    # known_fields 精確匹配
    for etype, value in known.items():
        if not value:
            continue
        idx = 0
        while True:
            pos = text.find(value, idx)
            if pos == -1:
                break
            _add(pos, pos + len(value), etype, value, "known_field")
            idx = pos + 1

    # 正則掃描
    for pat, etype, src in [
        (_RE_CREDIT_CARD,    "CREDIT_CARD", "regex_card"),
        (_RE_PHONE_MOBILE,   "PHONE",       "regex_phone"),
        (_RE_PHONE_LANDLINE, "PHONE",       "regex_phone"),
        (_RE_ADDRESS_TW,     "ADDRESS",     "regex_addr"),
    ]:
        for m in pat.finditer(text):
            _add(m.start(), m.end(), etype, m.group(), src)

    # 姓名上下文
    for m in _RE_NAME_CTX.finditer(text):
        name = m.group(1)
        if 2 <= len(name) <= 4 and name[0] in _COMMON_SURNAMES:
            _add(m.start(1), m.end(1), "PERSON", name, "regex_name")

    hits.sort(key=lambda x: x[0])
    return [(etype, s, e, v, src) for s, e, etype, v, src in hits]


# ═══════════════════════════════════════════════════════════
# 主去識別化引擎
# ═══════════════════════════════════════════════════════════

class PrivacyEncoder:

    def __init__(self, use_presidio: bool = False):
        self.use_presidio = use_presidio and PRESIDIO_AVAILABLE
        if self.use_presidio:
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "zh", "model_name": "zh_core_web_sm"}],
            })
            self.analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
            print("[Encoder] Presidio 引擎已啟動")
        else:
            print(f"[Encoder] 雙次掃描模式（Presidio={'未安裝' if not PRESIDIO_AVAILABLE else '已停用'}）")

    def encode_record(self, record: dict, trace_id: Optional[str] = None) -> EncodeResult:
        """
        完整雙次掃描：Pass-1 正則盲掃 → Pass-2 結構化欄位比對 → 殘留掃描
        """
        tid  = trace_id or f"TRACE_{str(record.get('id', uuid.uuid4().hex[:4])).zfill(3)}"
        text = record.get("complaint_text", "")

        known = {
            "PERSON":      record.get("name", ""),
            "PHONE":       record.get("phone", ""),
            "ADDRESS":     record.get("address", ""),
            "CREDIT_CARD": record.get("credit_card", ""),
        }
        known_clean = {k: v for k, v in known.items() if v}

        # ── Pass-1 ───────────────────────────────────────────────────────────
        p1_detections = _pass1_with_known(text, known_clean)

        counters = {}
        value_to_token = {}   # 文本出現值 → token（用於替換）
        canonical_map  = {}   # token → 完整欄位值（用於 Vault / 還原）
        entities = []

        # 反查：欄位型別 → 完整欄位值（供後面修正 Vault 用）
        field_canonical = {
            "PERSON":      record.get("name", ""),
            "PHONE":       record.get("phone", ""),
            "ADDRESS":     record.get("address", ""),
            "CREDIT_CARD": record.get("credit_card", ""),
        }

        for etype, start, end, value, source in p1_detections:
            if value not in value_to_token:
                counters[etype] = counters.get(etype, 0) + 1
                token = f"[{etype}_{counters[etype]}]"
                value_to_token[value] = token
                # Vault 存完整欄位值：若 value 是欄位值的子串，用欄位值；否則用 value
                fc = field_canonical.get(etype, "")
                vault_val = fc if (fc and (value in fc or fc in value)) else value
                vault.set(tid, token, vault_val)
                canonical_map[token] = vault_val
            else:
                token = value_to_token[value]
            entities.append(PIIEntity(
                token=token, entity_type=etype,
                real_value=value, start=start, end=end, source=source
            ))

        # ── Pass-2 ───────────────────────────────────────────────────────────
        p2_findings = _pass2_detect(text, record, value_to_token)

        for finding in p2_findings:
            etype     = finding["entity_type"]
            real_val  = finding["real_value"]
            canonical = finding["canonical"]   # 永遠是完整欄位值

            # 同類型有既有代號則共用，否則新建
            existing_token = next(
                (tok for v, tok in value_to_token.items() if tok.startswith(f"[{etype}_")),
                None
            )
            if existing_token:
                token = existing_token
                # 用完整 canonical 更新 Vault（確保還原是完整地址）
                vault.set(tid, token, canonical)
                canonical_map[token] = canonical
            else:
                counters[etype] = counters.get(etype, 0) + 1
                token = f"[{etype}_{counters[etype]}]"
                vault.set(tid, token, canonical)
                canonical_map[token] = canonical

            if real_val not in value_to_token:
                value_to_token[real_val] = token
                for pos_s, pos_e in finding["positions"]:
                    entities.append(PIIEntity(
                        token=token, entity_type=etype,
                        real_value=real_val, start=pos_s, end=pos_e,
                        source=finding["source"]
                    ))

        # ── 文本替換（最長優先）──────────────────────────────────────────────
        encoded = text
        for real, token in sorted(value_to_token.items(), key=lambda x: -len(x[0])):
            encoded = encoded.replace(real, token)

        # ── Pass-3: NLP 安全網 (Presidio兜底) ──────────────────────────────────
        if self.use_presidio:
            nlp_results = self.analyzer.analyze(text=encoded, language="zh")
            for res in nlp_results:
                if res.score < 0.5:
                    continue
                real_val = encoded[res.start : res.end]
                # 排除已經替換的標籤，例如 [PERSON_1]
                if "[" in real_val and "]" in real_val:
                    continue

                etype_map = {
                    "PERSON": "PERSON", "PHONE_NUMBER": "PHONE",
                    "LOCATION": "ADDRESS", "EMAIL_ADDRESS": "EMAIL"
                }
                my_etype = etype_map.get(res.entity_type, res.entity_type)

                if real_val not in value_to_token:
                    counters[my_etype] = counters.get(my_etype, 0) + 1
                    token = f"[{my_etype}_{counters[my_etype]}]"
                    
                    value_to_token[real_val] = token
                    vault.set(tid, token, real_val)
                    canonical_map[token] = real_val
                    
                    entities.append(PIIEntity(
                        token=token, entity_type=my_etype,
                        real_value=real_val, start=res.start, end=res.end,
                        source="pass3_nlp"
                    ))

            # 針對 Pass-3 新發現的實體再做一次替換
            for real, token in sorted(value_to_token.items(), key=lambda x: -len(x[0])):
                if real in encoded:
                    encoded = encoded.replace(real, token)

        # ── 殘留掃描 ─────────────────────────────────────────────────────────
        residual = _residual_scan(encoded, record, value_to_token)

        # vault_snapshot 用 canonical_map（token → 完整欄位值），確保還原正確
        vault_snapshot = canonical_map

        return EncodeResult(
            trace_id=tid,
            original_text=text,
            encoded_text=encoded,
            entities=entities,
            vault_snapshot=vault_snapshot,
            pass2_findings=p2_findings,
            residual_hits=residual,
            tag_list=set(vault_snapshot.keys()),
        )

    def encode(self, text: str, known_entities: Optional[dict] = None,
               trace_id: Optional[str] = None) -> EncodeResult:
        """純文本去識別化（僅 Pass-1，無結構化欄位）"""
        tid = trace_id or f"TRACE_{uuid.uuid4().hex[:8].upper()}"
        detections = _pass1_with_known(text, known_entities or {})

        counters = {}
        value_to_token = {}
        entities = []

        for etype, start, end, value, source in detections:
            if value not in value_to_token:
                counters[etype] = counters.get(etype, 0) + 1
                token = f"[{etype}_{counters[etype]}]"
                value_to_token[value] = token
                vault.set(tid, token, value)
            else:
                token = value_to_token[value]
            entities.append(PIIEntity(
                token=token, entity_type=etype,
                real_value=value, start=start, end=end, source=source
            ))

        encoded = text
        for real, token in sorted(value_to_token.items(), key=lambda x: -len(x[0])):
            encoded = encoded.replace(real, token)

        vault_snapshot = {tok: real for real, tok in value_to_token.items()}
        return EncodeResult(
            trace_id=tid, original_text=text, encoded_text=encoded,
            entities=entities, vault_snapshot=vault_snapshot,
        )