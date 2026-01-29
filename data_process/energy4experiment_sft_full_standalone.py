#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter, defaultdict


# -------------------------
# Common I/O
# -------------------------
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def copy_file(src: str, dst: str) -> None:
    Path(dst).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            buf = fsrc.read(1024 * 1024)
            if not buf:
                break
            fdst.write(buf)


def concat_jsonl(inputs: List[str], output: str) -> None:
    Path(output).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fout:
        for p in inputs:
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        json.loads(s)
                    except Exception:
                        continue
                    fout.write(s + "\n")


# -------------------------
# Step 1: convert_messages_to_tsformat_syn_en (standalone)
# -------------------------

PROMPT_TEMPLATE_OLD =  """
You are given weekly U.S. gasoline prices (OT):
- Context window (last 3 months, weekly OT):
<ts3m><ts3m/>
- Current month only (weekly dOT):
<tsdot><tsdot/>

Clarification about time-series inputs:
- The tags <ts3m>...</ts3m> and <tsdot>...</tsdot> contain learned latent vectors from a time-series encoder.
- They do NOT represent human-readable numerical values.
- You must NOT guess, fabricate, or interpret literal numeric values or directions from them.
- Only reason about high-level temporal patterns implicitly encoded within these embeddings.

Task:
1. Carefully analyze the time-series trends and reason step-by-step about plausible real-world causes.  
2. Write a short *reasoning paragraph* summarizing your interpretation of the data trend and potential driving factors.  
- This paragraph should sound like an analytical narrative (e.g., “The observed trend in the data indicates…”).  
- It should include your reasoning path that connects the observed pattern to possible events.  
3. Then, hypothesize up to 5 **REAL-TIME events** that occurred WITHIN THE CURRENT MONTH ONLY.  

Output format:
- First, write your reasoning paragraph (this is your thinking narrative).
- After that, write the final structured results after ⟪FINAL⟫ in strict JSON format:

⟪FINAL⟫
{
"hypotheses": [
    {
    "key": "NAME|ACTION|OBJECT|DIRECTION",
    }
]
}

Allowed token sets:
('NAME ∈ {market, eia, opec, opec_plus, united_states, api, nigera, saudi_arabia}',
'ACTION ∈ {price_change, report_release, production_cut, forecast_lower, forecast_lower, forecast_raise, production_raise, extend_cut, sanction_impose, purchase, release, export_ban_lift}',
'OBJECT ∈ {crude_oil, gasoline, price, production, natural_gas, inventory, diesel, export, jet_fuel, rig_count, brent, forecast}',
'DIRECTION ∈ {ambiguous, down, up}')

Rules:
- Do NOT fabricate tokens outside these sets.
- Do NOT repeat identical AAOD keys.
- The reasoning paragraph should be fluent and analytic, not a list of events.
""".lstrip("\n")

PAT_P = re.compile(r"\b\d{4}-\d{2}-\d{2}:\s*P\s*=\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
PAT_dP = re.compile(r"\b\d{4}-\d{2}-\d{2}:\s*dP\s*=\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def strip_empty_summary_fields(ans_text: Optional[str]) -> str:
    if not ans_text:
        return ""
    s = ans_text.strip()
    try:
        obj = json.loads(s)

        def walk(x):
            if isinstance(x, dict):
                return {k: walk(v) for k, v in x.items() if not (k == "summary" and v == "")}
            if isinstance(x, list):
                return [walk(v) for v in x]
            return x

        cleaned = walk(obj)
        return json.dumps(cleaned, ensure_ascii=False)
    except Exception:
        s = re.sub(r'\s*,\s*"summary"\s*:\s*""', "", s)
        s = re.sub(r'"summary"\s*:\s*""\s*,\s*', "", s)
        s = re.sub(r'"summary"\s*:\s*""\s*(?=\})', "", s)
        return s


def parse_series_from_user_text(txt: str) -> Tuple[List[float], List[float]]:
    p_vals = [float(m.group(1)) for m in PAT_P.finditer(txt or "")]
    dp_vals = [float(m.group(1)) for m in PAT_dP.finditer(txt or "")]
    return p_vals, dp_vals


def iter_input_json_or_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.readline()
    is_jsonl = False
    if head.strip():
        try:
            json.loads(head)
            is_jsonl = True
        except Exception:
            is_jsonl = False

    if is_jsonl:
        yield from iter_jsonl(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    yield it
        elif isinstance(obj, dict):
            yield obj
        else:
            raise ValueError("Unsupported JSON structure in input")


def get_role_content(messages: List[Dict[str, Any]], role: str) -> Optional[str]:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == role:
            return m.get("content")
    return None


def convert_messages_to_tsformat_syn_en(input_path: str, output_path: str, strict: bool = True, allow_empty: bool = False, max_lines: int = 0) -> Dict[str, int]:
    out_rows = []
    total_in = total_out = skipped = 0
    for obj in iter_input_json_or_jsonl(input_path):
        total_in += 1
        if not isinstance(obj, dict) or "messages" not in obj or not isinstance(obj["messages"], list):
            skipped += 1
            continue
        user_text = get_role_content(obj["messages"], "user") or ""
        asst_text = get_role_content(obj["messages"], "assistant") or ""

        p_vals, dp_vals = parse_series_from_user_text(user_text)
        rec = {
            "prompt": PROMPT_TEMPLATE_OLD,
            "answer": strip_empty_summary_fields(asst_text),
            "ts3m": {"vals": p_vals, "mask": [1] * len(p_vals)},
            "tsdot": {"vals": dp_vals, "mask": [1] * len(dp_vals)},
        }

        has_p = len(p_vals) > 0
        has_dp = len(dp_vals) > 0
        if strict:
            ok = has_p and has_dp
        else:
            ok = (has_p or has_dp) or allow_empty
        if not ok:
            skipped += 1
            continue
        if (not has_p or not has_dp) and not allow_empty:
            skipped += 1
            continue

        out_rows.append(rec)
        total_out += 1
        if max_lines and total_out >= max_lines:
            break

    write_jsonl(output_path, out_rows)
    return {"in": total_in, "out": total_out, "skipped": skipped}


# -------------------------
# Step 2: make_monthly_windows_with_answer (standalone)
# -------------------------
_FMT_TRIES = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
]


def parse_date_safe(v: Any) -> Optional[date]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        parts = s.replace("/", "-").split("-")
        if len(parts) >= 3 and parts[0].isdigit():
            y = int(parts[0]); m = int(parts[1]); d = int(parts[2])
            return date(y, m, d)
    except Exception:
        pass
    for fmt in _FMT_TRIES:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    return None


def ym_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def iso_year_week(d: date) -> Tuple[int, int]:
    y, w, _ = d.isocalendar()
    return int(y), int(w)


def read_prices_from_csv(path: str, date_col: str = "date", ot_col: str = "OT") -> List[Tuple[date, float]]:
    rows: List[Tuple[date, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            d = parse_date_safe(r.get(date_col))
            if d is None:
                continue
            raw = r.get(ot_col)
            if raw is None or str(raw).strip() == "":
                continue
            try:
                p = float(raw)
            except Exception:
                continue
            rows.append((d, p))
    rows.sort(key=lambda x: x[0])
    return rows


def load_event_rows(path: Optional[str], date_field: str = "date") -> List[Tuple[date, str]]:
    rows: List[Tuple[date, str]] = []
    if not path:
        return rows
    for obj in iter_jsonl(path):
        d = parse_date_safe(obj.get(date_field))
        if d is None:
            continue
        kw = obj.get("keyword")
        if not kw:
            continue
        kw = str(kw).strip()
        if not kw:
            continue
        rows.append((d, kw))
    rows.sort(key=lambda x: x[0])
    return rows


def compress_to_weekly(prices: List[Tuple[date, float]]) -> List[Tuple[Tuple[int, int], date, float]]:
    last_by_week: Dict[Tuple[int, int], Tuple[date, float]] = {}
    for d, p in prices:
        yw = iso_year_week(d)
        prev = last_by_week.get(yw)
        if prev is None or d >= prev[0]:
            last_by_week[yw] = (d, p)
    rows = [(yw, dp[0], dp[1]) for yw, dp in last_by_week.items()]
    rows.sort(key=lambda x: (x[0][0], x[0][1]))
    return rows


def month_span(min_d: date, max_d: date) -> List[str]:
    ys, ms = min_d.year, min_d.month
    ye, me = max_d.year, max_d.month
    y, m = ys, ms
    out = []
    while (y < ye) or (y == ye and m <= me):
        out.append(f"{y:04d}-{m:02d}")
        if m == 12:
            y += 1; m = 1
        else:
            m += 1
    return out


def build_month_index_by_week(weekly_rows) -> Dict[str, List[int]]:
    idx = defaultdict(list)
    for i, (_, d, _) in enumerate(weekly_rows):
        idx[ym_key(d)].append(i)
    return idx


def month_last_week_index(weekly_rows, ym: str, month_idx) -> Optional[int]:
    lst = month_idx.get(ym)
    if not lst:
        return None
    return lst[-1]


def take_ts3m(weekly_rows, end_index: int, length: int) -> List[float]:
    start = max(0, end_index - length + 1)
    return [weekly_rows[i][2] for i in range(start, end_index + 1)]


def month_dP_series(weekly_rows, ym: str, month_idx) -> List[float]:
    idxs = month_idx.get(ym, [])
    out: List[float] = []
    for i in idxs:
        p_t = weekly_rows[i][2]
        if i - 1 >= 0:
            p_tm1 = weekly_rows[i - 1][2]
            out.append(p_t - p_tm1)
    return out


def select_month_keywords(event_rows: List[Tuple[date, str]], ym: str, topk: int) -> List[str]:
    items = [kw for (d, kw) in event_rows if ym_key(d) == ym]
    if not items:
        return []
    cnt = Counter(items)
    sorted_items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in sorted_items[:max(0, topk)]]


def build_answer_str(keywords: List[str]) -> str:
    return json.dumps({"hypotheses": [{"key": k} for k in keywords]}, ensure_ascii=False)


def make_monthly_windows_with_answer(
    price_csv: str,
    events_jsonl: Optional[str],
    answer_topk: int,
    output_path: str,
    date_col: str = "date",
    ot_col: str = "OT",
    ts3m_len: int = 13,
    min_ts3m_len: int = 8,
    allow_short: bool = False,
    events_date_col: str = "date",
) -> Dict[str, int]:
    price_rows = read_prices_from_csv(price_csv, date_col=date_col, ot_col=ot_col)
    if not price_rows:
        raise ValueError("No valid (date, OT) rows parsed from CSV.")
    weekly = compress_to_weekly(price_rows)
    min_d, max_d = weekly[0][1], weekly[-1][1]
    months = month_span(min_d, max_d)
    month_idx = build_month_index_by_week(weekly)

    event_rows = load_event_rows(events_jsonl, events_date_col) if events_jsonl else []

    out_rows: List[Dict[str, Any]] = []
    produced = 0
    skipped_short = 0
    for ym in months:
        end_idx = month_last_week_index(weekly, ym, month_idx)
        if end_idx is None:
            continue
        ts3m_vals = take_ts3m(weekly, end_idx, ts3m_len)
        if len(ts3m_vals) < min_ts3m_len and not allow_short:
            skipped_short += 1
            continue
        tsdot_vals = month_dP_series(weekly, ym, month_idx)
        keywords = select_month_keywords(event_rows, ym, answer_topk) if event_rows else []
        answer_str = build_answer_str(keywords) if keywords else ""
        out_rows.append(
            {
                "prompt": PROMPT_TEMPLATE_OLD,
                "answer": answer_str,
                "ts3m": {"vals": ts3m_vals, "mask": [1] * len(ts3m_vals)},
                "tsdot": {"vals": tsdot_vals, "mask": [1] * len(tsdot_vals)},
            }
        )
        produced += 1

    write_jsonl(output_path, out_rows)
    return {"months_emitted": produced, "months_skipped_short": skipped_short}


# -------------------------
# Step 3: filter_keep_up_any_triple (standalone)
# -------------------------
def _norm_token(t: Any) -> str:
    return re.sub(r"\s+", "_", (t or "").strip().lower())


def _parse_answer(answer_obj: Any) -> Tuple[List[Tuple[str, str, str, str, str]], str, Optional[dict]]:
    parsed = None
    ctype = "other"
    if isinstance(answer_obj, str):
        try:
            parsed = json.loads(answer_obj)
            ctype = "string"
        except Exception:
            return [], ctype, None
    elif isinstance(answer_obj, dict):
        parsed = answer_obj
        ctype = "dict"
    else:
        return [], ctype, None

    keys: List[Tuple[str, str, str, str, str]] = []
    if not parsed or "hypotheses" not in parsed or not isinstance(parsed["hypotheses"], list):
        return [], ctype, parsed
    for h in parsed["hypotheses"]:
        if not isinstance(h, dict):
            continue
        k = h.get("key")
        if not k:
            continue
        parts = [_norm_token(p) for p in str(k).split("|")]
        parts += [""] * (4 - len(parts))
        n, a, o, d = parts[:4]
        keys.append((n, a, o, d, str(k)))
    return keys, ctype, parsed


def _rebuild_answer(new_keys: List[Tuple[str, str, str, str, str]], ctype: str) -> Any:
    new_hypos = [{"key": rk} for (_, _, _, _, rk) in new_keys]
    out = {"hypotheses": new_hypos}
    if ctype == "dict":
        return out
    return json.dumps(out, ensure_ascii=False)


def filter_keep_up_any_triple(input_path: str, output_path: str) -> Dict[str, int]:
    modified = 0
    total = 0
    removed_total = 0
    out_rows: List[Dict[str, Any]] = []
    for ln, obj in enumerate(iter_jsonl(input_path), 1):
        total += 1
        keys, ctype, parsed = _parse_answer(obj.get("answer"))
        if not keys or parsed is None:
            out_rows.append(obj)
            continue

        triple_dirs = defaultdict(set)
        for (n, a, o, d, _rk) in keys:
            triple_dirs[(n, a, o)].add(d)
        conflicts = {t for t, dirs in triple_dirs.items() if "up" in dirs and "down" in dirs}
        if not conflicts:
            out_rows.append(obj)
            continue

        new_keys = []
        removed_here = 0
        for (n, a, o, d, rk) in keys:
            if (n, a, o) in conflicts:
                if d == "up":
                    new_keys.append((n, a, o, d, rk))
                else:
                    removed_here += 1
            else:
                new_keys.append((n, a, o, d, rk))

        seen = set()
        deduped = []
        for item in new_keys:
            rk = item[4]
            if rk in seen:
                removed_here += 1
                continue
            seen.add(rk)
            deduped.append(item)
        new_keys = deduped

        if removed_here > 0:
            modified += 1
            removed_total += removed_here
            obj["answer"] = _rebuild_answer(new_keys, ctype)
        out_rows.append(obj)

    write_jsonl(output_path, out_rows)
    return {"total": total, "modified_lines": modified, "removed_keys": removed_total}


# -------------------------
# Step 4: filter_empty_answer (standalone)
# -------------------------
def filter_empty_answer(input_path: str, output_path: str) -> Dict[str, int]:
    kept = removed = 0
    out_rows: List[Dict[str, Any]] = []
    for obj in iter_jsonl(input_path):
        ans = obj.get("answer", "")
        if not isinstance(ans, str) or ans.strip() == "":
            removed += 1
            continue
        out_rows.append(obj)
        kept += 1
    write_jsonl(output_path, out_rows)
    return {"kept": kept, "removed": removed}


# -------------------------
# Step 5: batch_reason_concurrent_en (standalone)
# -------------------------
SYSTEM_MSG = (
    """
You are a domain analyst specializing in crude oil markets. 
Your task is to write a single hypothetical explanatory paragraph for two short time-series: 
(1) a 3-month weekly series (OT) and 
(2) a current-month weekly series (dOT).

Your writing must be strictly hypothetical. 
You never assert the AAOD events as facts. 
Every event reference must use one of these hedge phrases:
  - “would be consistent with”
  - “could reflect”
  - “a plausible explanation is”
  - “may be explained by”
  - “one candidate is”

Rules:
- Describe only the observed patterns in OT and dOT (no numbers or dates).
- Then justify **each** AAOD event separately: if AAOD lists N events, you must propose N hypothetical “actor + action” candidates, each tied to some specific pattern feature.
- Each event mention must be explicit and not merged with others.
- Events must follow “actor + action” style (e.g., “OPEC+ production guidance”, “United States inventory report”).
- End with a concrete physical-market mechanism (inventories, refinery runs, seasonal demand, logistics constraints, etc.).
- One paragraph only; no lists, no meta-language.
"""
)

USER_TEMPLATE = """
### ROLE
You generate synthetic reasoning data for supervised fine-tuning. 
Your explanation must hypothetically justify the patterns using the same number of candidate events as listed in AAOD.

### INPUT
- Last 3 months (weekly OT): {ts3m_array}
- Current month (weekly dOT): {tsdot_array}

### AAOD EVENTS (ground truth; do NOT state them as facts)
{answer_json}

### TASK
Write exactly one paragraph with this structure:

1) Describe the OT pattern qualitatively (trend, turning point, volatility) without numbers or dates.
2) Describe the dOT pattern qualitatively (latest shift, sign streak, volatility) without numbers or dates.
3) For **each** AAOD event, generate one hypothetical “actor + action” candidate using ONLY the allowed hedge phrases.  
   - The number of hypothetical events must match the number of AAOD events.  
   - Each candidate must be explicitly separated in the text (e.g., with commas or semicolons within the same paragraph).  
4) Optionally tie some candidates to specific dOT features (e.g., late-month stabilization, mid-month dip).
5) End with a concrete physical-market mechanism (e.g., inventories, refinery runs, seasonal consumption, transportation bottlenecks and so on).

Constraints:
- All events are hypothetical; never assert them as facts.
- No lists or bullets; one paragraph only.

"""


def build_reason_messages(ts3m_vals, tsdot_vals, answer_json_str: str) -> List[Dict[str, str]]:
    ts3m_array = json.dumps(ts3m_vals, ensure_ascii=False)
    tsdot_array = json.dumps(tsdot_vals, ensure_ascii=False)
    user = USER_TEMPLATE.format(ts3m_array=ts3m_array, tsdot_array=tsdot_array, answer_json=answer_json_str.strip())
    return [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": user}]


def extract_reason_only(text: str) -> str:
    if not isinstance(text, str):
        return "Reason: (fallback) Unable to generate explanation."
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
    for mk in ["⟪FINAL⟫", "<answer_json>", "</answer_json>", "{", "<|im_end|>"]:
        if mk in t:
            t = t.split(mk, 1)[0].strip()
    t = " ".join(t.split())
    if len(t) < 30:
        t += " The patterns indicate a meaningful short-term shift consistent with the events."
    t = t[:1400]
    if not t.endswith((".", "!", "?")):
        t += "."
    return t


def call_openai_reason(messages: List[Dict[str, str]], model: str, temperature: float = 0.2, max_output_tokens: int = 220, seed: int = 7) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
        seed=seed,
    )
    raw = resp.choices[0].message.content if resp.choices else ""
    return extract_reason_only(raw)


def process_one_row(idx: int, row: Dict[str, Any], model: str) -> Tuple[int, Dict[str, Any], str]:
    gold_answer_str = row.get("answer", "")
    ts3m_vals = (row.get("ts3m") or {}).get("vals", [])
    tsdot_vals = (row.get("tsdot") or {}).get("vals", [])
    messages = build_reason_messages(ts3m_vals, tsdot_vals, str(gold_answer_str))
    try:
        reason = call_openai_reason(messages, model=model)
    except Exception as e:
        print(f"[API error idx={idx}] {e}")
        reason = "Reason: (fallback) Unable to generate explanation due to API error."
    answer_out = f"{reason}\n</think>\n⟪FINAL⟫\n{str(gold_answer_str).strip()}\n"
    new_row = {"prompt": row.get("prompt", ""), "answer": answer_out, "ts3m": row.get("ts3m"), "tsdot": row.get("tsdot")}
    return idx, new_row, reason


def batch_reason_concurrent(input_path: str, output_path: str, model: str = "gpt-4o-mini", workers: int = 8, sleep: float = 0.0) -> Dict[str, int]:
    rows = list(iter_jsonl(input_path))
    n = len(rows)
    out_rows: List[Optional[Dict[str, Any]]] = [None] * n
    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one_row, idx, row, model) for idx, row in enumerate(rows)]
        for fut in as_completed(futures):
            idx, new_row, reason = fut.result()
            out_rows[idx] = new_row
            finished += 1
            if sleep > 0:
                time.sleep(sleep)
            if finished % 50 == 0:
                print(f"[progress] finished {finished}/{n}; last reason preview: {reason[:120]}...")
    write_jsonl(output_path, [r for r in out_rows if r is not None])
    return {"total": n, "written": n}


# -------------------------
# Step 6/7: modify_prompts (standalone)
# -------------------------
PROMPT_TEMPLATE_NEW = """
You are given weekly U.S. gasoline prices (OT):
- Context window (last 3 months, weekly OT):
<ts3m><ts3m/>
- Current month only (weekly dOT):
<tsdot><tsdot/>

Clarification about time-series inputs:
- The tags <ts3m>...</ts3m> and <tsdot>...</tsdot> contain learned latent vectors from a time-series encoder.
- They do NOT represent human-readable numerical values.
- You must NOT guess, fabricate, or interpret literal numeric values or directions from them.
- Only reason about high-level temporal patterns implicitly encoded within these embeddings.

Task:
1. Carefully analyze the time-series trends and reason step-by-step about plausible real-world causes.  
2. Write a short *reasoning paragraph* summarizing your interpretation of the data trend and potential driving factors.  
- This paragraph should sound like an analytical narrative (e.g., “The observed trend in the data indicates…”).  
- It should include your reasoning path that connects the observed pattern to possible events.  
3. Then, hypothesize up to 5 **REAL-TIME events** that occurred WITHIN THE CURRENT MONTH ONLY.  

Output format:
- First, write your reasoning paragraph (this is your thinking narrative).
- After that, write the final structured results after ⟪FINAL⟫ in strict JSON format:

⟪FINAL⟫
{
"hypotheses": [
    {
    "key": "NAME|ACTION|OBJECT|DIRECTION",
    }
]
}

Allowed token sets:
('NAME ∈ {market, eia, opec, opec_plus, united_states, api, nigera, saudi_arabia}',
'ACTION ∈ {price_change, report_release, production_cut, forecast_lower, forecast_lower, forecast_raise, production_raise, extend_cut, sanction_impose, purchase, release, export_ban_lift}',
'OBJECT ∈ {crude_oil, gasoline, price, production, natural_gas, inventory, diesel, export, jet_fuel, rig_count, brent, forecast}',
'DIRECTION ∈ {ambiguous, down, up}')

Rules:
- Do NOT fabricate tokens outside these sets.
- Do NOT repeat identical AAOD keys.
- The reasoning paragraph should be fluent and analytic, not a list of events.
""".lstrip("\n")


def modify_prompts(input_path: str, output_path: str) -> Dict[str, int]:
    rows = list(iter_jsonl(input_path))
    for item in rows:
        if isinstance(item, dict) and "prompt" in item:
            item["prompt"] = PROMPT_TEMPLATE_NEW
    write_jsonl(output_path, rows)
    return {"total": len(rows)}


def replace_think_tag_inplace(path: str) -> None:
    tmp = str(Path(path).with_suffix(Path(path).suffix + ".tmp"))
    out_rows: List[Dict[str, Any]] = []
    for obj in iter_jsonl(path):
        ans = obj.get("answer")
        if isinstance(ans, str) and ans:
            ans2 = (
                ans.replace("\r\n</think>\r\n", " ")
                .replace("\n</think>\n", " ")
                .replace("\r\n</think>\n", " ")
                .replace("\n</think>\r\n", " ")
            )
            obj["answer"] = ans2
        out_rows.append(obj)
    write_jsonl(tmp, out_rows)
    copy_file(tmp, path)
    try:
        os.remove(tmp)
    except Exception:
        pass


# -------------------------
# CLI / Orchestration
# -------------------------
def repo_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def parse_args() -> argparse.Namespace:
    root = repo_root()
    ap = argparse.ArgumentParser(description="Standalone full SFT pipeline (energy4expriment.md 104-239).")

    # (104-110)
    ap.add_argument("--syn-input", default=f"{root}/syndata_en/syndata_v1/sft_windows.jsonl")
    ap.add_argument("--syn-output", default=f"{root}/dataset4sft_en/sft_windows_syn.jsonl")
    ap.add_argument("--syn-strict", action="store_true", default=True)
    ap.add_argument("--no-syn-strict", action="store_false", dest="syn_strict")

    # (112-118)/(121-138)
    ap.add_argument("--price-csv", default=f"{root}/dataset/energy_noevent.csv")
    ap.add_argument("--events-train", default=f"{root}/data4dedup_en/events_dedup_min5topk1000_before2022_train.jsonl")
    ap.add_argument("--events-vali", default=f"{root}/data4dedup_en/events_dedup_min5topk1000_during2022_2023_vail.jsonl")
    ap.add_argument("--events-test-topk30", default=f"{root}/data4dedup_en_old/events_dedup_min5topk1000_after2023_test.jsonl")
    ap.add_argument("--events-test-topk1000", default=f"{root}/data4dedup_en_old/events_dedup_min1topk1000_after2023_test_stat.jsonl")
    ap.add_argument("--answer-topk-train", type=int, default=5)
    ap.add_argument("--answer-topk-vali", type=int, default=5)
    ap.add_argument("--answer-topk-test-topk30", type=int, default=30)
    ap.add_argument("--answer-topk-test-topk1000", type=int, default=1000)

    ap.add_argument("--out-train", default=f"{root}/dataset4sft_en/monthly_windows_with_answer_train_new_server.jsonl")
    ap.add_argument("--out-vali", default=f"{root}/dataset4sft_en/monthly_windows_with_answer_vail_new_server.jsonl")
    ap.add_argument("--out-test-topk30", default=f"{root}/dataset4sft_en/monthly_windows_with_answer_test_new_server_v2.jsonl")
    ap.add_argument("--out-test-topk1000", default=f"{root}/dataset4sft_en/monthly_windows_with_answer_test_new_server_new.jsonl")

    # (143-151)
    ap.add_argument("--out-train-direction", default=f"{root}/dataset4sft_en/monthly_windows_with_answer_train_direction_new_server.jsonl")
    ap.add_argument("--syn-output-direction", default=f"{root}/dataset4sft_en/sft_windows_syn_direction_new_serve.jsonl")

    # (156-171)
    ap.add_argument("--out-train-direction-nonempty", default=f"{root}/dataset4sft_en/monthly_windows_with_answer_train_direction_empty_new_server.jsonl")
    ap.add_argument("--out-vali-nonempty", default=f"{root}/dataset4sft_en/monthly_windows_with_answer_vail_empty_new_server.jsonl")
    ap.add_argument("--test-empty-input", default=f"{root}/dataset4sft_en/monthly_windows_with_answer_test_new_server.jsonl")
    ap.add_argument("--out-test-nonempty", default=f"{root}/dataset4sft_en/monthly_windows_with_answer_test_empty_new_server.jsonl")
    ap.add_argument("--syn-output-direction-nonempty", default=f"{root}/dataset4sft_en/sft_windows_syn_direction_empty_new_serve.jsonl")

    ap.add_argument("--sync-test-topk30-to-test-empty-input", action="store_true", default=True)
    ap.add_argument("--no-sync-test-topk30-to-test-empty-input", action="store_false", dest="sync_test_topk30_to_test_empty_input")

    # (210-223) reason
    ap.add_argument("--reason-model", default="gpt-4o-mini")
    ap.add_argument("--reason-workers", type=int, default=8)
    ap.add_argument("--reason-sleep", type=float, default=0.0)
    ap.add_argument("--openai-api-key", default="")
    ap.add_argument("--openai-base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--out-train-reason", default=f"{root}/dataset4sft_en/monthly_windows_with_answer_train_direction_empty_reason_new_server.jsonl")
    ap.add_argument("--out-syn-reason", default=f"{root}/dataset4sft_en/sft_windows_syn_direction_empty_reason_new_serve.jsonl")
    ap.add_argument("--out-test-reason", default=f"{root}/dataset4sft_en/monthly_windows_with_answer_test_empty_reason_new_server.jsonl")

    # merge
    ap.add_argument("--merge-train-jsonl", default=None, help="Merge input: train JSONL path (default: --out-train-reason)")
    ap.add_argument("--merge-syn-jsonl", default=None, help="Merge input: synthetic JSONL path (default: --out-syn-reason)")
    ap.add_argument("--merge-output", default=f"{root}/dataset4sft_en/new_server_fintune.jsonl")

    # modify prompt
    ap.add_argument("--merged-prompt-output", default=f"{root}/dataset4sft_en/new_server_fintune_prompt_v2.jsonl")
    ap.add_argument("--modify-test-prompt-inplace", action="store_true", default=True)
    ap.add_argument("--no-modify-test-prompt-inplace", action="store_false", dest="modify_test_prompt_inplace")
    ap.add_argument("--test-prompt-output", default=None)

    # strip think
    ap.add_argument("--strip-think-tag", action="store_true", default=True)
    ap.add_argument("--no-strip-think-tag", action="store_false", dest="strip_think_tag")

    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # env for OpenAI
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url

    def _print(step: str, detail: str) -> None:
        print(f"[{step}] {detail}")

    # Step 1
    _print("Step1", f"convert syn {args.syn_input} -> {args.syn_output} (strict={args.syn_strict})")
    if not args.dry_run:
        convert_messages_to_tsformat_syn_en(args.syn_input, args.syn_output, strict=args.syn_strict)

    # Step 2: monthly windows
    _print("Step2", f"monthly train {args.price_csv} + {args.events_train} -> {args.out_train} (topk={args.answer_topk_train})")
    if not args.dry_run:
        make_monthly_windows_with_answer(args.price_csv, args.events_train, args.answer_topk_train, args.out_train)

    _print("Step3", f"monthly vali {args.price_csv} + {args.events_vali} -> {args.out_vali} (topk={args.answer_topk_vali})")
    if not args.dry_run:
        make_monthly_windows_with_answer(args.price_csv, args.events_vali, args.answer_topk_vali, args.out_vali)

    _print("Step4", f"monthly test(topk30) {args.price_csv} + {args.events_test_topk30} -> {args.out_test_topk30} (topk={args.answer_topk_test_topk30})")
    if not args.dry_run:
        make_monthly_windows_with_answer(args.price_csv, args.events_test_topk30, args.answer_topk_test_topk30, args.out_test_topk30)

    if args.sync_test_topk30_to_test_empty_input:
        _print("Step4b", f"copy {args.out_test_topk30} -> {args.test_empty_input}")
        if not args.dry_run:
            copy_file(args.out_test_topk30, args.test_empty_input)

    _print("Step5", f"monthly test_stat(topk1000) {args.price_csv} + {args.events_test_topk1000} -> {args.out_test_topk1000} (topk={args.answer_topk_test_topk1000})")
    if not args.dry_run:
        make_monthly_windows_with_answer(args.price_csv, args.events_test_topk1000, args.answer_topk_test_topk1000, args.out_test_topk1000)

    # Step 6: keep_up_any_triple
    _print("Step6", f"keep_up_any_triple train {args.out_train} -> {args.out_train_direction}")
    if not args.dry_run:
        filter_keep_up_any_triple(args.out_train, args.out_train_direction)

    _print("Step7", f"keep_up_any_triple syn {args.syn_output} -> {args.syn_output_direction}")
    if not args.dry_run:
        filter_keep_up_any_triple(args.syn_output, args.syn_output_direction)

    # Step 8: filter empty
    _print("Step8", f"filter_empty train {args.out_train_direction} -> {args.out_train_direction_nonempty}")
    if not args.dry_run:
        filter_empty_answer(args.out_train_direction, args.out_train_direction_nonempty)

    _print("Step9", f"filter_empty vali {args.out_vali} -> {args.out_vali_nonempty}")
    if not args.dry_run:
        filter_empty_answer(args.out_vali, args.out_vali_nonempty)

    _print("Step10", f"filter_empty test {args.test_empty_input} -> {args.out_test_nonempty}")
    if not args.dry_run:
        filter_empty_answer(args.test_empty_input, args.out_test_nonempty)

    _print("Step11", f"filter_empty syn {args.syn_output_direction} -> {args.syn_output_direction_nonempty}")
    if not args.dry_run:
        filter_empty_answer(args.syn_output_direction, args.syn_output_direction_nonempty)

    # Step 12: reason
    _print("Step12", f"reason(train) {args.out_train_direction_nonempty} -> {args.out_train_reason} (workers={args.reason_workers}, model={args.reason_model})")
    if not args.dry_run:
        batch_reason_concurrent(args.out_train_direction_nonempty, args.out_train_reason, model=args.reason_model, workers=args.reason_workers, sleep=args.reason_sleep)

    _print("Step13", f"reason(syn) {args.syn_output_direction_nonempty} -> {args.out_syn_reason}")
    if not args.dry_run:
        batch_reason_concurrent(args.syn_output_direction_nonempty, args.out_syn_reason, model=args.reason_model, workers=args.reason_workers, sleep=args.reason_sleep)

    _print("Step14", f"reason(test) {args.out_test_nonempty} -> {args.out_test_reason}")
    if not args.dry_run:
        batch_reason_concurrent(args.out_test_nonempty, args.out_test_reason, model=args.reason_model, workers=args.reason_workers, sleep=args.reason_sleep)

    # Step 15: merge
    merge_train = args.merge_train_jsonl or args.out_train_reason
    merge_syn = args.merge_syn_jsonl or args.out_syn_reason
    _print("Step15", f"merge train={merge_train} syn={merge_syn} -> {args.merge_output}")
    if not args.dry_run:
        concat_jsonl([merge_train, merge_syn], args.merge_output)

    # Step 16: strip think tag (optional)
    if args.strip_think_tag:
        _print("Step16", "strip \\n</think>\\n in reason outputs + merged")
        if not args.dry_run:
            for p in [args.out_train_reason, args.out_syn_reason, args.out_test_reason, args.merge_output]:
                replace_think_tag_inplace(p)

    # Step 17: modify prompts
    _print("Step17", f"modify prompts merged {args.merge_output} -> {args.merged_prompt_output}")
    if not args.dry_run:
        modify_prompts(args.merge_output, args.merged_prompt_output)

    if args.modify_test_prompt_inplace:
        _print("Step18", f"modify prompts test in-place {args.out_test_nonempty} -> {args.out_test_nonempty}")
        if not args.dry_run:
            modify_prompts(args.out_test_nonempty, args.out_test_nonempty)
    else:
        if not args.test_prompt_output:
            raise ValueError("When --no-modify-test-prompt-inplace is set, you must provide --test-prompt-output")
        _print("Step18", f"modify prompts test {args.out_test_nonempty} -> {args.test_prompt_output}")
        if not args.dry_run:
            modify_prompts(args.out_test_nonempty, args.test_prompt_output)


if __name__ == "__main__":
    main()


