#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _chdir_to_repo_root() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)


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


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def copy_file(src: str, dst: str) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            buf = fsrc.read(1024 * 1024)
            if not buf:
                break
            fdst.write(buf)


def _canon_token(s: str) -> str:
    if s is None:
        return ""
    t = str(s).lower().strip()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def build_keyword(name: str, action: str, obj: str, direction: str) -> str:
    n = _canon_token(name)[:60]
    a = (action or "").strip().lower()
    o = (obj or "").strip().lower()
    d = (direction or "").strip().lower()
    return f"{n}|{a}|{o}|{d}"


def normalize_summary(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def normalize_keypair(date_v: Any, keyword_v: Any) -> Tuple[str, str]:
    d = "" if date_v is None else str(date_v).strip()
    k = "" if keyword_v is None else str(keyword_v).strip().lower()
    return (d, k)


def dedup_only(records: List[Dict[str, Any]], keep: str = "first") -> List[Dict[str, Any]]:
    assert keep in {"first", "last"}
    if keep == "first":
        seen_summary: Set[str] = set()
        seen_pair: Set[Tuple[str, str]] = set()
        out: List[Dict[str, Any]] = []
        for rec in records:
            s_key = normalize_summary(rec.get("summary"))
            p_key = normalize_keypair(rec.get("date"), rec.get("keyword"))
            if (s_key and s_key in seen_summary) or (p_key in seen_pair):
                continue
            if s_key:
                seen_summary.add(s_key)
            seen_pair.add(p_key)
            out.append(rec)
        return out
    by_summary: Dict[str, Dict[str, Any]] = {}
    by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in records:
        s_key = normalize_summary(rec.get("summary"))
        p_key = normalize_keypair(rec.get("date"), rec.get("keyword"))
        if s_key:
            by_summary[s_key] = rec
        by_pair[p_key] = rec
    kept_map: Dict[str, Dict[str, Any]] = {}
    for rec in list(by_summary.values()) + list(by_pair.values()):
        key = json.dumps(rec, sort_keys=True, ensure_ascii=False)
        kept_map[key] = rec
    return list(kept_map.values())


def count_keywords(records: List[Dict[str, Any]]) -> Counter:
    return Counter((r.get("keyword", "") or "").strip().lower() for r in records)


def filter_by_keyword_count(records: List[Dict[str, Any]], min_count: int) -> List[Dict[str, Any]]:
    if min_count is None or min_count <= 1:
        return records
    cnt = count_keywords(records)
    return [r for r in records if cnt[(r.get("keyword", "") or "").strip().lower()] >= min_count]


def keep_topk_keywords(records: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if not k or k <= 0:
        return records
    cnt = count_keywords(records)
    sorted_keys = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    keep_set = set([kw for kw, _ in sorted_keys[:k]])
    return [r for r in records if ((r.get("keyword", "") or "").strip().lower()) in keep_set]


def dedup_pipeline(
    input_path: str,
    output_path: str,
    keep: str = "first",
    recompute_keyword: bool = False,
    min_keyword_count: int = 1,
    topk_keywords: int = 0,
) -> Dict[str, int]:
    all_records = list(iter_jsonl(input_path))
    total = len(all_records)
    if recompute_keyword:
        for r in all_records:
            if not r.get("keyword"):
                r["keyword"] = build_keyword(
                    r.get("name", ""),
                    r.get("action", ""),
                    r.get("object", ""),
                    r.get("direction", ""),
                )
    deduped = dedup_only(all_records, keep=keep)
    filtered = filter_by_keyword_count(deduped, min_keyword_count)
    topk_filtered = keep_topk_keywords(filtered, topk_keywords)
    write_jsonl(output_path, topk_filtered)
    kept = len(topk_filtered)
    return {"total": total, "kept": kept, "dropped": max(0, total - kept)}


_FMT_TRIES = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d",
    "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%m-%d-%Y",
]


def parse_date_safe(s: Any) -> Optional[date]:
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None
    if len(txt) >= 10:
        head10 = txt[:10]
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
            try:
                return datetime.strptime(head10, fmt).date()
            except Exception:
                pass
    for fmt in _FMT_TRIES:
        try:
            return datetime.strptime(txt, fmt).date()
        except Exception:
            continue
    return None


def split_by_date(
    input_path: str,
    threshold_str: str,
    before_output: str,
    after_output: str,
    date_field: str = "date",
    eq_goes_to: str = "after",
    invalid_output: Optional[str] = None,
) -> Dict[str, int]:
    assert eq_goes_to in {"after", "before"}
    thr = parse_date_safe(threshold_str)
    if thr is None:
        raise ValueError(f"Unable to parse threshold date: {threshold_str!r}")
    Path(before_output).parent.mkdir(parents=True, exist_ok=True)
    Path(after_output).parent.mkdir(parents=True, exist_ok=True)
    if invalid_output:
        Path(invalid_output).parent.mkdir(parents=True, exist_ok=True)
    f_before = open(before_output, "w", encoding="utf-8")
    f_after = open(after_output, "w", encoding="utf-8")
    f_invalid = open(invalid_output, "w", encoding="utf-8") if invalid_output else None
    stats = {"total": 0, "before": 0, "after": 0, "invalid": 0}
    try:
        for rec in iter_jsonl(input_path):
            stats["total"] += 1
            d = parse_date_safe(rec.get(date_field))
            if d is None:
                stats["invalid"] += 1
                if f_invalid is not None:
                    f_invalid.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            if d < thr:
                f_before.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["before"] += 1
            elif d > thr:
                f_after.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["after"] += 1
            else:
                if eq_goes_to == "after":
                    f_after.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["after"] += 1
                else:
                    f_before.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["before"] += 1
    finally:
        f_before.close()
        f_after.close()
        if f_invalid is not None:
            f_invalid.close()
    return stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Standalone energy pipeline (dedup + split-by-date).")
    ap.add_argument("--input-events", default="eventsdata_en/events_flat_simple.jsonl")
    ap.add_argument("--dedup-min5-min-keyword-count", type=int, default=5)
    ap.add_argument("--dedup-min5-topk-keywords", type=int, default=1000)
    ap.add_argument("--dedup-min5-output", default="data4dedup_en/events_dedup_min5topk1000.jsonl")
    ap.add_argument("--dedup-min1-min-keyword-count", type=int, default=1)
    ap.add_argument("--dedup-min1-topk-keywords", type=int, default=1000)
    ap.add_argument("--dedup-min1-output", default="data4dedup_en/events_dedup_min1topk1000_new.jsonl")
    ap.add_argument("--split1-threshold", default="2022-01-01")
    ap.add_argument("--split2-threshold", default="2023-01-01")
    ap.add_argument("--split3-threshold", default="2023-01-01")
    ap.add_argument("--split-min1-input", default="data4dedup_en/events_dedup_min1topk1000.jsonl")
    ap.add_argument("--sync-min1-output-to-split-input", action="store_true", default=True)
    ap.add_argument("--no-sync-min1-output-to-split-input", action="store_false", dest="sync_min1_output_to_split_input")
    ap.add_argument("--split1-before-output", default="data4dedup_en/events_dedup_min5topk1000_before2022_train.jsonl")
    ap.add_argument("--split1-after-output", default="data4dedup_en/events_dedup_min5topk1000_after2022_testvali.jsonl")
    ap.add_argument("--split2-before-output", default="data4dedup_en/events_dedup_min5topk1000_during2022_2023_vail.jsonl")
    ap.add_argument("--split2-after-output", default="data4dedup_en/events_dedup_min5topk1000_after2023_test.jsonl")
    ap.add_argument("--split3-before-output", default="data4dedup_en/events_dedup_min1topk1000_before2023_test_stat.jsonl")
    ap.add_argument("--split3-after-output", default="data4dedup_en/events_dedup_min1topk1000_after2023_test_stat.jsonl")
    ap.add_argument("--keep", choices=["first", "last"], default="first")
    ap.add_argument("--recompute-keyword", action="store_true")
    ap.add_argument("--date-field", default="date")
    ap.add_argument("--eq-goes-to", choices=["after", "before"], default="after")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    _chdir_to_repo_root()
    args = parse_args()
    if args.dry_run:
        print("[DryRun] Planned steps:")
        print(f"1) dedup(min=5, topk=1000) {args.input_events} -> {args.dedup_min5_output}")
        print(f"2) dedup(min=1, topk=1000) {args.input_events} -> {args.dedup_min1_output}")
        if args.sync_min1_output_to_split_input:
            print(f"2b) copy {args.dedup_min1_output} -> {args.split_min1_input}")
        print(f"3) split thr={args.split1_threshold} {args.dedup_min5_output} -> ({args.split1_before_output}, {args.split1_after_output})")
        print(f"4) split thr={args.split2_threshold} {args.split1_after_output} -> ({args.split2_before_output}, {args.split2_after_output})")
        print(f"5) split thr={args.split3_threshold} {args.split_min1_input} -> ({args.split3_before_output}, {args.split3_after_output})")
        return
    s1 = dedup_pipeline(
        input_path=args.input_events,
        output_path=args.dedup_min5_output,
        keep=args.keep,
        recompute_keyword=args.recompute_keyword,
        min_keyword_count=args.dedup_min5_min_keyword_count,
        topk_keywords=args.dedup_min5_topk_keywords,
    )
    print(f"[Step1 Dedup min5] total={s1['total']} kept={s1['kept']} dropped={s1['dropped']} -> {args.dedup_min5_output}")
    s2 = dedup_pipeline(
        input_path=args.input_events,
        output_path=args.dedup_min1_output,
        keep=args.keep,
        recompute_keyword=args.recompute_keyword,
        min_keyword_count=args.dedup_min1_min_keyword_count,
        topk_keywords=args.dedup_min1_topk_keywords,
    )
    print(f"[Step2 Dedup min1] total={s2['total']} kept={s2['kept']} dropped={s2['dropped']} -> {args.dedup_min1_output}")
    if args.sync_min1_output_to_split_input:
        copy_file(args.dedup_min1_output, args.split_min1_input)
        print(f"[Step2b Sync] {args.dedup_min1_output} -> {args.split_min1_input}")
    st3 = split_by_date(
        input_path=args.dedup_min5_output,
        threshold_str=args.split1_threshold,
        before_output=args.split1_before_output,
        after_output=args.split1_after_output,
        date_field=args.date_field,
        eq_goes_to=args.eq_goes_to,
    )
    print(f"[Step3 Split] total={st3['total']} before={st3['before']} after={st3['after']} invalid={st3['invalid']}")
    st4 = split_by_date(
        input_path=args.split1_after_output,
        threshold_str=args.split2_threshold,
        before_output=args.split2_before_output,
        after_output=args.split2_after_output,
        date_field=args.date_field,
        eq_goes_to=args.eq_goes_to,
    )
    print(f"[Step4 Split] total={st4['total']} before={st4['before']} after={st4['after']} invalid={st4['invalid']}")
    if not os.path.exists(args.split_min1_input):
        raise FileNotFoundError(f"Split-3 input not found: {args.split_min1_input}")
    st5 = split_by_date(
        input_path=args.split_min1_input,
        threshold_str=args.split3_threshold,
        before_output=args.split3_before_output,
        after_output=args.split3_after_output,
        date_field=args.date_field,
        eq_goes_to=args.eq_goes_to,
    )
    print(f"[Step5 Split] total={st5['total']} before={st5['before']} after={st5['after']} invalid={st5['invalid']}")


if __name__ == "__main__":
    main()