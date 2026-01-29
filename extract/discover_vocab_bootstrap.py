#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def aggregate_suggestions(input_jsonl: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    agg: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for obj in iter_jsonl(input_jsonl):
        term = obj.get("term")
        slot = obj.get("slot")
        if not isinstance(term, str) or not isinstance(slot, str):
            continue
        try:
            score = float(obj.get("score", 0.0))
        except Exception:
            score = 0.0
        key = (slot, term)
        if key not in agg:
            agg[key] = {"slot": slot, "term": term, "count": 0, "scores": []}
        agg[key]["count"] += 1
        agg[key]["scores"].append(score)
    for v in agg.values():
        scores = v.get("scores") or []
        v["avg_score"] = float(sum(scores) / max(1, len(scores)))
        v.pop("scores", None)
    return agg


def main() -> None:
    _chdir_to_repo_root()
    ap = argparse.ArgumentParser(description="Aggregate vocab suggestions and output high-support terms.")
    ap.add_argument("--input", default="eventsdata_en/vocab_suggestions.jsonl", help="Input JSONL from extraction stage.")
    ap.add_argument("--out", default="eventsdata_en/vocab_suggestions_filtered.jsonl", help="Output JSONL (term/slot/count/avg_score).")
    ap.add_argument("--thr", type=int, default=10, help="Minimum count threshold (default: 10).")
    ap.add_argument("--min-score", type=float, default=0.0, help="Optional minimum average score (default: 0.0).")
    ap.add_argument("--sort", choices=["count", "avg_score"], default="count")
    args = ap.parse_args()

    agg = aggregate_suggestions(args.input)
    rows = [v for v in agg.values() if int(v["count"]) >= int(args.thr) and float(v["avg_score"]) >= float(args.min_score)]
    if args.sort == "avg_score":
        rows.sort(key=lambda r: (-float(r["avg_score"]), -int(r["count"]), r["slot"], r["term"]))
    else:
        rows.sort(key=lambda r: (-int(r["count"]), -float(r["avg_score"]), r["slot"], r["term"]))

    write_jsonl(args.out, rows)

    by_slot = defaultdict(int)
    for r in rows:
        by_slot[r["slot"]] += 1

    print("=== VOCAB BOOTSTRAP SUMMARY ===")
    print(json.dumps({"input": args.input, "out": args.out, "thr": args.thr, "min_score": args.min_score, "kept_total": len(rows), "kept_by_slot": dict(by_slot)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()