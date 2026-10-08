"""Generation. One answer per (question, arm, replicate), appended to JSONL.

Separate from scoring on purpose: re-scoring must never re-generate. A crash at
question 800 of 1,020 must not cost the first 800, so every row is keyed and a re-run
skips what is already there.

Guards, in the order they would otherwise bite:
  * prompt parity -- every LLM arm must report the same template hash, or the run
    refuses to start. This is the tier-ablation bug made impossible rather than
    remembered.
  * cost ceiling -- a hard abort, defaulting low. The first real run should cost cents.
  * dry run -- prints the exact prompts without calling anything, which is how arm
    parity gets checked by eye before any money is spent.
  * resume -- on by default; --fresh is the explicit opt-out.

Run:
  python -m benchmark.run --part pilot --species 5 --dry-run
  python -m benchmark.run --part pilot --species 5 --arms G
  python -m benchmark.run --part pilot --species 5 --arms G,A1,C,D,E,F --max-cost 1.00
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import io
import json
import sys
from pathlib import Path

from mpcr_rag.store import pg_store

from . import arms as A
from . import client, prompts
from .evidence_pack import build as build_pack

DATA = Path(__file__).resolve().parent / "data"


def _load_questions(part: str) -> list[dict]:
    p = DATA / f"questions_{part}.jsonl"
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def pick_species(rows: list[dict], n: int) -> list[str]:
    """N species spread across strata rather than the first N alphabetically, so a
    small run still touches both ends of the long tail. Deterministic."""
    by_stratum: dict[str, list[str]] = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: (r.get("stratum", ""), r["species"])):
        by_stratum.setdefault(r.get("stratum", ""), [])
        if r["species"] not in by_stratum[r["stratum"]]:
            by_stratum[r["stratum"]].append(r["species"])
    out: list[str] = []
    while len(out) < n and any(by_stratum.values()):
        for s in list(by_stratum):
            if by_stratum[s] and len(out) < n:
                out.append(by_stratum[s].pop(0))
    return out


def _assert_prompt_parity(selected: list[str]) -> str:
    """Every model arm shares one template. Checked, not trusted."""
    h = prompts.template_hash()
    llm = [n for n in selected if A.ARMS[n].is_llm]
    if not llm:
        return h
    hashes = {n: prompts.template_hash() for n in llm}
    if len(set(hashes.values())) != 1:
        raise SystemExit(f"prompt parity violated: {hashes}")
    return h


def _existing(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    seen = set()
    for line in io.open(path, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("error"):
            seen.add((r["qid"], r["arm"], r.get("replicate", 0)))
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate benchmark answers")
    ap.add_argument("--part", default="pilot", choices=["pilot", "main"])
    ap.add_argument("--arms", default=",".join(A.READY),
                    help=f"comma separated; ready: {','.join(A.READY)}")
    ap.add_argument("--species", type=int, default=0,
                    help="restrict to N species spread across strata (0 = all)")
    ap.add_argument("--qtypes", default="", help="comma separated filter")
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--max-cost", type=float, default=1.00,
                    help="hard abort once accumulated cost exceeds this (USD)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print prompts, call nothing, write nothing")
    ap.add_argument("--fresh", action="store_true", help="ignore existing rows")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    selected = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in selected if a not in A.ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}; known: {sorted(A.ARMS)}")
    not_ready = [a for a in selected if a not in A.READY]
    if not_ready:
        raise SystemExit(f"arms not ready: {not_ready} -- {[A.ARMS[a].note for a in not_ready]}")

    rows = _load_questions(args.part)
    if args.species:
        keep = set(pick_species(rows, args.species))
        rows = [r for r in rows if r["species"] in keep]
    if args.qtypes:
        want = {q.strip() for q in args.qtypes.split(",")}
        rows = [r for r in rows if r["qtype"] in want]
    rows.sort(key=lambda r: (r["species"], r["qtype"]))

    thash = _assert_prompt_parity(selected)
    out = Path(args.out) if args.out else DATA / f"answers_{args.part}.jsonl"
    done = set() if args.fresh else _existing(out)

    todo = [(r, a) for r in rows for a in selected
            if (r["qid"], a, args.replicate) not in done]

    print(f"part={args.part}  questions={len(rows)}  "
          f"species={len({r['species'] for r in rows})}  arms={selected}")
    print(f"template_hash={thash}  already done={len(done)}  to do={len(todo)}")
    if args.dry_run:
        print("DRY RUN -- nothing will be called or written\n")

    conn = pg_store.connect()
    spent = 0.0
    wrote = failed = 0
    fh = None if args.dry_run else io.open(out, "a", encoding="utf-8")
    try:
        with conn.cursor() as cur:
            for i, (q, arm_name) in enumerate(todo, 1):
                arm = A.ARMS[arm_name]
                pack = None
                if arm.evidence:
                    pack = build_pack(cur, q["species"], q["qtype"], level=arm.evidence,
                                      place_code=q["key"].get("place_code"))

                if args.dry_run:
                    if i <= len(selected) * 2:     # a couple of questions is enough to eyeball
                        ev = pack.text() if (pack and arm.is_llm) else None
                        print(f"=== {q['qid']} / arm {arm_name} "
                              f"({arm.model or 'template'}, evidence={arm.evidence}) ===")
                        if arm.is_llm:
                            for m in prompts.build(q["question"], ev):
                                print(f"[{m['role']}]\n{m['content']}\n")
                        else:
                            print(A.template_answer(q["question"], q["qtype"], pack.facts), "\n")
                    continue

                base = {"qid": q["qid"], "arm": arm_name, "replicate": args.replicate,
                        "species": q["species"], "qtype": q["qtype"],
                        "prompt_template_hash": thash,
                        "evidence_pack_hash": pack.hash if pack else None,
                        "evidence_providers": pack.providers if pack else [],
                        "evidence_level": arm.evidence,
                        "started_at": dt.datetime.now().isoformat(timespec="seconds")}
                try:
                    if arm.is_llm:
                        res = client.call(arm.model,
                                          prompts.build(q["question"], pack.text() if pack else None),
                                          pin=arm.pin)
                        spent += float(res.get("cost_usd") or 0.0)
                    else:
                        res = {"answer_text": A.template_answer(q["question"], q["qtype"],
                                                               pack.facts),
                               "model_id": None, "model_version": None,
                               "provider_pinned": None, "served_by": None, "pin_held": None,
                               "prompt_tokens": None, "completion_tokens": None,
                               "cost_usd": 0.0, "latency_ms": 0, "attempts": 1}
                    row = {**base, **res, "error": None}
                    wrote += 1
                except Exception as e:
                    row = {**base, "answer_text": None, "error": f"{type(e).__name__}: {e}"}
                    failed += 1

                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 20 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  ok={wrote} fail={failed} spent=${spent:.4f}")
                if spent > args.max_cost:
                    print(f"\nCOST CEILING hit (${spent:.4f} > ${args.max_cost:.2f}); stopping. "
                          f"Re-run to resume.", file=sys.stderr)
                    break
    finally:
        if fh:
            fh.close()
        conn.close()

    if not args.dry_run:
        print(f"\nwrote {wrote} rows, {failed} errors, ${spent:.4f} -> {out}")


if __name__ == "__main__":
    main()
