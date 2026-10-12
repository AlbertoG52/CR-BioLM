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
import base64
import collections
import concurrent.futures as cf
import threading
import datetime as dt
import io
import json
import sys
from pathlib import Path

from mpcr_rag.store import pg_store

from . import arms as A
from . import client, prompts, router
from .evidence_pack import build as build_pack
from .qmap import render as render_qmap

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


def _assert_prompt_parity(selected: list[str]) -> dict[str, str]:
    """Arms in the same prompt FAMILY share one template. Checked, not trusted.

    Two families are declared (prompts.FAMILIES): the grounded and careful closed-book
    arms, and the free closed-book baselines. Within a family the hash must be identical
    or the ablation measures the prompt; across families it must differ, or the free
    baseline is not actually free. Both directions are asserted.
    """
    llm = [n for n in selected if A.ARMS[n].is_llm]
    per_arm = {n: prompts.template_hash(A.ARMS[n].prompt) for n in llm}
    by_family: dict[str, set] = {}
    for n in llm:
        by_family.setdefault(A.ARMS[n].prompt, set()).add(per_arm[n])
    for fam, hashes in by_family.items():
        if len(hashes) != 1:
            raise SystemExit(f"prompt parity violated inside family {fam!r}: {hashes}")
    if len({next(iter(h)) for h in by_family.values()}) != len(by_family):
        raise SystemExit(f"prompt families collide: {by_family}")
    return per_arm


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
    ap.add_argument("--species-file", default="",
                    help="a frozen JSON list of species names; preferred over --species "
                         "for a run that goes in a paper, because the selection is then "
                         "a committed artefact rather than a function's behaviour")
    ap.add_argument("--qtypes", default="", help="comma separated filter")
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--max-cost", type=float, default=1.00,
                    help="hard abort once accumulated cost exceeds this (USD)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print prompts, call nothing, write nothing")
    ap.add_argument("--fresh", action="store_true", help="ignore existing rows")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel API calls. Generation is network-bound and idle "
                         "between calls, so this is the one stage worth fanning out")
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
    if args.species_file:
        keep = set(json.loads(Path(args.species_file).read_text(encoding="utf-8")))
        missing = keep - {r["species"] for r in rows}
        if missing:
            raise SystemExit(f"{len(missing)} species not in part={args.part}: "
                             f"{sorted(missing)[:3]}")
        rows = [r for r in rows if r["species"] in keep]
    elif args.species:
        keep = set(pick_species(rows, args.species))
        rows = [r for r in rows if r["species"] in keep]
    if args.qtypes:
        want = {q.strip() for q in args.qtypes.split(",")}
        rows = [r for r in rows if r["qtype"] in want]
    rows.sort(key=lambda r: (r["species"], r["qtype"]))

    thashes = _assert_prompt_parity(selected)
    out = Path(args.out) if args.out else DATA / f"answers_{args.part}.jsonl"
    done = set() if args.fresh else _existing(out)

    todo = [(r, a) for r in rows for a in selected
            if (r["qid"], a, args.replicate) not in done]

    print(f"part={args.part}  questions={len(rows)}  "
          f"species={len({r['species'] for r in rows})}  arms={selected}")
    fams = {A.ARMS[n].prompt for n in selected if A.ARMS[n].is_llm}
    print(f"prompt families={sorted(fams)}  hashes={sorted(set(thashes.values()))}")
    print(f"already done={len(done)}  to do={len(todo)}")
    if args.dry_run:
        print("DRY RUN -- nothing will be called or written\n")

    # Two phases, because they have different constraints. Preparation touches Postgres
    # and matplotlib, neither of which is thread-safe, so it stays serial; it is also
    # fast. The API calls are the slow part and are pure network, so they fan out.
    #
    # Scoring is NOT parallelised: it runs on one local GPU, and workers there would
    # contend for the same constrained resource rather than overlap.
    conn = pg_store.connect()
    jobs: list[dict] = []
    try:
        with conn.cursor() as cur:
            for i, (q, arm_name) in enumerate(todo, 1):
                arm = A.ARMS[arm_name]
                if arm.evidence == "map" and q["qtype"] not in A.MAP_QTYPES:
                    continue        # a picture carries nothing this key tests
                pack = None
                img_b64 = None
                map_shown = None
                routing = None
                if arm.routed and not args.dry_run:
                    # The router sees the question and nothing else. What it returns
                    # replaces the species, type and place the harness would otherwise
                    # have handed over; the gold values are recorded beside it so the
                    # routing error is auditable per row.
                    try:
                        r = router.route(cur, q["question"])
                    except Exception as e:
                        # A router call that failed is not a question the router
                        # declined. It becomes an error row, which resume retries.
                        jobs.append({"arm": arm, "messages": None, "template": None,
                                     "prep_error": f"router: {type(e).__name__}: {e}",
                                     "base": {"qid": q["qid"], "arm": arm_name,
                                              "replicate": args.replicate,
                                              "species": q["species"],
                                              "qtype": q["qtype"]}})
                        continue
                    rec = router.recipe(r)
                    routing = {"route": r["route"], "species": r["species"],
                               "place_code": r["place_code"], "served": rec is not None,
                               "route_ok": r["route"] == router.QTYPE_ROUTE.get(
                                   q["qtype"], q["qtype"]),
                               "species_ok": r["species"] == q["species"],
                               "place_ok": (q["key"].get("place_code") is None
                                            or r["place_code"] == q["key"].get("place_code")),
                               "cost_usd": r["proposal"].get("_cost_usd")}
                    if rec:
                        pack = build_pack(cur, rec[0], rec[1], level="full",
                                          place_code=rec[2])
                elif arm.routed:
                    pass            # dry run: routing costs a call, so it is skipped
                elif arm.evidence == "map":
                    pk = build_pack(cur, q["species"], q["qtype"], level="full",
                                    place_code=q["key"].get("place_code"))
                    # Region overlap is half of what a protected-area key turns on,
                    # so the map must carry it -- but NOT for vertiente, where which
                    # region the species occupies is the answer being tested.
                    regs = (pk.facts.get("regions")
                            if q["qtype"].startswith("park_") else None)
                    mp, map_shown = render_qmap(
                        q["species"], q["key"].get("place_code"),
                        elev_min=pk.facts.get("elev_min"),
                        elev_max=pk.facts.get("elev_max"),
                        highlight_regions=regs)
                    img_b64 = base64.b64encode(Path(mp).read_bytes()).decode()
                elif arm.evidence:
                    pack = build_pack(cur, q["species"], q["qtype"], level=arm.evidence,
                                      place_code=q["key"].get("place_code"))

                if args.dry_run:
                    if i <= len(selected) * 2:     # a couple of questions is enough to eyeball
                        ev = pack.text() if (pack and arm.is_llm) else None
                        print(f"=== {q['qid']} / arm {arm_name} "
                              f"({arm.model or 'template'}, evidence={arm.evidence}) ===")
                        if img_b64:
                            print(f"[image] {len(img_b64)//1024} KB base64 | {map_shown}")
                        if arm.is_llm:
                            for m in prompts.build(q["question"], ev, family=arm.prompt):
                                print(f"[{m['role']}]\n{m['content']}\n")
                        else:
                            print(A.template_answer(q["question"], q["qtype"], pack.facts), "\n")
                    continue

                unrouted = arm.routed and pack is None
                jobs.append({
                    "arm": arm,
                    "messages": (prompts.build(q["question"],
                                               pack.text() if pack else None,
                                               family=arm.prompt, image_b64=img_b64)
                                 if arm.is_llm and not unrouted else None),
                    "template": (A.UNROUTED if unrouted else None if arm.is_llm else
                                 A.template_answer(q["question"], q["qtype"], pack.facts)),
                    "routing": routing,
                    "base": {"qid": q["qid"], "arm": arm_name, "replicate": args.replicate,
                             "species": q["species"], "qtype": q["qtype"],
                             "prompt_template_hash": thashes.get(arm_name),
                             "prompt_family": arm.prompt,
                             "evidence_pack_hash": pack.hash if pack else None,
                             "evidence_providers": pack.providers if pack else [],
                             "map_shown": map_shown,
                             "routing": routing,
                             "evidence_level": arm.evidence},
                })
    finally:
        conn.close()

    if args.dry_run:
        return

    TEMPLATE_RES = {"model_id": None, "model_version": None, "provider_pinned": None,
                    "served_by": None, "pin_held": None, "prompt_tokens": None,
                    "completion_tokens": None, "reasoning_tokens": None,
                    "finish_reason": None, "cost_usd": 0.0, "latency_ms": 0, "attempts": 1}

    lock = threading.Lock()
    state = {"spent": 0.0, "wrote": 0, "failed": 0, "done": 0, "halt": False}

    def execute(job: dict) -> dict:
        base = {**job["base"],
                "started_at": dt.datetime.now().isoformat(timespec="seconds")}
        with lock:
            if state["halt"]:
                return {}
        if job.get("prep_error"):
            return {**base, "answer_text": None, "error": job["prep_error"]}
        try:
            if job["arm"].is_llm and job["messages"] is not None:
                res = client.call(job["arm"].model, job["messages"], pin=job["arm"].pin)
            else:
                res = {"answer_text": job["template"], **TEMPLATE_RES}
            return {**base, **res, "error": None}
        except Exception as e:
            return {**base, "answer_text": None, "error": f"{type(e).__name__}: {e}"}

    print(f"executing {len(jobs)} calls with {args.workers} workers")
    fh = io.open(out, "a", encoding="utf-8")
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(execute, jobs):
                if not row:
                    continue
                with lock:
                    state["spent"] += float(row.get("cost_usd") or 0.0)
                    if row.get("error"):
                        state["failed"] += 1
                    else:
                        state["wrote"] += 1
                    state["done"] += 1
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    if state["done"] % 25 == 0 or state["done"] == len(jobs):
                        print(f"  {state['done']}/{len(jobs)}  ok={state['wrote']} "
                              f"fail={state['failed']} spent=${state['spent']:.4f}")
                    if state["spent"] > args.max_cost and not state["halt"]:
                        state["halt"] = True
                        print(f"\nCOST CEILING hit (${state['spent']:.4f} > "
                              f"${args.max_cost:.2f}); stopping. Re-run to resume.",
                              file=sys.stderr)
    finally:
        fh.close()

    print(f"\nwrote {state['wrote']} rows, {state['failed']} errors, "
          f"${state['spent']:.4f} -> {out}")


if __name__ == "__main__":
    main()
