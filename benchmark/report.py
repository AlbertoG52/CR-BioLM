"""Build a browsable report of a run: the numbers, and the answers behind them.

The aggregate tables hide what matters. Six of the ten defects this experiment has
turned up were found by reading answers, not by reading a summary -- a checker that
scored the farm's elevation against the species, a denial of use counted as an
assertion, a map missing half its evidence. None of those move a precision figure
enough to notice; all of them are obvious the moment you read what an arm actually
said next to what the key expected.

So this emits one page per run carrying both: the per-arm metrics, and every answer
beside its key, its verdict, its claim-level breakdown and the evidence pack it was
given. It is the read-the-answers step made cheap enough to actually do.

Run:  python -m benchmark.report --part pilot
Writes benchmark/data/report_<part>.html
"""
from __future__ import annotations

import argparse
import html
import io
import json
from pathlib import Path

from mpcr_rag.store import pg_store

from . import analyze
from .arms import ARMS
from .evidence_pack import build as build_pack

DATA = Path(__file__).resolve().parent / "data"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in io.open(path, encoding="utf-8"):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def collect(part: str, with_packs: bool = True) -> dict:
    questions = {q["qid"]: q for q in _jsonl(DATA / f"questions_{part}.jsonl")}
    answers = _jsonl(DATA / f"answers_{part}.jsonl")
    verdicts = {(v["qid"], v["arm"]): v for v in _jsonl(DATA / f"verdicts_{part}.jsonl")}
    summ = analyze.summarize(list(verdicts.values()))

    # One pack per (species, qtype) at the full level: what the grounded arms saw.
    packs: dict[str, str] = {}
    if with_packs:
        conn = pg_store.connect()
        try:
            with conn.cursor() as cur:
                for qid, q in questions.items():
                    if not any(a["qid"] == qid for a in answers):
                        continue
                    try:
                        p = build_pack(cur, q["species"], q["qtype"], level="full",
                                       place_code=q["key"].get("place_code"))
                        packs[qid] = p.text()
                    except Exception as e:
                        packs[qid] = f"(pack unavailable: {type(e).__name__})"
        finally:
            conn.close()

    rows = []
    for qid, q in sorted(questions.items()):
        got = [a for a in answers if a["qid"] == qid]
        if not got:
            continue
        rows.append({
            "qid": qid, "species": q["species"], "qtype": q["qtype"],
            "question": q["question"], "key": q["key"],
            "key_level": q.get("key_level"), "stratum": q.get("stratum"),
            "n_records": q.get("n_records"), "pack": packs.get(qid, ""),
            "answers": [{
                "arm": a["arm"], "text": a.get("answer_text") or "",
                "error": a.get("error"),
                "evidence": a.get("evidence_level"), "family": a.get("prompt_family"),
                "model": (a.get("model_id") or "template").split("/")[-1],
                "served_by": a.get("served_by"), "cost": a.get("cost_usd"),
                "verdict": (verdicts.get((qid, a["arm"])) or {}).get("direct"),
                "claims": (verdicts.get((qid, a["arm"])) or {}).get("claims") or [],
                "claim_counts": (verdicts.get((qid, a["arm"])) or {}).get("claim_counts") or {},
            } for a in sorted(got, key=lambda x: x["arm"])],
        })
    spend = sum(float(a.get("cost_usd") or 0) for a in answers)
    return {"part": part, "rows": rows, "summary": summ,
            "arms": {n: {"note": a.note, "evidence": a.evidence, "prompt": a.prompt,
                         "model": (a.model or "template").split("/")[-1]}
                     for n, a in ARMS.items()},
            "totals": {"answers": len(answers), "verdicts": len(verdicts),
                       "species": len({r["species"] for r in rows}),
                       "questions": len(rows), "spend": round(spend, 4)}}


PAGE = """<title>CR-FloraLM Evidence Browser</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Public+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a fixed summary band, then a two-pane browser -- question list beside the
   answers for that question. Collapses to one column on a phone. */
:root {
  --bg: #f7f6f2; --surface: #ffffff; --line: #dddbd2; --ink: #1e2420;
  --ink-soft: #5d655e; --accent: #2f6b4f; --accent-soft: #e6efe9;
  --ok: #1f7a4d; --bad: #b3412e; --warn: #9a6b12; --idle: #6b7280;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
  --body: "Public Sans", system-ui, sans-serif;
  --display: "Fraunces", Georgia, serif;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #15181a; --surface: #1d2124; --line: #333a3d; --ink: #e8ece9;
  --ink-soft: #9aa5a0; --accent: #7fc0a0; --accent-soft: #223029;
  --ok: #68c295; --bad: #e8836e; --warn: #d9ae55; --idle: #8b949e;
  color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #15181a; --surface: #1d2124; --line: #333a3d; --ink: #e8ece9;
  --ink-soft: #9aa5a0; --accent: #7fc0a0; --accent-soft: #223029;
  --ok: #68c295; --bad: #e8836e; --warn: #d9ae55; --idle: #8b949e;
  color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font-family: var(--body);
       font-size: 15px; line-height: 1.55; }
.wrap { padding-inline: 16px; padding-block: 22px 40px; max-width: 1400px; margin: 0 auto; }
h1 { font-family: var(--display); font-weight: 700; font-size: clamp(22px, 3vw, 30px);
     margin: 0 0 2px; text-wrap: balance; letter-spacing: -0.01em; }
.sub { color: var(--ink-soft); font-size: 13px; margin-bottom: 20px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums;
        font-size: 13px; }
th, td { text-align: right; padding: 7px 9px; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
     color: var(--ink-soft); font-weight: 600; }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px;
          background: var(--surface); }
.band { margin-bottom: 26px; }
.arm { font-family: var(--mono); font-weight: 500; }
.browser { display: grid; grid-template-columns: 300px 1fr; gap: 18px; align-items: start; }
@media (max-width: 860px) { .browser { grid-template-columns: 1fr; } }
.qlist { border: 1px solid var(--line); border-radius: 8px; background: var(--surface);
         max-height: 72vh; overflow-y: auto; }
.qitem { padding: 9px 12px; border-bottom: 1px solid var(--line); cursor: pointer;
         font-size: 13px; }
.qitem:hover { background: var(--accent-soft); }
.qitem[aria-selected="true"] { background: var(--accent-soft);
         box-shadow: inset 3px 0 0 var(--accent); }
.qitem .sp { font-style: italic; }
.qitem .qt { font-family: var(--mono); font-size: 11px; color: var(--ink-soft); }
.panel { border: 1px solid var(--line); border-radius: 8px; background: var(--surface);
         padding: 16px; min-width: 0; }
.qtext { font-family: var(--display); font-size: 18px; line-height: 1.35;
         margin: 0 0 10px; text-wrap: balance; }
.keybox, .packbox { font-family: var(--mono); font-size: 12px; white-space: pre-wrap;
         background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
         padding: 10px; overflow-x: auto; margin: 0 0 14px; }
details summary { cursor: pointer; font-size: 12px; color: var(--ink-soft);
         text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
.ans { border-top: 1px solid var(--line); padding: 12px 0; min-width: 0; }
.ans h3 { margin: 0 0 4px; font-size: 13px; display: flex; gap: 8px; flex-wrap: wrap;
          align-items: baseline; }
.ans p { margin: 4px 0 0; }
.pill { font-family: var(--mono); font-size: 11px; padding: 1px 7px; border-radius: 99px;
        border: 1px solid currentColor; white-space: nowrap; }
.v-correct { color: var(--ok); } .v-wrong { color: var(--bad); }
.v-overstated, .v-understated, .v-partially_correct { color: var(--warn); }
.v-abstained, .v-not_scored, .v-beyond_record { color: var(--idle); }
.meta { color: var(--ink-soft); font-size: 11px; font-family: var(--mono); }
.claims { margin: 6px 0 0; padding: 0; list-style: none; font-size: 12px; }
.claims li { display: flex; gap: 7px; padding: 2px 0; }
.c-supported { color: var(--ok); } .c-contradicted { color: var(--bad); }
.c-not_in_record { color: var(--warn); } .c-unverifiable { color: var(--idle); }
.filters { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
select, input { font: inherit; font-size: 13px; padding: 5px 8px; border-radius: 6px;
        border: 1px solid var(--line); background: var(--surface); color: var(--ink); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
<div class="wrap">
  <h1>CR-FloraLM Evidence Browser</h1>
  <div class="sub" id="sub"></div>

  <div class="band">
    <div class="scroll"><table id="summary"></table></div>
  </div>

  <div class="filters">
    <select id="f-qtype"><option value="">todos los tipos</option></select>
    <select id="f-verdict"><option value="">cualquier veredicto</option></select>
    <input id="f-text" type="search" placeholder="buscar especie…" size="18">
  </div>

  <div class="browser">
    <div class="qlist" id="qlist" role="listbox" aria-label="preguntas"></div>
    <div class="panel" id="panel"></div>
  </div>
</div>
<script id="payload" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("payload").textContent);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct = v => v == null ? "—" : (100*v).toFixed(1);

document.getElementById("sub").textContent =
  `${D.totals.questions} preguntas · ${D.totals.species} especies · ` +
  `${D.totals.answers} respuestas · ${D.totals.verdicts} veredictos · ` +
  `US$${D.totals.spend.toFixed(4)} · parte "${D.part}"`;

// ---- summary table: precision never without coverage
const S = D.summary, armNames = Object.keys(S);
const cols = [["arm","arm"],["n","n"],["scoreable","evaluables"],["answered","respondidas"],
  ["precision","precisión %"],["coverage","cobertura %"],["silence_correct","silencio ok %"],
  ["silence_asserted","inventado %"],["contradiction_rate","contradicho %"]];
let t = "<thead><tr>" + cols.map(c => `<th>${c[1]}</th>`).join("") + "<th>evidencia</th></tr></thead><tbody>";
for (const a of armNames) {
  const s = S[a], meta = D.arms[a] || {};
  const ci = s.precision_ci ? ` <span class="meta">[${(100*s.precision_ci[0]).toFixed(0)}–${(100*s.precision_ci[1]).toFixed(0)}]</span>` : "";
  t += `<tr><td class="arm">${esc(a)}</td><td>${s.n}</td><td>${s.scoreable}</td><td>${s.answered}</td>`
     + `<td>${pct(s.precision)}${ci}</td><td>${pct(s.coverage)}</td>`
     + `<td>${pct(s.silence_correct)}</td><td>${pct(s.silence_asserted)}</td>`
     + `<td>${pct(s.contradiction_rate)}</td>`
     + `<td class="meta">${esc(meta.evidence ?? "—")} · ${esc(meta.model ?? "")}</td></tr>`;
}
document.getElementById("summary").innerHTML = t + "</tbody>";

// ---- filters
const qtypes = [...new Set(D.rows.map(r => r.qtype))].sort();
const verdicts = [...new Set(D.rows.flatMap(r => r.answers.map(a => a.verdict)).filter(Boolean))].sort();
for (const q of qtypes) document.getElementById("f-qtype").add(new Option(q, q));
for (const v of verdicts) document.getElementById("f-verdict").add(new Option(v, v));

let sel = null;
function visible() {
  const qt = document.getElementById("f-qtype").value;
  const vd = document.getElementById("f-verdict").value;
  const tx = document.getElementById("f-text").value.trim().toLowerCase();
  return D.rows.filter(r =>
    (!qt || r.qtype === qt) &&
    (!vd || r.answers.some(a => a.verdict === vd)) &&
    (!tx || r.species.toLowerCase().includes(tx)));
}
function renderList() {
  const rows = visible();
  document.getElementById("qlist").innerHTML = rows.map(r =>
    `<div class="qitem" role="option" data-qid="${esc(r.qid)}" aria-selected="${sel===r.qid}">
       <div class="sp">${esc(r.species)}</div>
       <div class="qt">${esc(r.qtype)}${r.n_records!=null?` · ${r.n_records} reg.`:""}</div>
     </div>`).join("") || `<div class="qitem">sin resultados</div>`;
  if (rows.length && !rows.some(r => r.qid === sel)) select(rows[0].qid);
}
function select(qid) {
  sel = qid;
  const r = D.rows.find(x => x.qid === qid);
  document.querySelectorAll(".qitem").forEach(e =>
    e.setAttribute("aria-selected", e.dataset.qid === qid));
  if (!r) { document.getElementById("panel").innerHTML = ""; return; }
  const ans = r.answers.map(a => {
    const cc = a.claim_counts || {};
    const counts = ["supported","contradicted","not_in_record","unverifiable"]
      .filter(k => cc[k]).map(k => `${k}=${cc[k]}`).join(" · ");
    const claims = (a.claims || []).map(c =>
      `<li><span class="c-${esc(c.verdict)}">●</span><span>${esc(c.text)}</span></li>`).join("");
    return `<div class="ans">
      <h3><span class="arm">${esc(a.arm)}</span>
        <span class="pill v-${esc(a.verdict || "not_scored")}">${esc(a.verdict || "sin veredicto")}</span>
        <span class="meta">${esc(a.model)} · evidencia ${esc(a.evidence ?? "ninguna")} · prompt ${esc(a.family)}${a.served_by?` · ${esc(a.served_by)}`:""}</span>
      </h3>
      ${a.error ? `<p class="c-contradicted">${esc(a.error)}</p>`
                : `<p>${esc(a.text)}</p>`}
      ${counts ? `<div class="meta">${esc(counts)}</div>` : ""}
      ${claims ? `<ul class="claims">${claims}</ul>` : ""}
    </div>`;
  }).join("");
  document.getElementById("panel").innerHTML = `
    <p class="qtext">${esc(r.question)}</p>
    <div class="meta" style="margin-bottom:10px">${esc(r.species)} · ${esc(r.qtype)}
      · nivel de clave ${esc(r.key_level)} · estrato ${esc(r.stratum)}</div>
    <details open><summary>clave</summary>
      <div class="keybox">${esc(JSON.stringify(r.key, null, 1))}</div></details>
    ${r.pack ? `<details><summary>paquete de evidencia (nivel completo)</summary>
      <div class="packbox">${esc(r.pack)}</div></details>` : ""}
    ${ans}`;
}
document.getElementById("qlist").addEventListener("click", e => {
  const it = e.target.closest(".qitem"); if (it && it.dataset.qid) select(it.dataset.qid);
});
for (const id of ["f-qtype","f-verdict","f-text"])
  document.getElementById(id).addEventListener("input", renderList);
renderList();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the run report")
    ap.add_argument("--part", default="pilot", choices=["pilot", "main"])
    ap.add_argument("--no-packs", action="store_true",
                    help="skip evidence packs (faster, but the page loses the "
                         "side-by-side that catches checker defects)")
    args = ap.parse_args()

    data = collect(args.part, with_packs=not args.no_packs)
    if not data["rows"]:
        raise SystemExit(f"no answers for part={args.part}")
    out = DATA / f"report_{args.part}.html"
    payload = json.dumps(data, ensure_ascii=False).replace("</script>", "<\\/script>")
    out.write_text(PAGE.replace("__DATA__", payload), encoding="utf-8")
    kb = out.stat().st_size // 1024
    print(f"{data['totals']} -> {out} ({kb} KB)")


if __name__ == "__main__":
    main()
