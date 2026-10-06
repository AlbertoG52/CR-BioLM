# The experiment: what we run, what we compare, what we show

Protocol for the Paper 2 / thesis experiment. Written to be executable: every arm,
metric and figure says what it needs and whether that exists today.

Status legend used throughout:
`READY` exists and verified - `BUILD` must be written - `COST` needs API spend -
`EXPERT` needs a human - `OPTIONAL` nice to have, drop under time pressure

---

## 1. The claim the experiment has to support

Two separate claims, deliberately not merged:

**C1 (truthfulness).** For a tropical regional flora, in Spanish, a language model's
answers to everyday plant questions become less reliable as the species becomes less
documented, and grounding the answer in the published record removes most of that
decay.

**C2 (answerability).** For each kind of everyday question, there is a measurable
fraction of the flora that no source can settle. A system that abstains there is
behaving correctly, and any evaluation that scores silence as error is mismeasuring.

C1 needs a model run. **C2 does not** - it is computable from the catalogue today, and
it is the half that fits the journal's Biodiversity Diagnoses tradition. If the API
budget or the calendar collapses, C2 still stands as a paper.

---

## 2. Design: one factorial table, not a list of systems

The arms are a 2-factor design - **generator** x **evidence** - because that is what
makes the comparison interpretable. A list of seven named systems invites the reviewer
to ask what differs between any two; a factorial table answers it.

|                           | no evidence | live web | flora text only | full evidence pack |
|---------------------------|-------------|----------|-----------------|--------------------|
| **frontier model**        | A1          | B        | E               | D                  |
| **second frontier model** | A2          | --       | --              | -- `OPTIONAL`      |
| **local open weights**    | C           | --       | --              | F                  |
| **template, no LLM**      | --          | --       | --              | G                  |

What each arm is for:

| Arm | System | Answers the question |
|-----|--------|----------------------|
| A1 | Frontier model, closed book | What does a model know about Costa Rican plants unaided? The ChatGPT baseline. |
| A2 | A different frontier vendor, closed book `OPTIONAL` | Is the closed-book result vendor-specific or a property of the long tail? |
| B | Frontier model + web search | What does a person actually get today? The honest competitor. |
| C | Local open-weights, closed book | Reproducible floor, no proprietary service. |
| D | **Ours**: frontier generator over the full evidence pack | Does grounding fix it? |
| E | Ablation: same generator, flora text only | How much do occurrences, parks, regions and DEM add over the book alone? |
| F | Local generator over the full evidence pack | **Can a small local model, grounded, match a frontier model?** |
| G | Template answerer, no LLM | How much of the benchmark needs generation at all? The floor nobody can argue with. |

Three of these are worth stating plainly because they carry the paper:

- **D vs E** is the thesis's own contribution isolated. Everything built past the
  Manual extraction - GBIF cleaning, SINAC polygons, botanical regions, DEM, vernacular
  layer - is justified or not by this one contrast.
- **F vs D** is the result a Costa Rican institution actually needs. If grounding lets
  an 8B model on a mini-PC match a frontier API, that is the deployable finding, and it
  reframes the paper from "LLMs are unreliable" to "here is what to run instead".
- **G** is the floor. If G is within a few points of D on most question types, the
  honest conclusion is that much of this needs a database and not a language model, and
  saying so is a stronger paper than hiding it.

### Optional arm worth considering

`OPTIONAL` **D+SDM**: the full pack plus the random-forest suitability surface as an
extra evidence provider, used only by the `park_possible` and `cultivation` question
types. This is the one place the thesis's modelling work can be put *inside* the QA
system rather than sitting beside it. It makes the thesis one object instead of two.
Cheap to add once D exists: same prompt, one extra evidence block.

---

## 3. What is held constant (the confound that already bit us once)

The tier-ablation prompt bug is the precedent: arms differed in more than the thing
being ablated, so the ablation measured the prompt, not the evidence.

Rules the harness enforces, not the operator:

1. **One prompt template** for every LLM arm. The evidence block is swapped; nothing
   else - not the instructions, not the output format, not the abstention wording.
2. **Identical abstention licence** in every arm. Every LLM arm is told it may say it
   does not know. If only the grounded arm is given permission to abstain, the
   abstention comparison is meaningless.
3. **Identical output contract** (answer prose; no JSON schema differences between arms).
4. **Same questions, same order, same seed.** Paired design throughout, so every
   comparison is within-question.
5. **Temperature 0** for the headline run; replication handled separately (sec. 4).
6. **The claim extractor used by the verifier must not be any arm's generator.**
   Today `verify.EXTRACTOR = ollama:qwen3-vl:8b-instruct`; arm C/F must therefore use a
   different local model, or the extractor must change. Unfixed, a reviewer will call
   this self-evaluation. `BUILD`
7. **Evidence pack is versioned and hashed** per question, stored with the answer, so a
   re-run is provably the same evidence.

---

## 4. Run units, replication, scale

Unit of observation: one `(question, arm, replicate)` -> one answer -> one verification.

| | Pilot `READY` | Main `READY` |
|---|---|---|
| Species | 18 | 99 |
| Questions | 190 | 1,020 |
| Strata covered | 8 of 9 | 9 of 9 |
| Disjoint from the other | yes | yes |

Question inventory (pilot counts; main scales about 5.4x):

| Question type | n | Key level | What settles it |
|---|---|---|---|
| elevation | 18 | stated | Manual |
| vertiente | 18 | stated | Manual |
| flowering | 18 | stated | Manual |
| endemism | 18 | stated | Manual range statement |
| growth_form | 15 | stated (10) / inferred_relatives (5) | Manual, genus only when unambiguous |
| park_confirmed | 16 | stated_by_records | snapshot records inside the polygon |
| park_possible | 17 | inferred_data | region + elevation overlap, no records |
| park_unlikely | 16 | inferred_data | no overlap, no records |
| cultivation | 18 | inferred_data | elevation/slope rule |
| light | 18 | **pending_expert** | nothing in the record - experts label |
| outside_record | 18 | none (15) / stated (3) | nothing - **abstention is correct** |

Two of eleven types have no key in the record. That is not a defect of the benchmark;
it is the measurement for C2, and it is why `light` and `outside_record` exist.

**Replication.** Temperature 0 is not determinism. Report run-to-run variability from
`k = 3` replicates on a fixed 60-question subset (balanced across types and strata),
not on the whole benchmark. One number in the methods - "max swing across replicates
was X points" - buys the credibility without 3x the budget.

**Scale and wall-clock**, the part that actually constrains the calendar:

| Stage | Units | Where it runs | Estimate |
|---|---|---|---|
| Pilot generation, LLM arms | 190 x 6 = 1,140 | mixed | hours |
| Main generation, paid arms (A1, D, E full; B, A2 subset) | 3,740 | API | hours |
| Main generation, local arms (C, F) | 2,040 | mini-PC | about 10 h, run overnight |
| **Main verification, every arm** | **5,780** | local extractor | **about 8 h** |

The verification pass is as expensive in wall-clock as the generation. Budget it, keep
it a **separate resumable stage** from generation, and never re-generate to re-verify.

**Cost** (estimates at mid-tier frontier pricing, to be replaced with metered actuals):

| Stage | Estimate |
|---|---|
| Pilot, all frontier arms, k=1 | about $11 |
| Pilot variance subset, k=3 | about $3 |
| Main, all frontier arms | about $60 |
| Second-vendor closed book (A2) | about $6 |
| **Total** | **about $80** |

Web search (arm B) dominates per-question cost and is the least predictable; meter it
on the pilot before committing to the main run.

---

## 4b. How many questions actually get executed

### Inventory

| | Count |
|---|---|
| Distinct questions written `READY` | **1,210** (190 pilot + 1,020 main) |
| Species | 117 (18 + 99), 11 per stratum in the main sample |
| Question types | 11, averaging 93 questions each in the main set |
| Main questions scoreable today | 921 |
| Main questions awaiting expert labels (`light`) | 99 |
| Main questions where abstention is the correct answer | 93 |

Nothing in the experiment runs over 6,946 species. The catalogue-wide number appears
only in the answerability analysis (sec. 6), which is a database query, not a model run.

### Which arms need the full set, and which do not

Counterintuitive, and it decides the budget. Pairs needed for 80% power at alpha 0.05,
paired (McNemar), under the effect sizes we expect:

| Contrast | Expected difference | Pairs needed |
|---|---|---|
| D vs A1 — grounded vs closed book | 30 pts | **28** |
| G vs D — template vs grounded | 10 pts | 194 |
| D vs B — grounded vs web search | 15 pts | 102 |
| F vs D — local vs frontier, both grounded | 6 pts | 434 |
| D vs E — full pack vs flora text only | 7 pts | 350 |
| D vs E — if the gap is only 5 pts | 5 pts | **626** |

The headline result needs almost no data. The **ablation and the local-model contrast
need nearly the whole set**, because they are the small effects. So the arms to put on
a subset are the ones with large expected effects, not the ones that look expensive.

What also forces the full set: the arm x question-type heatmap (Fig. 9) gives 93
questions per type at n = 1,020 and only 37 at 40% of it, and the long-tail curve
(Fig. 5) gives 113 questions per G x L cell at full size.

### Recommended main run

| Arm | Questions | Units | Spend |
|---|---|---|---|
| A1 frontier, closed book | 1,020 | 1,020 | paid, cheapest prompt |
| D frontier, full pack | 1,020 | 1,020 | paid |
| E frontier, flora only | 1,020 | 1,020 | paid |
| C local, closed book | 1,020 | 1,020 | free, local |
| F local, full pack | 1,020 | 1,020 | free, local |
| G template, no model | 1,020 | 1,020 | free, instant |
| B frontier + web | **340** subset | 340 | paid, dearest per call |
| A2 second vendor, closed book | **340** subset | 340 | paid |
| | | **6,800 units** | **about 3,740 paid calls, about $42** |

B and A2 drop to a stratified 340-question subset because both contrasts they serve
need around 100 pairs or fewer. B is the most expensive call in the experiment and the
one least worth paying for at full size.

Verification is 6,800 minus arm G, whose answers are deterministic and whose claims are
known by construction: about 5,780 LLM extractions, roughly 8 hours, overnight.

### The pilot sizes the main run (pre-specify this now, not after)

The table above assumes effect sizes. The pilot measures them. Rule, fixed in advance:

1. Run the pilot (190 questions, all arms).
2. From the observed discordance, compute the pairs needed for the **smallest**
   contrast that the paper claims — D vs E.
3. If that number is under 400, run the reduced main: **7 species per stratum, 63
   species, about 650 questions**, saving roughly a third of everything.
4. If it is over 400, run the full 1,020.
5. Either way the decision is recorded before the main run starts, and the main run
   happens once.

This is the difference between choosing a sample size and justifying one. Deciding
after seeing the main-run results is the thing reviewers look for.

---

## 5. Metrics

### Level 1 - the direct answer against the key `READY`

Verdicts already implemented: `correct`, `partially_correct`, `overstated`,
`understated`, `wrong`, `abstained`, `beyond_record`, `not_scored`.

Reported as a pair, never as one accuracy number:

- **Precision when answering** = correct / (all non-abstained, scoreable)
- **Coverage** = scoreable answers given / scoreable questions

Reporting them jointly is what stops the obvious gaming in both directions: a system
that abstains everywhere has perfect precision and no coverage; one that always guesses
has full coverage and poor precision. The paper's headline figure is this plane.

Separately, on the questions where the record is silent:

- **Correct silence** = abstained / `key_level == none` - here abstention *is* the right
  answer, and inventing a medicinal use is the failure.

### Level 2 - every claim against the full record `READY`

`supported` - `contradicted` - `not_in_record` - `unverifiable`

- **Contradiction rate** = contradicted / all claims. The hallucination measure.
- **Support rate** = supported / all claims.
- **Unverifiable rate** - not a penalty. A claim about light or use that the Manual
  cannot judge lands here, and the split between "the model knows something real that
  the record lacks" and "the model made it up" is exactly what the expert adjudication
  (sec. 8) resolves on a sample.

The `contradicted` / `not_in_record` distinction is load-bearing and must survive
review: absence from the record is not absence in the field.

### Covariates carried on every row `READY`

`stratum` (G1-3 x L1-3) - `n_global` - `n_records` - `qtype` - `key_level` -
`key_review_priority` - family

---

## 6. Analyses

**Primary, pre-registered.** Mixed-effects logistic regression over scoreable questions:

```
correct ~ arm + log10(n_global) + log1p(n_records) + (1 | species) + (1 | qtype)
```

Random intercepts because questions repeat within species and within type; treating
1,020 questions as independent would overstate significance. Pre-registered
hypotheses: the `n_records` coefficient is positive and significant for A1 and C, and
the `arm x log1p(n_records)` interaction is negative for D - grounding flattens the
decay.

**Secondary.**

1. Paired comparison of each arm against D on identical questions (McNemar, plus
   paired bootstrap CIs on the precision difference).
2. D vs E: the value of everything past the book.
3. F vs D: can local grounded match frontier grounded.
4. G vs D: how much needs generation at all.
5. Correct-silence rate by arm on `key_level == none`.
6. Contradiction rate by arm and by stratum.

**Exploratory, labelled as such.** Per-question-type breakdowns (11 types x 7 arms is a
multiple-comparison field; no per-cell p-values in the headline claims).

**Answerability (C2), no model involved.** Catalogue-wide, all 6,946 species - measured
today:

| Question type | Settled by the Manual | |
|---|---|---|
| morphology | 6,946 | 100.0% |
| discussion / notes | 6,935 | 99.8% |
| elevation | 6,912 | 99.5% |
| habitat / forest type | 6,901 | 99.4% |
| vertiente | 6,883 | 99.1% |
| botanical regions | 6,762 | 97.4% |
| flowering months | 6,626 | 95.4% |
| growth form, own text | 6,614 | 95.2% |
| fruiting months | 69 | **1.0%** |
| **uses** | **0** | **0.0%** |
| **light preference** | **0** | **0.0%** |

Two findings in that table are the C2 result, and both are honest about the source: the
Manual is a taxonomic treatment, so it settles where a plant grows almost always, and
what it is good for never. A benchmark that asked only elevation questions would score
any system at ceiling and miss this entirely.

Caveat to state in the paper rather than hide: `uses`, `global_range` and
`common_names` are fields the extractor leaves to an LLM pass that has not been run
(`field_extractor.py:10`), so 0% for `uses` currently conflates "the Manual is silent"
with "we have not extracted it". The endemism key already works around this by reading
the distribution paragraph directly. **Before this table goes in the paper, run the
use-mention detector over the full catalogue and report the measured mention rate.**
`BUILD`

---

## 7. Figures and tables

Ordered by what they prove. The "needs" column is the gate.

| # | Figure | What it proves | Needs |
|---|--------|----------------|-------|
| 1 | Pipeline schematic: Manual -> catalogue -> evidence providers -> pack -> arms -> two-level verification | that the test bed is a system, not a script | nothing `READY` |
| 2 | Sampling plane: all 6,946 species as points in log(global GBIF) x log(CR records), 9-cell grid overlaid, 99 sampled species highlighted, Spearman 0.46 annotated | the sample spans the long tail, and the two axes are not redundant | nothing `READY` |
| 3 | **Answerability heatmap**: question type x evidence level, cell = fraction of the flora settled | C2, catalogue-wide, no model | small `BUILD` |
| 4 | **Precision-coverage plane**, one point per arm, D and G annotated | the headline C1 result, ungameable in either direction | pilot run |
| 5 | **Long-tail decay**: x = log local records (binned), y = P(correct) with CI ribbons, one line per arm; second panel for global count | that decay exists and grounding flattens it | main run |
| 6 | Verdict composition: stacked bar per arm (correct / partial / overstated / understated / wrong / abstained / beyond_record) | *how* each arm fails, not just how often - overstatement is the interesting failure | pilot run |
| 7 | Claim-level composition: stacked bar per arm (supported / contradicted / not_in_record / unverifiable) | the hallucination figure | pilot run |
| 8 | **Silence test**: for `outside_record`, bar per arm = abstained vs invented a use | the most legible figure in the paper; a reader needs no metric definition to read it | pilot run |
| 9 | Precision heatmap, arm x question type | where grounding helps (parks, elevation) and where nothing helps (light) | main run |
| 10 | Checker validation: confusion matrix, checker verdict vs human verdict | that the measuring instrument is calibrated - without this a reviewer discounts everything | 40 hand labels `EXPERT` |
| 11 | Cost / latency vs precision per arm | the deployability argument; this journal's readers run tools | metering `OPTIONAL` |
| 12 | Spanish vs English, paired, same questions | whether the language penalty is real | subset run `OPTIONAL` |

Tables: T1 data components `READY` - T2 the design matrix of sec. 2 - T3 question
inventory `READY` - T4 model identifiers, versions and retrieval dates - T5 main
results (coverage, precision, wrong rate, contradiction rate, correct silence, per arm)
- T6 mixed-model coefficients - T7 checker calibration.

**Four of twelve figures and three tables need no model run at all** (1, 2, 3, plus
T1/T2/T3). Build those first: they are the paper's spine, they cost nothing, and they
turn the remaining work into a results section rather than a paper.

---

## 8. Human involvement

Sized so it can actually be asked of busy people.

| Task | Volume | Why it is needed |
|---|---|---|
| Light labels (sun/shade) | 18 pilot + 99 main species, 1 item each | turns `pending_expert` into a scoreable key; without it `light` stays `not_scored` |
| Double-labelling for agreement | 40 species, 2 experts | Cohen's kappa - a key with no agreement estimate is not a key |
| High-priority key review | 6 pilot + 30 main keys | `key_review_priority == high`, where automated extraction is least safe |
| Claim adjudication | about 150 sampled `unverifiable` claims | splits real knowledge beyond the record from invention; this is what makes `unverifiable` publishable rather than a shrug |
| Checker validation labels | 40 answers | Figure 10 |

The light labels and the agreement subset are the only ones on the critical path.

---

## 9. Harness contract `BUILD`

Three stages, each separately resumable, each writing JSONL. Generation and
verification must never be coupled.

```
benchmark/run.py       (qid, arm, replicate) -> answers_<part>.jsonl
benchmark/score.py     answers                -> verdicts_<part>.jsonl
benchmark/analyze.py   verdicts               -> tables + figures
```

Every answer row carries: `qid`, `arm`, `replicate`, `model_id`, `model_version`,
`prompt_template_hash`, `evidence_pack_hash`, `evidence_providers[]`,
`retrieved_urls[]` (arm B), `prompt_tokens`, `completion_tokens`, `cost_usd`,
`latency_ms`, `started_at`, `answer_text`, `error`.

Non-negotiable properties:

- **Idempotent and resumable.** Keyed on `(qid, arm, replicate)`; a re-run skips
  completed rows. A crash at question 800 of 1,020 must not cost the first 800.
- **Append-only.** Never rewrite a row; a changed prompt is a new template hash.
- **Cost ceiling.** Hard abort at a configured spend, default low.
- **Dry-run mode** that prints the exact prompts for one species per stratum without
  calling anything. This is how arm parity (sec. 3) gets checked by eye before spending.
- **Arms as plugins**, one module each, so a new arm cannot perturb an existing one.

---

## 10. Threats to validity and what each requires

| Threat | Handling |
|---|---|
| **Web arm hits our own sources.** The Manual and GBIF are online; arm B may retrieve exactly our evidence, making it look grounded. | Log every retrieved URL; report what fraction of B's answers touched the Manual, GBIF or this project's outputs. Report it even if it is high - it is a finding about what is already reachable. |
| **Frozen snapshot vs live GBIF.** D cites the frozen DOI; B may see newer records. | Quantify: how many of the main-run species gained records since the snapshot. State the direction of the bias. |
| **Keys describe the record, not the field.** | `KEY_SCOPE = "record"` is already carried on every question and must appear in the paper's own words, not only in the data. |
| **Abstention gaming.** | Precision never reported without coverage (sec. 5). |
| **Self-evaluation.** Extractor shares a family with the local arms. | Change the extractor or the local arm model (sec. 3, rule 6). |
| **Benchmark contamination.** Pilot species tuned the prompts. | Pilot and main are disjoint `READY`; the main run happens once, after prompts freeze, and is not iterated on. |
| **Multiple comparisons** over 11 types x 7 arms. | Primary metric pre-registered; per-type analysis labelled exploratory. |
| **Stratified sample is not the flora.** | Reweight to population by the frame counts per cell that the sampler already records `READY`. |
| **Translation effects.** Questions are Spanish; keys derive from Spanish text. | Arm prompts stay in Spanish; the English comparison (Fig 12) is a separate paired analysis, not a substitution. |

---

## 11. Phases and gates

| Phase | Work | Cost | Gate before moving on |
|---|---|---|---|
| **0** | Answerability catalogue-wide; use-mention detector; Figures 1, 2, 3; T1-T3 | none | C2 stands alone as a result |
| **1** | Harness + arms G, C, F. Dry-run prompt parity check. | none | identical prompts across arms, verified by eye |
| **2** | Checker validation on 40 free-arm answers | none `EXPERT` | checker agrees with human labels; Figure 10 |
| **3** | Pilot, frontier arms, k=1, metered | about $11 | prompts freeze; per-question cost known |
| **4** | Expert light labels + high-priority key review | none `EXPERT` | `light` becomes scoreable |
| **5** | Main run, all arms, once | about $70 | - |
| **6** | Analysis, figures, write-up | none | - |

Phases 0-2 cost nothing and need no approval. Phase 3 is the first spend and is the
natural checkpoint to stop and look.

---

## 12. Decisions still open

1. **Which frontier model**, and whether A2 (second vendor) is worth about $6.
   Recommend yes: it is the cheapest defence against "you tested one model".
2. **Which local model** for C and F - and it must differ from the verifier's extractor.
3. **Whether D+SDM becomes an arm.** Recommend yes if time allows; it is what makes the
   thesis a single piece of work.
4. **Spanish-vs-English** (Fig 12): in or out.
5. **Whether `light` ships as a scored type or as a declared gap.** If the expert labels
   do not arrive, reporting it as unanswerable is a legitimate result, not a hole.
