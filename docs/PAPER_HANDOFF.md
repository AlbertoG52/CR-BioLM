# Paper handoff: the CR-FloraLM question-answering experiment

Everything a fresh session needs to write the paper. Written 2026-10-09, after the
harness was built and shaken down on 5 and 20 species.

Read this with `docs/EXPERIMENT_PLAN.md`, which holds the design rationale, the power
arithmetic and the pre-registered findings. This file is the state of play, the file
map, and above all **why each question is keyed the way it is** — the section of the
paper a reviewer will probe hardest.

---

## 1. What the experiment asks

One question: **when someone asks about a Costa Rican plant, is the answer true?** And
where the published record is silent, does the system know to stop?

Two claims, deliberately separated:

- **C1 (truthfulness).** Answers become less reliable as the species becomes less
  documented, and grounding in the published record removes most of that decay.
- **C2 (answerability).** For each kind of everyday question there is a measurable
  fraction of the flora that no source can settle. A system that abstains there is
  behaving correctly, and an evaluation that scores silence as error is mismeasuring.

C1 needs a model run. **C2 does not** — it is a database query over the catalogue, it is
already measured (sec. 5), and it fits the journal's Biodiversity Diagnoses tradition.

Target venue: *Biodiversity Informatics* (University of Kansas). Draft at `paper2/`,
compiles clean. The journal publishes no LLM-evaluation work at all; the related-work
positioning and four in-journal anchors are already in `paper2/refs.bib`.

---

## 2. Where everything lives

| Path | What it is |
|---|---|
| `benchmark/questions.py` | Generates all questions and keys. **The reasoning in sec. 4 lives here.** |
| `benchmark/regions.py` | One definition of a species' botanical range (Manual ∪ records ≥5) |
| `benchmark/cooccurrence.py` | The co-occurrence question: compatibility key, three-tier name resolution |
| `benchmark/evidence_pack.py` | What a grounded arm is shown, at two levels |
| `benchmark/qmap.py` | The map stimulus for the multimodal arms |
| `benchmark/prompts.py` | One prompt per family; parity is asserted, not trusted |
| `benchmark/arms.py` | The ten arms as a generator × evidence table |
| `benchmark/client.py` | One API call: upstream pin, served-by, cost, reasoning tokens |
| `benchmark/run.py` | Generation. Parallel, resumable, cost-capped, dry-run |
| `benchmark/score.py` | Scoring. Separate stage, separately resumable |
| `benchmark/verify.py` | The two-level checker. **No LLM judge** (sec. 6) |
| `benchmark/analyze.py` | Tables from verdict rows |
| `benchmark/report.py` | Builds the browsable HTML report of a run |
| `benchmark/answerability.py` | C2, catalogue-wide, no model |
| `benchmark/counterfactual.py` | Finding F10: alter one fact, re-ask |
| `benchmark/data/species_sample.json` | The frozen 3×3 stratified sample (18 pilot + 99 main) |
| `benchmark/data/species_20.json` | The 20-species family-balanced subset |
| `docs/EXPERIMENT_PLAN.md` | Design, power, pre-registered findings F1–F10 |
| `paper2/main.tex` | The draft |

Data the experiment runs on: 6,946 species from the *Manual de Plantas de Costa Rica*
(Tomos II–VI), a frozen GBIF download (DOI 10.15468/dl.8yhee8, 999,625 cleaned Costa
Rican records), 143 SINAC protected areas, 28 botanical regions, an SRTM DEM.

---

## 3. Selection criteria

### The main sample — two popularity axes

Species are sampled on **two axes that correlate only moderately** (Spearman 0.46):

- **G — global familiarity**: GBIF occurrence records worldwide. A model's parametric
  knowledge should follow this.
- **L — local evidence**: records in the frozen Costa Rican snapshot. How much
  occurrence data exists to ground an answer.

Tertiles of each give 9 cells, with **equal allocation per cell**, so the off-diagonal
cases — globally famous but barely collected here, or well recorded here and obscure
elsewhere — are as well represented as the diagonal. Population estimates must reweight
by the frame counts the sampler records.

Frame: catalogue species resolving to a GBIF accepted key. Excluded are unnamed
morphospecies ("Licaria sp. 3"), which cannot be asked about, and species sharing an
accepted key with another Manual species, where occurrence evidence cannot be
attributed.

Pilot (18 species) and main (99) are **disjoint**: prompts and rubrics are tuned on the
pilot, and the main run happens once.

### The 20-species shakedown sample

`species_20.json` is a different design: **one random species from each of 20 randomly
drawn families**, seeded for reproducibility. It samples taxonomic breadth rather than
the record-count strata, and covers 7 of 9 strata unevenly (G1L2 has 5, G2L2 has 1).

**This is the right trade for a shakedown and the wrong one for the paper's first two
findings.** F1 and F2 are a regression across the record-count axis; they need species
spread within strata, which family sampling does not give. Use the stratified sample
for any claim about the long tail.

### Sizing the real run

From the pilot's measured discordance (not assumed effect sizes):

| Contrast | Difference | Pairs needed |
|---|---|---|
| D vs A1F (headline) | 47.9 pts | 15 |
| D vs DM (text vs map) | 50.0 pts | 13 |
| D vs E on park questions | 66.7 pts | 9 |
| D vs F (frontier vs open) | 6.2 pts | 123 |
| A1F vs CF (two models) | 10.6 pts | 160 |
| D vs G (model vs template) | 4.2 pts | 186 |

Every headline contrast is already overpowered. **Pairwise power is not what sizes the
run** — the binding constraint is the F1/F2 regression, which needs species spread.
A 100–200 species run is justified by that, not by the paired tests.

Cost is not a constraint: **$0.0013 per paid call**, so 100 species across all arms is
roughly $7.

---

## 4. The question reasoning — read this before writing the methods

Twelve question types, 1,329 questions in Spanish. Seven are field lookups. **Five apply
a rule, and the rules are where a reviewer will push.**

Every key states what it is: `KEY_SCOPE = "record"`. The keys describe what the
*published record* supports, not field truth. A species may well grow somewhere the
Manual never mentions; the key is still correct about the record. Say this in the paper's
own words, not only in the data.

### Protected areas — three question types from one template

The pool is national parks and biological reserves with a DEM elevation profile,
excluding mostly-marine ones. For each park the code computes records inside the
polygon, whether the park's botanical regions intersect the species', and whether the
park's elevation span (p05–p95) overlaps the species' range including outliers.

- **sí** — at least one record inside since 1970
- **posible** — no records at all, but regions overlap *and* elevations overlap
- **improbable** — no records, and neither overlap holds

A park matching none of these cleanly — region overlap without elevation overlap — is
**not used**; an ambiguous key scores noise.

The rule deliberately refuses to read absence of records as absence of the species.
Collection effort in Costa Rica is uneven, so "posible" is the honest verdict for a park
nobody has botanised. This is the reasoning the text pack carries explicitly and a map
cannot (sec. 7).

**Known weakness:** `park_unlikely` does not discriminate. In the pilot every arm scored
about 4 of 4 on it, including the closed-book model that scored 0/4 and 0/5 on the other
two park types — "improbable" is the safe answer when you know nothing, so the
uninformed guess coincides with the key. Report it separately or drop it; it dilutes the
ablation.

### What counts as a species' range

**The Manual's regions plus any region holding at least five cleaned records.** Same
threshold the validated expert map uses (`renderer._GBIF_CLUSTER_MIN = 5`), and that
renderer was expert-validated for the BIP paper — cite it rather than defending the
number.

*Anthurium ravenii* is listed from four Caribbean lowland regions while 103 of its
records sit in Osa-Golfito. Keying a park question there "improbable" would be keying
against the evidence. 50 of 117 sampled species widen; 10 of 300 park questions change.

**The principle, and it matters for the paper:** occurrence evidence never *overrules* a
stated fact, but it does *extend* one. For elevation the records give a competing range —
423–1248 m against the Manual's 950–1050 — and two ranges cannot both be the answer, so
the Manual wins and the records are withheld from the pack. Regions are additive: "also
here" does not contradict "here", so the union is coherent and strictly better evidence.
One definition in `benchmark/regions.py`, read by the keys, the pack, the map and the
co-occurrence check, so a key and the evidence it is scored against cannot drift apart.

### Cultivation

The question names one of eleven real settings with its elevation and slope — Puntarenas
at 10 m, Monteverde at 1,400 m, Volcán Irazú at 2,800 m.

- **probable** — site elevation inside the Manual's range and the slope matches
- **incierto** — inside the outlier range, or inside the core range on the wrong slope
- **improbable** — otherwise

The key carries an explicit caveat: *growing outside the natural range under cultivation
is NOT tested — an answer saying so is not wrong*. A gardener can keep a cloud-forest
orchid alive at sea level in a greenhouse; the key is about natural occurrence.

**Open question for a botanist:** whether a slope mismatch should really downgrade a
within-range elevation to *incierto*.

### Growth form

Read from the first 160 characters of the species' own description plus a family
default. Only when that yields nothing does it fall back to the genus description, **and
only when the genus names a single growth form**.

That restriction exists because genus descriptions list every form in the genus. Without
it, *Stellaria media* — chickweed — came out as a vine and an epiphyte. 13 of 97 keys use
the genus fallback; whether those are sound is a botanical judgement.

### Endemism

Not read from a field: `global_range` is empty for every species, so the rule reads the
distribution paragraph after the first semicolon and looks for territory outside Costa
Rica (*S Méx.–Col.*, *Nic.–Col.*, *Pantrop.*). 98.4% coverage.

### Edibility — and why the key was wrong

The question asks whether a plant is edible or medicinal. The key originally turned on a
broad use-cue regex that also matched *madera*, *ornamental* and a bare *se usa*, so
eight of nine "the Manual states a use" cases were nothing of the kind. *Dalbergia
retusa* was keyed as stating a use because its entry mentions rosewood timber, and a
system that correctly declined on edibility scored `understated`.

**The paper reports 1.9% (133 species) for this question**, not the 3.6% (249) that
"any use mentioned" gives. Both numbers are real; they measure different things, and the
answerability table should carry the broad one while this question carries the narrow
one.

### Co-occurrence — the key is compatibility, not a gold list

*"Si encuentro X en el campo, ¿qué otras plantas podría ver en el mismo sitio?"*

A gold list is untestable. Species sharing a region and an overlapping elevation band
number **1,900–3,200 — about a third of the flora** — and occurrence data does not
tighten it: co-recorded species in the same grid cells run 1,200–2,500, because the cells
are coarse and the well-collected ones hold thousands of plants. Almost any real Costa
Rican plant would score correct.

So the key inverts: not whether a named plant is on a list, but whether it is **possible
at all**. Co-occurrence is common; incompatibility is rare and mechanical to detect.
Scored on **precision over what the answer volunteers, never recall**.

Each named species resolves into one of three tiers, with synonyms resolved first:

| Tier | Meaning | Scored |
|---|---|---|
| assessable | in the catalogue with regions and elevation | compatibility decided |
| outside_scope | real and GBIF-recorded in CR, outside our volumes | not held against the answer |
| unrecognised | in neither reference | reported, **never** called fabrication |

The middle tier exists because the first version scored absence from the catalogue as a
fabricated name — but *Rhizophora mangle* is a common Costa Rican mangrove and is absent
from it, and the catalogue holds **zero fern species**. The ingested volumes are not the
whole flora. The second reference is the GBIF snapshot's 13,504 species.

The subject is excluded from its own neighbour list: almost every answer restates the
species it was asked about, and a plant is trivially compatible with itself.

### Light and uses — the types with no key

`light` is stated for **0 of 6,946** species; use for 1.9%. These are not defects in the
benchmark, they are the measurement for C2, and they test whether a system knows to stop
rather than whether it knows an answer. `light` currently scores `not_scored` and becomes
usable only with expert labels; if those do not arrive, reporting light preference as
unanswerable is a legitimate result.

---

## 5. Results already in hand

### C2 — answerability, catalogue-wide, no model

`benchmark/answerability.py` imports the key predicates from `questions.py` rather than
reimplementing them, so the table describes the benchmark that exists.

| Question type | Settled | Of | |
|---|---|---|---|
| elevation, cultivation | 6,912 | 6,946 | 99.5% |
| vertiente | 6,883 | 6,946 | 99.1% |
| endemism | 6,838 | 6,946 | 98.4% |
| flowering | 6,626 | 6,946 | 95.4% |
| park_possible | 6,209 | 6,917 | 89.8% |
| park_unlikely | 5,873 | 6,917 | 84.9% |
| growth_form | 5,855 | 6,946 | 84.3% |
| park_confirmed | 5,458 | 6,917 | 78.9% |
| **outside_record** | **133** | 6,946 | **1.9%** |
| **light** | **0** | 6,946 | **0.0%** |

The Manual settles where a plant grows for almost every species and what it is good for
for almost none. A benchmark of elevation questions alone would score every system at
ceiling and miss this.

### C1 — the five-species pilot (directional only)

48 scoreable questions per arm. Treat as direction, not measurement.

| Arm | Evidence | Precision | 95% CI | Coverage |
|---|---|---|---|---|
| G template | full pack | 100% | [93, 100] | 100% |
| **D** Mistral | full pack | **97.9%** | [89, 100] | 97.9% |
| F Gemma | full pack | 89.6% | [78, 95] | 100% |
| E Mistral | flora only | 83.3% | [70, 91] | 100% |
| A1 Mistral | none, careful prompt | 59.5% | [43, 74] | 77.1% |
| CF Gemma | none, free prompt | 58.7% | [44, 72] | 100% |
| **A1F** Mistral | none, free prompt | 51.1% | [37, 65] | 100% |
| C Gemma | none, careful prompt | 100% | [61, 100] | **12.5%** |
| FM Gemma | **map only** | 50.0% | [29, 71] | 100% |
| DM Mistral | **map only** | 44.4% | [25, 66] | 100% |

**The ablation landed exactly where pre-registered.** D and E are identical on every
Manual-stated type and diverge only on `park_confirmed` (3/3 vs 1/4) and `park_possible`
(5/5 vs 1/5). The evidence layer answers what the book cannot; it does not improve
accuracy generally. Closed-book scores 0/4 and 0/5 on those same rows.

**Arm G's 100% is not a result.** It implements each key's rule over the key's own
inputs — a consistency check on the keys and a ceiling on what the evidence layer can
deliver, not evidence that the task is easy.

---

## 6. The evaluation, and why there is no judge

Paper 1 scored answers with an LLM judge ensemble and needed a self-enhancement guard.
**Paper 2 has none.** Nothing scores quality.

1. **Level 1** — the key is typed data. A local model extracts *what the answer
   asserted*; deterministic parsers and a stance table decide correctness against the
   key. The model reports; it does not adjudicate.
2. **Level 2** — a local model splits the answer into claims. Every number, month, slope
   and growth form is re-parsed from the claim *text* by regex. The model's own judgement
   of a claim is never used.
3. **Humans** adjudicate only the residue: sampled `unverifiable` claims, where a
   botanist separates real knowledge the record lacks from invention.

The only model in the evaluation path is a **segmenter**, it is neither generator, and it
assigns no score. That retires the whole LLM-as-judge objection family — verbosity bias,
position bias, self-enhancement — and is worth one sentence in the methods for exactly
that reason.

Verdicts: `correct · partially_correct · overstated · understated · wrong · abstained ·
beyond_record · not_scored`. Claims: `supported · contradicted · not_in_record ·
unverifiable`. The `contradicted` / `not_in_record` split is load-bearing: absence from
the record is not absence in the field.

**Reported together, never alone:** precision, coverage, contradiction rate, correct
silence. Arm C is the standing argument — 100% precision on 12.5% coverage.

---

## 7. Findings worth writing up

Beyond the ablation, three the pilot produced:

**The careful prompt was suppressing the behaviour under study.** The shared prompt told
closed-book models to decline rather than guess and not to invent places, elevations,
months or uses — close to an instruction not to answer, since the questions ask for
exactly those. Gemma declined 42 of 48. With a minimal prompt both models answered
everything, and the invention appeared: asked whether *Aspidosperma cruentum* is edible,
the free model called it **"cocobolo"** and described bark infusions for fever, malaria
and asthma. Our own vernacular index maps cocobolo to *Dalbergia retusa*; this species'
recorded common name is *Bayo rojo*. For *Clidemia gracilis* it invented two common names
absent from the index plus saponins, soap use and treatments for diarrhoea. **A model
told not to invent cannot show the invention the paper is about.** Both prompt families
are reported.

**The map is a worse interface, and the reason is specific.** On the 18 questions a map
can carry, the same model scores 100% from the text pack and 44.4% from the map. But the
failure localises: `vertiente` 4/5, `park_unlikely` 3/4, `park_confirmed` 1/4,
**`park_possible` 0/5**. The dominant failure is `understated` — 7 for Mistral, 9 for
Gemma, zero wrong from Gemma. The map does not make a model hallucinate; it makes it
cautious. Reading the answers shows why: the map arm correctly saw the records outside
the boundary and said so. What it could not see is the reasoning licence — *absence of
records is not evidence of absence*, which the text pack states and a figure cannot.
**A serialized pack can attach an epistemic caveat to evidence; a picture cannot.** That
names a mechanism and predicts where maps do fine, which the two passing rows confirm.

**Co-occurrence discriminates.** First run, three species: grounded 3/3 compatible on all
three; free closed-book 1/2, 1/2, 0/1. It reached for *Quercus costaricensis* twice — a
well-known oak at 2,300–3,600 m — for subjects at 1,150–2,200 m and 1,550 m. Naming the
charismatic species and ignoring the elevation is the failure the question was built to
catch, and it echoes the charismatic-species bias Uryu (2025) reports for IUCN data.

---

## 8. The defect log — this is a paper section

Ten defects surfaced during the shakedown; **six would have reached a paper**, and most
were invisible in summary tables. The commit history carries each with its reasoning.
Together they make the argument that an evaluation of this kind needs a read-the-answers
step, not only an aggregate pass.

| # | Defect | Why it mattered |
|---|---|---|
| 1 | Checker scored the *farm's* elevation against the species | Cultivation questions state the site elevation; a good answer restates it, and every arm that did was marked `contradicted`. Penalised exactly the right behaviour. |
| 2 | Endemism answer quoted the whole distribution tail | Dragged Costa Rican localities into a "not endemic" justification, where they read as unsupported claims |
| 3 | Empty completion recorded as an answer | The paid model reasons: 301 of 335 completion tokens. At a tight budget content came back empty, and an empty answer scores as an abstention — a silent data-quality failure |
| 4 | Counterfactual pack contradicted itself | Shifted the structured elevation but not the Manual paragraph, so a model correctly following the prose would be recorded as ignoring the evidence |
| 5 | Silence metric counted the wrong label | Read 0% while the template was declining every silent question |
| 6 | **A denial of use scored as an assertion** | *"No se menciona ningún uso medicinal"* is the correct answer. All nine such answers were scored `beyond_record`, producing an "80% invention rate" I reported as the run's most interesting finding. It was one line of code. |
| 7 | Errored verdicts counted as done | 20 timeout rows would never have been retried; affected arms would silently carry fewer observations |
| 8 | Elevation band never drew | `mask()` asked to fill an int16 DEM with NaN. 2.4% of the raster sat in the band that came back empty, and a blanket `except` swallowed it |
| 9 | **The map was missing half the evidence** | A park key turns on region overlap *and* elevation overlap. The stimulus drew the band but left every region identical, so the comparison measured the missing layer rather than the interface |
| 10 | **One shared random stream** | Changing the park rule altered how many draws it consumed and silently rewrote 60 cultivation questions. *Mimosa asperata* moved from Volcán Irazú at 2,800 m to Puntarenas at 10 m — both keyed correctly, as answers to different questions |

Defects 6 and 9 are the ones to lead with: both produced plausible numbers, and both were
found by reading answers beside their keys.

---

## 9. State of play, and what the big run needs

**Done:** catalogue, evidence providers, 1,329 questions with typed keys, the two-level
checker calibrated 24/24 and 12/12, ten arms, parallel resumable generation, scoring,
analysis, the HTML report generator, C2 measured catalogue-wide.

**Running now:** 20 species × 236 questions × 10 arms, ~$1.50.

**Not done, in priority order:**

1. **Validate the checker on 40 hand-labelled answers.** The highest-value item, because
   six of ten defects were found by reading answers. An aggregate-only run hides the next
   one. This is Figure 10 and without it a reviewer discounts every other number.
2. **Run the counterfactual (F10).** Built and verified on 75 perturbable pilot
   questions, never actually run. The mechanical check on whether the grounded arm *uses*
   the pack or merely cites it.
3. **Expert light labels** — 117 species, one item each, plus 40 double-labelled for a
   kappa. Turns `light` from `not_scored` into a scoreable type.
4. **Arm B (web search) is unproven.** OpenRouter no longer exposes `:online` ids; search
   is a request-time plugin whose backend becomes part of the system under test. Verify
   it or drop the arm and say so.
5. **Caption the map with the caveat** and re-run the 18 spatial questions. If adding
   *"la ausencia de registros no es evidencia de ausencia"* closes the 0/5 gap on
   `park_possible`, the F8 finding sharpens from "maps are worse" to a named mechanism.

**For the 100–200 species run specifically:** use the stratified sample, not the family
one. Keep the machine quiet while scoring — a background game cost a 12× slowdown, 3.4
tokens/s against 41 warm and idle. Generation is parallel and resumable; scoring is
deliberately serial because it is GPU-bound. Budget roughly $7–15 and expect at least one
re-run after a defect, since there have been ten so far.

**A data defect to report upstream:** two distinct SINAC polygons are both labelled
*Parque Nacional Tortuguero*. A park question could be generated against either, and
`where_to_see` lists it twice.

---

## 10. The gap worth naming in the paper

**The benchmark cannot see what makes an answer useful.** Arm G answers "Es posible" in
one sentence; arm D answers with two overlap facts, the park's elevation profile, and the
location and date of every known record. Both score `correct`. Every key is a single
value, so an answer carrying the nearest records, the flowering month and the specific
sector scores identically to a bare verdict.

Two things follow, and both are future work the paper should name rather than hide:

- An **informativeness axis** scored separately from correctness. ReCEval separates
  whether a reasoning step is correct from whether it is informative, and the prior-work
  review flagged this as the dimension our design lacks. "Grounded, correct and vacuous"
  is a real cell we currently cannot see.
- A ***"¿dónde y cuándo debería buscar esta planta?"* question type.** The system's most
  useful function, the provider for it already exists (`places.where_to_see`), and it is
  not in the benchmark at all.

On positioning: the prior-work review returned a blunt verdict worth respecting. Scoring
premises separately from inference is **not novel** — it is attribution evaluation,
entailment trees and Claim–Evidence–Reasoning rubrics in different clothes. What *is*
novel is that the target proposition is absent from the entire corpus rather than from a
retrieved passage, that the inferences are defeasible rather than deductive, and that
biodiversity informatics has nothing comparable: current LLM work on floras **extracts
stated** traits, while trait imputation infers unstated values statistically with no
derivation to audit. The risk is an NLP-literate reviewer, not an ecology one. Say it
first.
