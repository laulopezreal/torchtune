# Project Imperium — Instructions Index (distilled for reuse)

Source: `M/instructions broken/part-{1,4,5,6,7,8,12,13}.pdf`
(parts 2, 3 marked `ZZZZ NOP` and skipped). Last upstream update: 2026-05-16.

Read this first. It captures every rule we need from the 287-page guide.
If something here conflicts with the upstream PDF, the PDF wins — but call
that out before changing the index.

---

## Part ↔ Section map

| Part | Section | Topic | Pages |
|---|---|---|---|
| 1   | §1–3  | Overview, Project Expectations, Payments | 7 |
| 2   | §4    | Changelog (ZZZZ NOP — skip) | 2 |
| 3   | §5–6  | LLM Usage, Task Creation & Scope (ZZZZ NOP — skip) | 5 |
| 4   | §7    | End-to-End Task Workflow | 4 |
| 5   | §8    | Task Claiming & Setup | 5 |
| 6   | §9    | Repo Exploration & Task Scope | 5 |
| 7   | §10–11| Prompt & Golden Answer Writing + Rubric Writing & Difficulty Calibration | 19 |
| 8   | §12   | Rubric Design 101 | 5 |
| 9   | §13a  | Example Task 5368 — RCA Medium | 43 |
| 10  | §13b  | Example Task 5680 — PR Triage Medium | 57 |
| 11  | §13c  | Example Task 5354 — Onboarding Hard | 67 |
| 12  | §13d  | Example Task 5137 — Architecture Hard ← closest ref for Task 6600 | 63 |
| 13  | §14   | Reviewer Guidelines | 5 |

---

## Project gist

Authors write **codebase-grounded** SWE Q&A tasks against a fixed repo at a
fixed commit. Tasks evaluate whether models can answer realistic
engineering questions **using the repo**, not from generic knowledge.

### The four intents (Part 6 / §9)

| Intent | Question shape | Has `pr.patch`? |
|---|---|---|
| **Root Cause Analysis** | "Why does X behave this way?" — observed symptom, hidden mechanism | Yes |
| **PR Triage & Impact** | "What does this PR change and what could break?" — blast radius | Yes |
| **Code Onboarding & Comprehension** | "Walk me through how X works" — multi-module flow | No |
| **Architecture & System Design** | "How should I extend / restructure for feature X?" — where it fits + config + tradeoffs | Sometimes |

### Deliverables per task

- `prompt_statement.md` — the question, repo-specific, no answer hints, no large pasted code.
- `golden_answer.md` — expert answer, conclusion-first, names files/functions/flows, supports rubric atoms.
- `rubric.json` — tagged + weighted criteria with rationale + dependencies.
- `task_metadata.json` — `{"category": <intent>}`.
- `Dockerfile` — clones repo at BASE SHA (prefilled by scaffold; do not edit unless flagged).
- `pr.patch` — only for RCA / PR-triage / some architecture-with-patch tasks.

---

## Workflow lifecycle (Part 4 / §7)

1. Writer claims task → confirms metadata.
2. Writer explores repo at BASE SHA → finds repo-grounded question.
3. Writer drafts prompt + golden answer + rubric together.
4. Writer runs Airtable QC tools (Prompt QC, Rubric QC, Golden QC, Validate Golden, Step 5 eval) — apply *human judgment* to outputs.
5. Writer calibrates difficulty (Step 5 pass rate must land in band).
6. Writer submits → reviewer checks → either approved, edited, or sent back.
7. Approved task is finalized ("merged").

### Pay structure (Part 1 / §3)

- $120 / merged task (0–3 revision cycles); $0 if ≥4 cycles.
- One-shot bonus: +$30 Easy/Medium, +$50 Hard.
- $20/month LLM-tool reimbursement after first merge.
- Target: ≥5 merged tasks / week; full cycle within 5 hours; initial draft 3–4 hours.

---

## Prompt rules (Part 7)

- Ask **one** clear engineering question.
- Name useful starting points (files/modules/symptoms) without naming the answer.
- Never paste large code blocks or answer-path conclusions.
- Match the assigned intent (RCA prompt ≠ architecture prompt).
- Use realistic developer language ("I'm trying to understand…", "What should I flag before approval?"). No "world-class architect" role-play.
- Preserve answerability: include the symptom / patch / area-of-uncertainty needed for the question to be solvable.

### Hard-task hardening (Part 7, end)

- Research before writing — know enough to ask about a non-obvious interaction.
- Give fair starting context but preserve discovery (don't name every file).
- For PR triage: trace indirect effects, not diff summary.
- For architecture: ground in real constraints (recurring user need, current limitation).
- For onboarding: ask about *how several pieces work together*, not single-function definitions.
- For RCA: distinguish plausible-wrong explanations; hide file/line.

---

## Golden answer rules (Part 7)

- Start with the answer (conclusion-first).
- Name the files/functions/modules that prove the claim.
- Explain the causal chain — not just "uses X" but *why X produces the observed outcome*.
- Cover plausible wrong alternatives when they clarify.
- Write so each element can become an **atomic rubric criterion** (separate mechanism, files, contrast path, implications).
- Avoid LLM-style filler; repo-grounded prose only.

---

## Rubric rules (Part 7 + Part 8)

### Tags

- **reasoning** — cause, mechanism, tradeoff, control/data flow, architectural implication. "The thinking part."
- **completeness** — required factual elements that must appear. "Discovered as part of thinking but needed."
- **style** — organization, clarity, readability. **Never critical, never penalty.**

### Weights

- **critical** — core correctness; absence makes the answer materially wrong.
- **bonus** — useful optional nuance; safe follow-up; commonly used for style.
- **penalty** — harmful false claim / misleading shortcut. **Cannot be a direct reversal of a positive criterion** (double counting).

### Criterion writing rules

- **Atomic** — one observable behavior per criterion. Avoid "and"/"or" except when truly inseparable, or for enumerated alternatives ("a, b, or c").
- **Measurable** — present-tense action verbs (States, Identifies, Explains, Includes, Distinguishes, Penalizes). No "correctly / clearly / good / robust" without an observable anchor.
- **Specific but not over-specific** — mentioning function names is fine; line numbers force the model into one path.
- **Tied to repo evidence** — every positive criterion must be defensible from the codebase.
- **Allow valid alternatives** — at least one critical criterion should accept an equivalent correct path.
- **Qualifiers explicit** — use "all", "every", "each", or "one or more of: a, b, c". Avoid bare comma lists.
- **No placeholders** — every `rationale`, `description` final before submission.
- **Golden alignment** — golden must score 100% on critical, hit at least one bonus, trigger zero penalties.

### Dependencies (`dependent_on`)

- Use only when downstream criterion can't be fairly evaluated without the upstream one (e.g., "describe MinSize behavior" depends on "identify MinSize as the override point").
- Never make a critical dependent on a bonus.
- Leave empty if not needed.

### Criterion limits (Part 7)

| Tag/weight | Range |
|---|---|
| critical | 7–23 (must include both reasoning + completeness as critical) |
| bonus | 4–17 |
| style | 2–6 |
| penalty | 5–10 |
| **Total** | ~35 (40 max for Hard) |

### Rubric JSON shape (from Task 5137)

```json
{
  "id": "reasoning-17738",
  "tag": "reasoning",
  "weight": "critical",
  "description": "Identifies that existing tags already being used by cloud provider node groups can be used…",
  "rationale": "Configuring the schedule is important to control which node group gets which schedule",
  "dependent_on": []
}
```

`id` format: `<tag>-<numericId>`. Numeric ID is arbitrary but unique.

---

## Difficulty bands (Part 7)

| Difficulty | Step 5 pass rate (±3%) | Shape |
|---|---|---|
| Too Easy | >85% | Not valid — surface lookups |
| Easy | 60–85% | 1–2 files, no code execution |
| Medium | 35–60% | Cross-file reasoning, moderate PR impact |
| **Hard** | **0–35%** | Multi-hop, subtle edges, architecture tradeoffs, system-level impact |

### Calibration fix table

| Symptom | Likely cause | Fix |
|---|---|---|
| Too easy | Prompt leaks answer / rubric accepts generic | Remove breadcrumbs; scope to non-obvious interaction; demand deeper evidence |
| Too hard / ambiguous | Missing symptom or file context; rubric demands unsupported detail | Add starting context; relax over-specific criteria |
| Wrong intent | Final task drifted from intent | Retarget prompt or flag task |
| Rubric mismatch | Golden ≠ 100% critical, or alternatives penalized | Rewrite criteria from final golden; add alternative paths |

---

## Task 5137 as a structural template (Part 12)

Architecture Hard reference. Anchor structural decisions here.

- **Prompt:** ~50 words. *"I want to implement the ability to autoscale a cluster depending on time of day. Help with the architecture — which places to add this and how to define the config mechanism."* Names no files; names the feature and the kind of guidance wanted.
- **Golden answer:** ~600 words, sectioned (Core Logic Issues / Configuration Management / Tag Parsing / Inner Workings / Conclusion). Conclusion-first within each section. Cites concrete files (`aws_manager.go`, `aws_cloud_provider.go`, `auto_scaling_groups.go`, `utils/schedule_parser.go`). Defines a struct, walks `strict`/`relaxed` modes, names `Refresh` cadence (~1 min).
- **Rubric:** 23 criteria. ~7 critical reasoning + 6 critical completeness + 5 bonus reasoning + 4 bonus completeness + 2 style bonus + 5 penalty. Penalties target false architectural claims (e.g., "modifying MinSize alone scales the cluster" — wrong), not direct reversals of positives.
- **Step 5 eval:** test model scored 19.4 — comfortably in Hard band (0–35%). Test model passed only 2 criticals (the easiest reasoning step + the dependent MinSize description) and triggered 1 penalty.

### Lessons from Task 5137 QC feedback

- Rubric QC docked points for being **somewhat narrow** on alternatives — "multiple critical items mandate specific mechanisms… rule out alternative configurations that could also meet the prompt." We should write at least one *"any of: X, Y, Z"* critical to absorb valid alternate designs.
- Rubric QC also docked "**overprescriptive**" — too many specific filenames in critical criteria. Lesson: use specific filenames in *some* criticals (proves the answer is repo-grounded) but accept families ("aws_manager.go *or similar cloud-provider manager file*") in others.

---

## Reviewer checklist (Part 13 / §14) — what we must satisfy

Reviewers verify, in this order:

1. **Assignment & repo setup** — repo, commit, language, intent, difficulty, Dockerfile, metadata all consistent.
2. **Prompt quality** — developer-style, repo-specific, fair, concise, intent-aligned, no answer leak, sounds human (non-ASCII = red flag for LLM authorship).
3. **Golden answer** — correct, complete, repo-grounded, causal chain explained, names files/functions, scores 100% on QC.
4. **Rubric quality** — atomic, measurable, tagged + weighted correctly, dependencies logical, fair to alternatives, golden earns full credit.
5. **QC & validation** — all Airtable QC tools run, human judgment applied.
6. **Difficulty & policy** — Step 5 score in band; not LLM-authored (non-ASCII flag); not a duplicate.

Reviewer time budget: 30–45 min. After 3 revision cycles, reviewer either direct-edits or escalates rather than sending back again.

---

## Process anti-patterns to avoid

- Pasting code/the answer path into the prompt.
- Long prompts that name every file (kills discovery).
- Vague criteria ("Explains the issue correctly") — unmeasurable.
- Compound criteria ("Names the handler AND explains routing AND mentions GraphQL").
- Penalty as direct reversal of a positive critical.
- Style criterion marked critical or penalty.
- Non-ASCII characters in prompt/golden/rubric (LLM-authorship red flag).
- Using an LLM to write the rubric (disqualification per Part 8).

---

## Hard Architecture authoring loop (synthesized for Task 6600)

For the methodology of how to **test** prompts against a subagent before committing
to a rubric, see `M/calibration-methodology.md` — the reusable playbook that
evolved during Task 6600.

1. **Pick the tension** (see CLAUDE.md § Design tensions) and the angle.
2. **Write a ~50-100-word prompt** that names the feature/goal and the kind of guidance wanted; do *not* name files.
3. **Draft the golden answer** sectioned conclusion-first, naming ~5–10 concrete files/functions, defining any new types/configs, walking the modes/tradeoffs, anchoring tradeoffs in real risks (blast radius, security, maintenance burden).
4. **Extract rubric atoms** from the golden — one criterion per discrete fact/mechanism/tradeoff. Aim ~23–35 total: 7–10 critical reasoning, 5–8 critical completeness, 4–6 bonus, 2–3 style bonus, 5–8 penalty.
5. **Penalties = wrong design claims**, not "did not include X". Examples in Task 5137: "claims modifying MinSize alone scales the cluster", "claims modes are unnecessary", "claims CA uses a YAML ConfigMap not tags".
6. **Add ≥1 critical with valid-alternatives qualifier** to absorb reasonable alternate designs.
7. **Run Airtable QC**; read outputs; apply judgment.
8. **Run Step 5**; aim for 0–35% pass rate. If too easy, remove answer-path breadcrumbs from prompt and add depth to criteria. If too hard, add starting context to prompt and relax over-specific criteria.
