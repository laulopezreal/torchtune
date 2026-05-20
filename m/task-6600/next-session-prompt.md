# Next session — Task 6600 Step 5 review

I'm continuing **Task 6600** (Project Imperium / Agentic Code QA). The deliverables are uploaded to Airtable and I've run **Step 5**. We'll look at the result together and decide what to do.

## Where things live

Repo: `laulopezreal/torchtune` on `main`. Working directory is `m/` (lowercase).

- `m/pytorch-torchtune-architecture/` — the submission (prompt, golden, rubric, task_metadata, Dockerfile)
- `m/task-6600/README.md` — working history overview
- `m/task-6600/prefix-tuning-shipped/run-{1,2,3}*-answer.md` — the 3 calibration runs we hand-graded
- `m/instructions broken/INDEX.md` — distilled Project Imperium rules
- `m/instructions broken/Project Imperium ... part-7.pdf` — prompt/golden/rubric rules (source of truth)
- `CLAUDE.md` (repo root) — Project Imperium context + torchtune design tensions map

**Read these in order before doing anything:** `m/task-6600/README.md`, then `m/pytorch-torchtune-architecture/prompt_statement.md`, then skim `m/pytorch-torchtune-architecture/rubric.json` for the 17 critical criteria.

## Task spec

- Intent: Architecture & System Design
- Difficulty: Hard (Step 5 target = 0-35% pass rate +/- 3%)
- BASE commit: `213f38605ff0b7b1e20f85a9e032710be04c82c9`
- Catalyst: adding Prefix Tuning as a third PEFT method alongside LoRA and DoRA
- The rubric's three reproducible Hard discriminators (missed by every calibration run): **RoPE design fork (1005), PackedDataset packing (1007), loss masking (1008)**

## What I want from you

Ask me for the Step 5 score, then handle the case that matches:

**Case A — Step 5 lands in Hard band (0-35% +/- 3%):**
- Confirm we're good to submit.
- Walk me through the final submission checklist (any last QC tool pass, anything in Airtable I might have missed).

**Case B — Step 5 above 35% (Medium / Easy band):**
- Diagnose with me. First question: did the test model hit any of the three reproducible Hard discriminators (1005 / 1007 / 1008)? If yes, the rubric is being graded too generously — tighten the language on those criteria. If no, the prompt is leaking more context than we thought.
- Per Part 7 calibration table, "too easy" fixes are: remove answer breadcrumbs, require deeper repo evidence, scope toward a non-obvious interaction. Do NOT add context.
- Decide one specific change (rubric tightening, prompt edit, or both) before recalibrating. Don't shotgun multiple changes.

**Case C — Step 5 below 0% (rare, "too hard"):**
- The rubric is demanding unsupported detail or the prompt is missing necessary context. Relax over-specific criteria; add starting context to the prompt.

## Hard guardrails

- The golden + rubric prose are **user-authored**. Do NOT rewrite them for me. Structural feedback, rubric grading, schema validation, mechanical tasks only.
- If I have natural typos in the golden (`I wold`, `proceed as norma`), do NOT fix them — they're a deliberate human-authorship signal per "not overly polished" guidance from the May 19 onboarding meeting.
- Part 7 has NO word-count requirement for goldens. Don't push cuts that lose rubric coverage.
- Don't switch prompts to make calibration land. The locked prompt v1 has the most calibration data (Runs 1-2 at 60-68% hand-grade); a different prompt invalidates that anchor.

## Useful context to surface only if relevant

- 4 calibration runs hand-graded before submission. Runs 1-2 (locked v1): ~60-68%. Run 3 (user variant): ~78-83% (HIGHER not lower — strong models recover dropped hints from code). Run 4 (external fresh-session): ~50-55%.
- Strong-model hand-grade 60-68% maps to roughly Hard-band Step 5 because Step 5 averages across model qualities and weaker models score lower.
- Subagent calibration runs had explicit context isolation (no CLAUDE.md or m/ access).

## First message to me

Open with: **"What's the Step 5 pass rate, and which criteria did the test model hit vs miss?"** Then branch into Case A/B/C above.
