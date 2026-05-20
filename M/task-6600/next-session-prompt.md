# Continuing Task 6600 — Project Imperium Architecture task on torchtune

I'm authoring a Hard Architecture & System Design task for Project Imperium against the torchtune fork at BASE commit `213f38605ff0b7b1e20f85a9e032710be04c82c9`. Catalyst: **adding Prefix Tuning as a third PEFT method** alongside LoRA and DoRA.

## Repo & layout

- Repo: `laulopezreal/torchtune` (fork of pytorch/torchtune), all work on `main`
- **Submission folder:** `M/pytorch-torchtune-architecture/` (prompt, golden, rubric, task_metadata, Dockerfile) — this is what gets uploaded to Airtable
- **Working history:** `M/task-6600/` (README, abandoned FP8 QAT catalyst, shipped Prefix Tuning calibration, candidate prompts)
- Read `M/README.md` and `M/task-6600/README.md` first for orientation
- Instruction rules distilled in `M/instructions broken/INDEX.md`; Part 7 PDF is the source of truth for prompt/golden/rubric rules

## Status snapshot

**Done:**
- Prompt **locked** at `prompt_statement.md` (v1, Prefix Tuning, ASCII clean, 2 paragraphs)
- Rubric structurally complete: 36 criteria (17 critical / 11 bonus / 8 penalty), all Part 7 limits pass, JSON schema validated against Task 5137 reference shape (`id`, `tag`, `weight`, `description`, `rationale`, `dependent_on`)
- 4 calibration runs complete (Runs 1-3 in `M/task-6600/prefix-tuning-shipped/`; Run 4 was an external fresh-session experiment graded inline). Runs 1-2 (v1) scored ~60-68% informal; Run 3 (user variant) ~78-83%; Run 4 ~50-55%
- **Reproducible misses across all runs: RoPE design fork, PackedDataset packing, loss masking** — these are the rubric's critical Hard discriminators (criteria 1005, 1007, 1008)

**Open — authorship risk:**

Per the May 19, 2026 onboarding meeting, AI-authored goldens and rubrics are an offboarding risk (Part 8). The current `golden_answer.md` and the `description`/`rationale` fields in `rubric.json` were AI-drafted. The user (Lau, lauureal@gmail.com) has been iterating on a personal golden through 5 rounds. The latest version (in chat history, not yet saved to file) is ~1800 words, in user's voice, with natural typos intentionally preserved (`I wold`, `proceed as norma`, `cleanup.p`).

## Work remaining, in priority order

1. **Latest "Lau golden" needs three small fixes** to recover lost rubric criteria:
   - Add ~7 file:line references (Part 7 requires ≥10 — current draft has ~5). Critical adds: `position_embeddings.py:69-122` (Fork 2 RoPE — also recovers partial 1005), `_checkpoint_client.py:177-182, 370-372, 543-548` (LoRA-merge section — also recovers partial 1010), `_packed.py:57-67, 139`, `_generation.py:72-112`, `kv_cache.py:32-41`, `convert_weights.py:240-309`, `lora_finetune_distributed.py:566-587`
   - Add 2 more HF PrefixTuningConfig field names (`num_virtual_tokens`, `token_dim`) to recover bonus 3003
   - Optional: rewrite PR 1 / PR 2 / PR 3 prose sections in user's voice (still close to AI version)

2. **Rubric `description`/`rationale` re-authoring.** Keep `id`, `tag`, `weight`, `dependent_on` exactly as-is in `M/pytorch-torchtune-architecture/rubric.json` (schema is validated). Rewrite only the `description` and `rationale` text fields for all 36 criteria in the user's voice.

3. **Save the final Lau golden** to `M/pytorch-torchtune-architecture/golden_answer.md` (overwriting the AI-drafted version).

4. **Upload to Airtable.** It's a TABLE — one row per criterion, NOT a JSON paste field. 36 rows to enter. Columns: `id`, `tag` (dropdown), `description`, `weight` (dropdown), `rationale`, `dependent_on` (linked records). Two dependency links: `reasoning-1004 → reasoning-1003`, `reasoning-1012 → reasoning-1009`. The user has already accidentally deleted a row once — undo with "DESHACER" if needed.

5. **Run Airtable QC tools** (Prompt QC, Rubric QC, Golden QC, Validate Golden). Apply human judgment to outputs (the tools occasionally hallucinate per meeting notes).

6. **Run Step 5** in Airtable. Target: Hard band (0-35% pass rate ±3%). Cannot be run locally.

## Key context not to forget

- **Part 7 does NOT specify a golden word count.** The ~600-word figure that came up earlier was an inference from Task 5137's reference golden. The user's ~1800-word golden is defensible because every paragraph hits at least one rubric criterion. Don't push for more cuts unless rubric coverage holds.
- **Calibration subagents had explicit context isolation** (no `CLAUDE.md` or `M/` access). Reproducible misses (RoPE / packing / loss masking) are structural to the catalyst, not the prompt.
- **Rubric JSON schema was validated** programmatically against Task 5137 shape — all 36 criteria pass. No schema work needed.
- **Run 3 found that dropping prompt scaffolding scored HIGHER, not lower.** Strong models recover dropped hints from code reading. Don't try to make the prompt "harder" by removing context.
- **The locked prompt is v1.** A third variant (the external experiment) scored slightly lower but n=1; don't switch prompts without re-calibrating.

## What I should NOT do

- Do NOT rewrite the golden answer or rubric prose **for** the user. AI-authored = offboarding risk per May 19 meeting. My role: structural feedback, rubric alignment grading, schema validation, mechanical tasks. The prose has to be the user's words.
- Do NOT fix the natural typos in the user's golden — they're a deliberate authorship signal per "not overly polished" guidance.
- Do NOT push for further word-count cuts past where rubric criteria start failing.

## Useful files

| Path | Why |
|---|---|
| `M/instructions broken/INDEX.md` | Distilled authoring rules |
| `M/instructions broken/Project Imperium ... part-7.pdf` | Prompt/golden/rubric rules (source of truth) |
| `M/calibration-methodology.md` | Reusable subagent-testing playbook |
| `M/task-6600/README.md` | Working history overview |
| `M/task-6600/prefix-tuning-shipped/run-{1,2,3}*-answer.md` | Calibration evidence |
| `M/pytorch-torchtune-architecture/rubric.json` | The 36-criterion rubric |
| `M/pytorch-torchtune-architecture/golden_answer.md` | AI-drafted golden — use as research, NOT as template |
| `M/pytorch-torchtune-architecture/prompt_statement.md` | The locked v1 prompt |

## Suggested first actions

1. Read `M/task-6600/README.md` for the journey
2. Read `M/pytorch-torchtune-architecture/prompt_statement.md` to refresh on the prompt
3. Skim the 17 critical rubric criteria so you know what the golden must hit
4. Ask me (Lau) what I want to tackle first — likely either (a) finalizing the golden with the 7 file:line refs + 2 HF fields, or (b) re-authoring the rubric descriptions/rationales
