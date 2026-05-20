# Task 6600 — working history

Working files for Task 6600 (`pytorch-torchtune-architecture`). Final deliverables live in `m/pytorch-torchtune-architecture/`, not here.

- **Intent:** Architecture & System Design
- **Difficulty:** Hard (target Step 5 pass rate 0-35%)
- **Language:** Python
- **BASE commit:** `213f38605ff0b7b1e20f85a9e032710be04c82c9`
- **Closest reference example:** Task 5137 (guide p220-282, distilled in `m/instructions broken/INDEX.md`).

## Layout

| Path | Purpose |
|---|---|
| `candidate_prompts.md` | The original 8 candidate Hard Architecture prompts drafted against the documented design tensions. The pool we picked from. |
| `fp8-qat-abandoned/` | First catalyst attempt: "Add FP8 QAT to torchtune." Abandoned because the catalyst was too greppable (`fp8` and `qat` keyword searches reliably landed agents on the right files) -- 4 calibration runs scored ~50-65%, above the Hard band. See `fp8-qat-abandoned/session-log.md` for the full journey. |
| `prefix-tuning-shipped/` | Second catalyst: "Add Prefix Tuning as a third PEFT method." Zero greppable scaffolding for "prefix" in the repo. 3 calibration runs scored 60-68% (Runs 1+2 locked v1) and 78-83% (Run 3, user-refined variant). Locked v1 is what landed in the submission. |

## What got shipped

`m/pytorch-torchtune-architecture/`:
- `prompt_statement.md` -- locked v1 Prefix Tuning prompt (~80 words)
- `golden_answer.md` -- ~3500 word golden, 7 sections (TL;DR / PEFT contract / 5 design forks / sequenced PR plan / honest read / limitations / files touched). **AI-drafted -- needs re-authoring before submission.**
- `rubric.json` -- 36 criteria (17 critical / 11 bonus incl. 2 style / 8 penalty), all Part 7 limits pass. **AI-drafted -- same concern.**
- `task_metadata.json` -- `{"category": "Architecture & System Design"}`
- `Dockerfile` -- clones torchtune at BASE SHA

## Catalyst comparison

| | FP8 QAT (abandoned) | Prefix Tuning (shipped) |
|---|---|---|
| Greppable | yes (`fp8`, `qat` land directly) | no (no "prefix" scaffolding in repo) |
| Calibration runs | 4 | 3 |
| Strong-model score range | 50-65% | 60-68% (v1) / 78-83% (variant) |
| Reproducible misses | -- | RoPE, PackedDataset packing, loss masking |
| Verdict | abandoned 2026-05-19 | locked 2026-05-20 |

## Open items before submission

1. **Re-author the golden in human voice.** Use the AI-drafted golden as private research; type the prose yourself. Aim ~1000-1500 words, plainer than the current ~3500. Drop the TL;DR and subsection labels. Keep file:line citations.
2. **Re-author rubric descriptions/rationales** for the same reason (Part 8 prohibits LLM-authored rubrics).
3. **Re-read prompt** in your own voice once.
4. **Airtable QC tools** (Prompt QC, Rubric QC, Golden QC, Validate Golden).
5. **Step 5** -- must land in Hard band (0-35% pass rate +/- 3%).

## Methodology notes

`m/calibration-methodology.md` -- the subagent-testing playbook that evolved here. Subagents are launched with explicit isolation (no access to `CLAUDE.md` or anything under `m/`), so calibration scores reflect what a strong Claude-class model recovers from upstream torchtune source alone.

## Calibration findings (cross-cutting)

- **Strong-model hand-grade 60-68% ≈ Step 5 pass rate in Hard band.** There's a ~30 pp gap because Step 5 averages across model qualities; weaker models score much lower than calibration subagents.
- **The user-refined variant scored HIGHER, not lower** than the locked v1 -- strong models recover dropped prompt scaffolding from code reading. Lesson: dropping load-bearing sentences from a prompt is not a reliable way to make it harder.
- **Three reproducible misses across all 3 runs**: RoPE-on-prefix design fork, PackedDataset position-ID interaction, loss masking for prefix tokens. These are the rubric's critical Hard discriminators (criteria 1005, 1007, 1008).
