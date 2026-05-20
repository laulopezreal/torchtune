# M/ — Project Imperium task authoring workspace

Working directory for **Project Imperium — Agentic Code QA** task authoring against the torchtune fork at BASE `213f38605ff0b7b1e20f85a9e032710be04c82c9`.

## Layout

| Path | Purpose |
|---|---|
| `pytorch-torchtune-architecture/` | **Final submission folder.** The five deliverables (`prompt_statement.md`, `golden_answer.md`, `rubric.json`, `task_metadata.json`, `Dockerfile`) for Task 6600. This is the folder that gets uploaded to Airtable. |
| `task-6600/` | Working history for Task 6600 — calibration runs, abandoned catalyst, candidate prompts, hand grades. Not part of the submission. See `task-6600/README.md`. |
| `instructions broken/` | Project Imperium instruction guide split per section, plus `INDEX.md` (distilled rules — read this first). |
| `Project Imperium _ Agentic Code QA Instruction Guide.pdf` | Full 287-page instruction guide. |
| `calibration-methodology.md` | Reusable playbook for testing prompts against subagents before locking. Evolved during Task 6600. |
| `pytorch-torchtune-architecture.zip` | Original task scaffold archive (untouched reference copy). |

## Start here

1. `instructions broken/INDEX.md` — distilled authoring rules.
2. `task-6600/README.md` — what was tried, what shipped, what's still owed.
3. `pytorch-torchtune-architecture/` — the actual deliverables.

## Status (2026-05-20)

Task 6600 deliverables are committed but **the golden and rubric were AI-drafted**. Per the May 19 onboarding meeting, golden answers must be human-authored; rubrics likewise per Part 8. Use the current artifacts as research material; re-author the prose before Airtable QC + Step 5.
