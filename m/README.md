# m/ — Project Imperium task authoring workspace

Working directory for **Project Imperium / Agentic Code QA** task authoring. Target repo: this fork of `pytorch/torchtune` pinned at BASE commit `213f38605ff0b7b1e20f85a9e032710be04c82c9`.

## What's where

| Path | What it is |
|---|---|
| **`pytorch-torchtune-architecture/`** | **Submission folder.** The five files that get uploaded to Airtable: `prompt_statement.md`, `golden_answer.md`, `rubric.json`, `task_metadata.json`, `Dockerfile`. |
| `task-6600/` | Working history: calibration runs, abandoned-catalyst archive, candidate prompts, hand grades, next-session handoff. Not submitted. |
| `instructions broken/` | The Project Imperium guide PDFs split by section, plus `INDEX.md` — the distilled rules summary. |
| `instructions broken/INDEX.md` | Distilled authoring rules. Read first. |
| `Project Imperium _ Agentic Code QA Instruction Guide.pdf` | Full 287-page guide. Use the per-section parts under `instructions broken/` for targeted reads. |
| `calibration-methodology.md` | Reusable subagent-testing playbook (evolved during Task 6600). |
| `pytorch-torchtune-architecture.zip` | Original task scaffold archive. Reference copy, untouched. |

## If you're new to this

1. Read `instructions broken/INDEX.md` — covers prompt rules, golden rules, rubric format + limits, difficulty bands, Task 5137 template.
2. Read `task-6600/README.md` — what was tried, what shipped, what's left to do.
3. Then look in `pytorch-torchtune-architecture/`.

## Status (2026-05-20)

Task 6600 deliverables are structurally complete and pushed to `main`:
- Prompt locked (v1 Prefix Tuning, ASCII clean)
- Rubric: 36 criteria, all Part 7 limits pass, JSON schema validated
- Golden: AI-drafted; the user has iterated a personal version through 5 rounds (latest is ~1800 words in user's voice)

**Open work:** finish authoring the golden + rubric prose in human voice, upload to Airtable, run QC tools, run Step 5. See `task-6600/README.md` for the punch list and `task-6600/next-session-prompt.md` for a fresh-session handoff brief.

**Why human authoring matters:** per the May 19 onboarding meeting, AI-drafted goldens and rubrics are an offboarding risk (Part 8). The structure can be AI-assisted research; the prose has to be the user's.
