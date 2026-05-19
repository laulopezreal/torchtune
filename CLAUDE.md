# CLAUDE.md

Notes for Claude Code sessions working in this repository.

## Repository context

This is a fork of `pytorch/torchtune` being used for **Project Imperium — Agentic Code QA** task authoring. The upstream project is winding down (see README), but the repo serves as the target codebase for SWE Q&A tasks: experts write prompts, golden answers, and rubrics grounded in this code at specific commits.

## Project Imperium — Instruction Guide Index

Source: `M/Project Imperium _ Agentic Code QA Instruction Guide.pdf` (287 pages, last updated 2026-05-16).

| # | Section | Page |
|---|---|---|
| 1 | Overview | 1 |
| 2 | Project Expectations | 3 |
| 3 | Payments | 5 |
| 4 | Change Log | 8 |
| 5 | LLM Usage Policy | 10 |
| 6 | Task Creation & Scope | 13 |
| 7 | End-to-End Task Workflow | 15 |
| 8 | Task Claiming & Setup | 19 |
| 9 | Repo Exploration & Task Scope | 24 |
| 10 | Prompt & Golden Answer Writing | 29 |
| 11 | Rubric Writing & Difficulty Calibration | 35 |
| 12 | Rubric Design 101 | 48 |
| 13 | Task Examples | 53 |
| 13a | — Example Task 5368: Root Cause Analysis (Medium) | 55 |
| 13b | — Example Task 5680: PR Triage & Impact Assessment (Medium) | 96 |
| 13c | — Example Task 5354: Code Onboarding & Comprehension (Hard) | 153 |
| 13d | — Example Task 5137: Architecture & System Design (Hard) | 220 |
| 14 | Reviewer Guidelines | 283 |

The guide also ships split into per-section parts under `M/instructions broken/` (parts 1–13, each tagged with its section: E2e Workflow, Repo Exploration, Task Claiming, Changelog, LLM Usage, etc.). Prefer the part files for targeted reads; the monolithic PDF is for full-doc searches.

## Project gist

Authors create **codebase-grounded** SWE Q&A tasks for a fixed repo at a fixed commit. Each task is one of four **intents**:

- **Root Cause Analysis** — diagnose a bug/behavior from the code.
- **PR Triage & Impact Assessment** — review a diff for risks and blast radius.
- **Code Onboarding & Comprehension** — explain how a subsystem works.
- **Architecture & System Design** — propose how to extend or restructure.

Each at **Easy / Medium / Hard** difficulty.

### Deliverables per task

- `prompt_statement.md` — developer-style question.
- `golden_answer.md` — human-authored, repo-grounded answer.
- `rubric.json` — tagged + weighted criteria.
- `task_metadata.json` — task config.
- `Dockerfile` — clones the target repo at the BASE SHA.

## Workflow shorthand

1. Claim task → confirm repo + BASE SHA locally (Setup, p19–23).
2. Explore repo for a question that **requires** repo evidence, not generic knowledge (p24–28).
3. Write prompt + golden answer with concrete file paths, functions, behaviors (p29–34).
4. Write rubric: tagged (reasoning/completeness/…) and weighted (critical/bonus), within criteria limits (p35–47, 48–52).
5. Calibrate to assigned difficulty (Hard requires cross-file, long-context reasoning).

## Active task (when applicable)

- **Task 6600** — `pytorch-torchtune-architecture`
  - Intent: Architecture & System Design
  - Difficulty: Hard
  - Language: Python
  - BASE commit: `213f38605ff0b7b1e20f85a9e032710be04c82c9`
  - Closest reference example: Task 5137 (guide p220–282).

## Pointers for future sessions

- Container is ephemeral; only what's committed persists. The `M/` folder is committed and readable.
- To read PDFs: `poppler-utils` is not preinstalled — install with `apt-get install -y poppler-utils`, then use `pdftotext -layout` or the `Read` tool's `pages` parameter.
- Default branch for task work: `claude/read-m-project-docs-EEew4` (or whichever feature branch the session specifies).
