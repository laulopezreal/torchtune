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

**Start here:** `M/instructions broken/INDEX.md` is the distilled, reusable summary of every rule that applies to authoring (prompt rules, golden rules, rubric rules + limits + JSON shape, difficulty bands + calibration, Task 5137 structural template, reviewer checklist, Hard Architecture authoring loop). Load it before drafting; only crack a part PDF when the index points there.

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
  - **Session log:** `M/task-6600/session-log.md` — read this before continuing. Captures the catalyst we picked (FP8 QAT, tension #6 quantization fragmentation), the journey through dead ends (ORPO recipe-duplication, IA3 PEFT↔checkpointer), three calibration runs, and the locked prompt. Run answers preserved as `run-{1,2,3}-answer.md`. Current prompt in `prompt-current.md`. Next concrete step is rubric design.

## Torchtune architecture (at BASE `213f38605`)

Working map of the codebase to ground Hard Architecture prompts. Keep this anchored to the BASE SHA — file paths and design tensions below are verified there.

### Layout

- `torchtune/` — library: `models/`, `modules/`, `datasets/`, `training/`, `generation/`, `rlhf/`, `config/`, `data/`, `utils/`, `_cli/`.
- `recipes/` — standalone training scripts that implement `FTRecipeInterface`. **No shared base class**; the idiom is copy-paste-modify per recipe (full vs. lora vs. dpo vs. qat × single_device vs. distributed).
- `recipes/configs/` — per-family YAML. `_component_:` paths drive `config.instantiate()` (see `torchtune/config/_instantiate.py`).

### Core abstractions

- **Recipes** — protocol-based interfaces in `torchtune/recipe_interfaces.py`: `FTRecipeInterface`, `EvalRecipeInterface`, `OrchestrationRecipeInterface`. Each recipe script (~700 LOC) implements the protocol independently.
- **Models** — split across:
  - `_component_builders.py` (low-level: builds `TransformerDecoder` from primitives)
  - `_model_builders.py` (factory functions like `lora_llama3_8b()`, referenced from YAML)
  - `_tokenizer.py` (per-family tokenizer)
  - `_parallelism.py` (TP plans, where present)
- **Modules** — `torchtune/modules/`:
  - `transformer.py`, `attention.py`, `position_embeddings.py`, `rms_norm.py`, `kv_cache.py`
  - `peft/` — LoRA / DoRA / QAT-LoRA adapters and state-dict utilities
  - `low_precision/` — NF4 / `to_nf4` integration (currently re-exported via `torchao.dtypes.nf4tensor`)
  - `loss/`, `tokenizers/`, `transforms/`, `model_fusion/`
- **Datasets** — `SFTDataset` wraps a HF dataset and applies `message_transform → model_transform`; `_packed.py` does sequence packing. Multimodal preprocessing lives in `transforms/` and is invoked from the dataset side.
- **Training** — `torchtune/training/`:
  - `_distributed.py` — FSDP + TP + CP wiring via `ParallelDims`
  - `checkpointing/` — `FullModelHFCheckpointer`, `FullModelMetaCheckpointer`, `FullModelTorchTuneCheckpointer`; each is adapter-aware (knows about LoRA/DoRA state-dict keys)
  - `quantization.py` — torchao integration
  - `metric_logging.py`, `memory.py`, `seed.py`

### Design tensions (gold for Hard Architecture prompts)

1. **Recipe duplication** — 850–1170 LOC of near-identical boilerplate per distributed recipe (~5,700 LOC across the six distributed scripts at BASE). Intentional: `recipe_interfaces.py` says *"torchtune strictly prohibits implementation inheritance"* and *"Minimizing code duplication is not the goal."* Cost: cross-cutting changes (e.g., new logging field) must be replicated N times.
2. **Config-vs-code seam** — model architecture is half declared in YAML (`_component_`, hyperparams) and half hardcoded in `_model_builders.py` factories. Adding a variant means editing both.
3. **PEFT ↔ checkpointer coupling** — every new adapter type (LoRA, DoRA, QAT-LoRA) requires changes inside the checkpointer classes to round-trip adapter state. Not pluggable.
4. **Distributed coordination scattered** — FSDP/TP/CP setup, gradient accumulation, and barrier logic are inlined in each distributed recipe rather than centralized; `ParallelDims` covers the topology but not the lifecycle.
5. **Dataset transform pipeline** — vision/multimodal preprocessing couples the dataset to a specific model's image transform (e.g., CLIP patch size, tile layout). Swapping vision encoders ripples back into dataset construction.
6. **Quantization integration is fragmented** — three integration points: builder-time (model factory wraps modules), recipe-time (post-load conversion), and QAT (training-time fake-quant). No single seam.

### Prompt-authoring notes

- The architecture task (6600) targets one of the above tensions. Hard prompts must require *cross-file* reasoning (recipe ↔ module ↔ training ↔ checkpointer) and have answers that aren't obtainable from generic LLM knowledge.
- "8 candidate Hard Architecture prompts" are being drafted against these tensions — keep candidates here as they stabilize.

## Pointers for future sessions

- Container is ephemeral; only what's committed persists. The `M/` folder is committed and readable.
- To read PDFs: `poppler-utils` is not preinstalled — install with `apt-get install -y poppler-utils`, then use `pdftotext -layout` or the `Read` tool's `pages` parameter.
- Default branch for task work: `claude/read-m-project-docs-EEew4` (or whichever feature branch the session specifies).
