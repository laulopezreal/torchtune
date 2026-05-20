# Task 6600 — Session Log

**Last updated:** 2026-05-20
**Status:** Prompt locked. Rubric design pending. No QC or Step 5 run yet.
**Task spec:** Architecture & System Design / Hard / Python / BASE `213f38605ff0b7b1e20f85a9e032710be04c82c9`.
**Closest reference example:** Task 5137 (guide p220–282, distilled in `m/instructions broken/INDEX.md`).

---

## Quick start for the next session

1. **Read first:** `m/instructions broken/INDEX.md` (rubric mechanics, prompt rules, Hard authoring loop).
2. **Then the methodology:** `m/calibration-methodology.md` — the reusable subagent-testing playbook this session evolved. Read before launching any further calibration runs.
3. **Then this file** for context on what we tried, what we picked, and why.
4. **Then:** `m/task-6600/prompt-current.md` (the locked prompt) and the three `run-N-answer.md` files (calibration evidence).
5. **Next concrete step:** draft `rubric.json` (target ~30 criteria) calibrated against the three runs. See the "Rubric design guidance" section below.

---

## What we ended up with

### The catalyst

**Add FP8 QAT to torchtune.** Quantization fragmentation (design tension #6 in `CLAUDE.md`) — there are 4–5 distinct integration points for quantization scattered across the codebase, each at a different lifecycle phase, with no unified seam. FP8 already exists in the repo at *training-start* lifecycle (`convert_to_float8_training` in `recipes/full_finetune_distributed.py`); QAT exists at the *module-swap* lifecycle for Int4/Int8 (`qat_*` recipes); the gap "FP8 QAT" forces the answerer to confront the whole fragmentation.

### The locked prompt

See `m/task-6600/prompt-current.md`. ~125 words. Asks the answerer to walk through implementing FP8 QAT, judge whether the codebase shape makes it clean or painful, and propose what to fix first — under the no-inheritance constraint and assuming an active-fork future.

### Calibration data

| Run | Prompt version | Subagent pass rate (estimated against the rubric we'll build) | Verdict |
|---|---|---|---|
| 1 | Over-coached (named files, pre-articulated phases, "lifecycle hooks" hinted) | ~80% | Easy–Medium. Too easy. |
| 2 | Tightened (removed all coaching) | ~55% | Medium. Better but still above Hard. |
| 3 | Tightened + two micro-tweaks (no-maintenance-escape + sixth-scheme-with-custom-forward) | ~75% | Tightening backfired — strong models used the extra constraints to demonstrate strength. |

**Diagnosis:** the FP8 QAT catalyst is too greppable. `fp8` and `qat` keyword searches reliably land competent models on the right files. Tightening the *prompt* doesn't reduce discovery — it only changes the depth of engagement with what's found. **Calibration to Hard band has to come from the rubric, not from more prompt tightening.**

### Decision (user-confirmed)

Lock the Run 3 prompt. Lean on the rubric to do difficulty calibration. The three subagent runs give us enough golden-answer signal + enough miss patterns to write a sharp rubric.

---

## How we got here — the journey we rejected

Skim this if you disagree with where we landed; the dead ends were instructive.

### Dead end 1: Original ORPO + recipe-duplication refactor

**Framing:** "Add ORPO support. Identify duplicated chunks across recipes. Propose a refactor honoring no-inheritance."

**Why it failed:** User pushed back: *"If duplication is encouraged in the recipe interface, what's the problem you're solving? This is curiosity, not engineering."* They were right.

`torchtune/recipe_interfaces.py:13-19` literally says: *"torchtune strictly prohibits implementation inheritance in the codebase. Minimizing code duplication is not the goal. Recipe-writers are encouraged to copy-paste-modify."*

The recipe duplication is **explicitly endorsed policy, not friction**. Asking the model to "refactor" something the maintainers chose violates Part 7's *"Make architecture tasks about real constraints"* rule (see `INDEX.md`). A careful answerer would push back on the prompt itself.

### Dead end 2: IA3 + PEFT↔checkpointer coupling (tension #3)

**Framing:** "Add IA3 (Infused Adapter by Inhibiting and Amplifying inner activations) as a new PEFT method. Every previous adapter (LoRA, DoRA, QAT-LoRA) forced edits inside the checkpointer classes. Design a registration mechanism for the next one."

**Why it failed:** User asked *"where exactly is the friction?"* and forced a code check. Result:

- The **base checkpointer logic** (`torchtune/training/checkpointing/_checkpointer.py`) is adapter-**agnostic**. It uses a single `ADAPTER_KEY` constant and treats whatever state-dict is under it as opaque. IA3 would round-trip through `FullModelMetaCheckpointer` and `FullModelTorchTuneCheckpointer` with zero edits.
- The **only** adapter-specific code is ~80 LOC in `torchtune/models/convert_weights.py`:
  - `_TO_PEFT_KEYS = {"lora_a": "lora_A", "lora_b": "lora_B", "magnitude": "lora_magnitude_vector"}` (line 218–222)
  - `_PEFT_CONFIG_EXPECTED_KEYS = ["target_modules", "r", "lora_alpha"]` (line 237)
  - `tune_to_peft_adapter_weights()` has a `q_proj`/`k_proj` + `lora_B` permutation special-case (lines 302–308)
- This friction matters **only if you want HF-PEFT compatibility** (so the trained adapter is loadable via `peft.PeftModel.from_pretrained()`). For pure torchtune-internal training, IA3 just works.

**Verdict:** Real friction, but ~80 LOC, single file, narrow design surface. User said *"too narrow, I want a juicier design surface."*

### What we kept from the dead ends

- **The trap "edit the checkpointers" pattern** is a great penalty-criterion source. A naive answer to ANY adapter/quantization question is to assume the checkpointers care. They mostly don't. This becomes a penalty in the rubric.
- **The no-inheritance policy** is a real, code-cited constraint in every Hard prompt we'd write against this repo. Quote: `torchtune/recipe_interfaces.py:14`.
- **The "what would break if you moved X" framing** is the canonical hard-reasoning move. Apply it to lifecycle phases in the rubric.

---

## The quantization-fragmentation surface (what's actually there)

Verified at BASE `213f38605`. This is what a strong answer has to cover.

### Five integration points

| # | Lifecycle moment | Where in code | Why there (what would break if moved) |
|---|---|---|---|
| 1 | **Module construction (builder-time NF4)** | `torchtune/modules/low_precision/nf4_linear.py` (`FrozenNF4Linear`); `quantize_base=True` flag in 10+ model builders, e.g. `torchtune/models/llama3/_component_builders.py:129-149`; `torchtune/modules/peft/lora.py:82-87` (LoRALinear.__init__ calls `to_nf4`) | NF4 changes the *parameter object* (it's a tensor subclass). Downstream FSDP sharding, `load_state_dict`, and checkpoint gather (`torchtune/training/_distributed.py:399-442, 486-517`) all special-case NF4. Can't load FP32 weights then convert in-place. |
| 2 | **Post-construction, pre-shard (FP8 training swap)** | `torchtune/training/quantization.py:247-273` (`convert_to_float8_training`); invoked at `recipes/full_finetune_distributed.py:617-627` (line 627). Filters out `output` projection. | TP plan resolution (`Float8ColwiseParallel` vs `ColwiseParallel`) needs to know module types. Moving later means TP can't dispatch correctly. Also `_validate_float8_tp_plan` (`quantization.py:277-302`) gates compatibility. |
| 3 | **Post-construction, pre-shard (QAT module swap)** | `torchtune/training/quantization.py:79-153` (`Int8DynActInt4WeightQuantizer`, `Int4WeightOnlyQuantizer`, QAT variants); invoked at `recipes/qat_distributed.py:677`, `recipes/qat_single_device.py:386` via `quantizer.prepare(model)`. | QAT is dtype-preserving forward-time simulation. `FakeQuantizedLinear` state-dict keys are a superset of `nn.Linear`'s, so moving after `load_state_dict` breaks `strict=True` resume. |
| 4 | **Mid-training (QAT delayed fake-quant toggle)** | `torchtune/training/quantization.py:56-71, 189-202` (`_enable_linear_fake_quant`, `_disable_linear_fake_quant`); registry `_quantizer_mode_to_disable_fake_quant` / `_quantizer_mode_to_enable_fake_quant`. Consumed in `recipes/qat_single_device.py:531-550` (`fake_quant_after_n_steps`). **Drift bug:** `recipes/qat_distributed.py:220` reads `fake_quant_after_n_steps` from config but never consumes it. **Latent bug:** `_enable_linear_fake_quant` at line 63 references `FakeQuantizedLinear` without importing it. | Fake-quant calibration / scales benefit from a warmup on un-faked activations. It's a function of `global_step`; has to live in the training loop. |
| 5 | **Post-training (PTQ / convert to deployable)** | `recipes/quantize.py:69-95`; calls `quantizer.convert(model)` if QAT-prepared, else `quantizer.quantize(model)`. | Quantizing weights is destructive and depends on the *trained* weight distribution. Can't happen at construction (don't know weights yet) or at training-time (would conflate training-precision compute with deployment-precision compute). |

### Auxiliary surfaces a complete answer should also touch

- **TP plan precision dispatch** — `torchtune/models/llama3/_parallelism.py:76-126` has `_get_fp8_llama_tp_training_plan` vs base plan; Llama4 (`models/llama4/_parallelism.py:183`) explicitly raises on FP8.
- **Per-step FSDP scale precompute** — `recipes/full_finetune_distributed.py:1057-1067` calls `precompute_float8_dynamic_scale_for_fsdp` after each optimizer step (tensorwise scaling only). None of the QAT recipes have this hook.
- **LoRA composition** — `QATLoRALinear` in `torchtune/modules/peft/lora.py:150-306` hardcodes int-style fake-quant in its forward. `swap_lora_linear_with_qat` in `torchtune/training/quantization.py:205-244` is the swap helper. `QATLoRALinear.from_lora_linear` (line 280-283) rejects `quantize_base=True` (QLoRA + QAT not stackable). DoRA + QAT not stackable either (`qat_lora_finetune_distributed.py:456`).
- **Adapter state-dict filter** — `torchtune/modules/peft/_utils.py:132`: `adapter_key_filter = lambda x: "lora" in x or "magnitude" in x`. Substring match. Called from `_checkpoint_client.py:170,365,537` and `_distributed.py:528`. Any new adapter with new state needs to update this filter.
- **PEFT injection via model builders** — 35 sites of `adapter_cls = DoRALinear if use_dora else LoRALinear` across `_component_builders.py` files. If/else doesn't scale to a third class.
- **Recipe drift between QAT recipes** — `qat_distributed.py:147` errors on compile, `qat_single_device.py:254` allows it. Only single-device consumes `fake_quant_after_n_steps` in the training loop.

---

## Golden answer structural template (synthesized from Run 1 + Run 3)

The strongest answer would have these sections, in order:

1. **TL;DR** — yes there's friction, name it concretely, give a recommendation in 3 sentences.
2. **The training-time fake-quant surface** — locate `quantization.py` registry, name all four modes today, point at `convert_to_float8_training` as the existing FP8 path.
3. **The LoRA composition surface** — `QATLoRALinear` forward hardcoded, `swap_lora_linear_with_qat` is the swap helper, the two PEFT-injection paths (in-model if/else vs post-construction swap).
4. **The checkpointer / adapter-state surface** — `adapter_key_filter` substring match in `_utils.py:132`; explicitly note that the base checkpointers don't know about QAT (this is good); name the FP8 amax-buffer state question.
5. **The recipe surface** — four QAT recipes today, ~2800 LOC of near-identical code, what each one would need.
6. **The minimal implementation, step by step** — torchao plumbing, registry entry, LoRA composition, TP plan, FSDP scale precompute, configs, tests.
7. **Honest read: is this clean? No, here's why** — enumerated load-bearing reasons.
8. **Concrete fix-first proposals** — `QuantizationScheme` dataclass registry, `AdapterModule.adapter_params()`-driven filter, scheme-agnostic `swap_*` rename + protocol, optionally `adapter_factory` callable in builders. All honor no-inheritance (composition + protocols + dataclasses, not base classes).
9. **What to call out in the PR description** — TP+FP8-QAT not yet supported, Llama4 unsupported, compile blocked, DoRA not stackable, QLoRA base + QAT not stackable.

A strong answer cites ~15–25 specific file:line references. Run 1 cited ~25; Run 3 cited ~30.

---

## Rubric design guidance for next session

### Target counts (from `INDEX.md`)

- Total: ~30 criteria (40 max for Hard).
- Critical: 7–23 (must have BOTH reasoning and completeness as critical).
- Bonus: 4–17.
- Style: 2–6 (never critical, never penalty).
- Penalty: 5–10 (never direct reversals of positives).

### Specific criteria the three runs tell us to include

**Critical reasoning (target 7–10):**

1. Identifies that quantization is integrated at multiple distinct lifecycle phases, not just one place. *(Both Run 2 and Run 3 missed the systematic framing.)*
2. Identifies builder-time NF4 integration as one of the lifecycle phases. *(Runs 2 and 3 both missed this entirely — this is the strongest discriminator.)*
3. Identifies that FP8 training already exists in the repo (`convert_to_float8_training`) and articulates that FP8 QAT is **not** the same thing.
4. Identifies the QAT module-swap (`quantizer.prepare`) as a separate phase from the FP8 training swap.
5. Articulates WHY each integration point is at its specific lifecycle moment — at least 2 of the 4 phases with a "what would break if moved" justification.
6. Identifies the `QATLoRALinear` forward as hardcoded for int-style fake-quant — implication: FP8 QAT + LoRA may need a different forward graph, or `FakeQuantizeConfig` happens to compose.
7. Proposes a unified registration mechanism (dataclass / protocol / scheme descriptor) rather than piecemeal helpers.
8. Proposal explicitly honors the no-inheritance policy — composition, protocols, dataclasses, or free functions.
9. Identifies at least one cross-cutting limitation: DoRA+QAT blocked, QLoRA-base+QAT blocked, compile blocked, Llama4+FP8 blocked.

**Critical completeness (target 5–8):**

10. Names `torchtune/training/quantization.py` as the locus of the quantizer registry.
11. Names `convert_to_float8_training` and `enable_fp8_training`.
12. Names `QATLoRALinear` and `swap_lora_linear_with_qat`.
13. Names at least one TP-plan file (`_parallelism.py`) and notes that FP8 has its own TP plan branch.
14. Names `recipes/quantize.py` as the post-training convert entry point.
15. Names that two new recipes (single-device, distributed) plus LoRA variant are needed *or* explicitly proposes to fold into existing recipes.

**Bonus (target 4–8):**

16. Identifies `adapter_key_filter` substring match in `peft/_utils.py:132` as a state-dict coupling. *(Only Run 3 found this.)*
17. Identifies the `adapter_cls = DoRALinear if use_dora else LoRALinear` 35-site pattern in model builders. *(Only Run 3.)*
18. Identifies recipe drift between QAT recipes (e.g. compile gating, `fake_quant_after_n_steps` honored only in single-device). *(Only Run 3.)*
19. Identifies the latent `FakeQuantizedLinear` import bug in `_enable_linear_fake_quant`. *(Run 1 and Run 3 found this; Run 2 didn't.)*
20. Identifies `precompute_float8_dynamic_scale_for_fsdp` as an FP8-specific per-step hook missing from QAT recipes.
21. Identifies the conceptual question of "FP8 fake-quant in fwd + FP8 all-gather of FSDP" being two different transforms that can't naively stack. *(Run 2 noted this.)*
22. Compares the proposed design to an existing in-repo composition precedent (e.g., `ParallelDims`). *(Run 1 did this.)*

**Style (target 2–3):**

23. Sections the answer (TL;DR, surfaces, fix-first proposals, etc.).
24. Cites specific `file:line` references throughout (not just file names).

**Penalty (target 5–8):**

25. Proposes a base class for recipes, schemes, or adapters (violates the explicit no-inheritance policy at `recipe_interfaces.py:14`).
26. Claims that minimizing code duplication is a torchtune design goal (it is explicitly NOT — see `recipe_interfaces.py:18`).
27. Claims FP8 QAT requires edits inside the checkpointer classes (`FullModelHFCheckpointer`, etc.). It doesn't — the base checkpointers are adapter/quantization agnostic. This is the IA3-dead-end trap baked into the rubric.
28. Claims that `convert_to_float8_training` is the same thing as FP8 QAT (it's not — it's a real-FP8 cast, not fake-quant simulation).
29. Proposes a naive "unify all four integration points into one place" without recognizing that they exist at different lifecycle moments for genuine reasons.
30. Recommends editing all 10+ model builder files to add an `enable_fp8_qat=True` flag (the if/else doesn't scale; should use post-construction swap instead).

### Calibration check (do this before submitting)

Score Run 1 and Run 3 by hand against the drafted rubric.

- If Run 1 ≥ 75% and Run 3 ≥ 65%: rubric too lenient, tighten criteria.
- If Run 1 ≤ 50% and Run 3 ≤ 35%: rubric too strict, relax over-specific criteria.
- Target: Run 1 ~60–70%, Run 3 ~45–55%. Step 5 average across 3 runs should then land 35–55%, which is upper Hard / lower Medium. Acceptable.

Critically: the golden answer must score 100% on critical and trigger zero penalties. Use the structural template above as the golden's outline.

---

## Anti-patterns we learned to avoid

Distilled from the three calibration runs.

1. **Don't pre-name files in the prompt.** Run 1 named `convert_to_float8_training` and the model immediately had two strong anchors. Hard tasks need discovery.
2. **Don't pre-articulate the answer path.** Run 1 said "phases happen at different moments (module construction, after loading…)" — that's the answer wrapped as a question.
3. **Don't add "Hard signals" to constrain the design.** Tweaks like "must compose with LoRA + future sixth scheme" prompted *deeper engagement*, which strong models used to demonstrate strength. Anti-calibration.
4. **Don't pick catalysts that are highly greppable.** `fp8` and `qat` keyword searches land any competent model on the right files in seconds. The current FP8 QAT catalyst is borderline-too-discoverable; we're relying on the rubric to compensate.
5. **Don't ask the model to refactor something the maintainers endorsed.** The original ORPO + recipe-duplication framing failed because `recipe_interfaces.py` explicitly endorses duplication. Always check that the friction is genuinely friction, not just an aesthetic preference.
6. **Don't trust untested claims about the codebase.** I claimed "every new adapter forces checkpointer edits" — code check showed the friction was actually ~80 LOC in `convert_weights.py`, not the checkpointer classes. Verify before committing the prompt to a tension.

---

## Concrete next steps for the next session

In order:

1. Read `m/instructions broken/INDEX.md` (full rubric/prompt rules).
2. Read this file end to end.
3. Open `m/task-6600/prompt-current.md` and the three `run-N-answer.md` files (Run 1 and Run 3 are the most important).
4. Draft `rubric.json` against the structure above. Aim for 28–32 criteria.
5. Score Run 1 and Run 3 against the draft rubric by hand. Adjust until calibration check above is satisfied.
6. Write the final `golden_answer.md` using the structural template. It must earn 100% critical + at least one bonus + zero penalties.
7. Write `prompt_statement.md` from `prompt-current.md`. Final ASCII check.
8. Verify `task_metadata.json` says `{"category": "Architecture & System Design"}`.
9. Run Airtable QC tools (Prompt QC, Rubric QC, Golden QC, Validate Golden). Apply human judgment.
10. Run Step 5 model eval. Target 35–55% (slightly above Hard band is acceptable given catalyst limitations; below 30% would mean tightening too far).
11. If Step 5 lands above 55%, **don't** tighten the prompt — sharpen the rubric instead (the anti-patterns above explain why).
12. Submit.

---

## Files in `m/task-6600/`

- `session-log.md` — this file
- `prompt-current.md` — the locked prompt
- `run-1-answer.md` — over-coached prompt + full subagent answer (golden-answer template)
- `run-2-answer.md` — tightened prompt + full subagent answer (Medium-band reference)
- `run-3-answer.md` — two-tweak prompt + full subagent answer (closest to a strong real submission)
