# Candidate Hard Architecture prompts — Task 6600

Working draft. Each candidate targets a documented design tension (CLAUDE.md
§ "Design tensions") and is grounded in specific files at BASE
`213f38605ff0b7b1e20f85a9e032710be04c82c9`. Pick one (or merge two), then
deepen into `prompt_statement.md` + `golden_answer.md` + `rubric.json`.

Each prompt must satisfy Hard-Architecture bar:
- **Cross-file reasoning** (recipe ↔ module ↔ training ↔ checkpointer ↔ config).
- **Not answerable from generic LLM knowledge** — answer requires this repo.
- **Proposes how to extend or restructure** (the Architecture intent), not
  "what does this code do" (that's Onboarding).
- Reads like a senior engineer's design-doc question, not a homework prompt.

---

## 1. Recipe-duplication refactor under the no-inheritance policy

**Tension:** ~5,700 LOC across 6 distributed recipes (`full_finetune_distributed.py`
1169 LOC, `lora_finetune_distributed.py` 985 LOC, `qat_lora_finetune_distributed.py`
955 LOC, `full_dpo_distributed.py` 1053 LOC, `lora_dpo_distributed.py` 866 LOC,
`ppo_full_finetune_single_device.py`). `torchtune/recipe_interfaces.py`
explicitly says *"torchtune strictly prohibits implementation inheritance"*
and *"Minimizing code duplication is not the goal."*

**Prompt direction:** We need to add a new universally-required field to
training logs (e.g., effective batch size after gradient accumulation) and a
new uniform behavior for OOM recovery. Touching every recipe is the
documented approach but is now blocking a release. Propose a refactor that
extracts the genuinely cross-cutting concerns (metric emission, gradient
accumulation harness, FSDP wrap helpers, barrier helpers) without violating
the no-inheritance policy. Define the seam, the migration path for one
existing recipe (pick LoRA distributed), and the test strategy. Argue what
*must* stay duplicated and why.

**Why Hard:** answerer must reconcile the policy in `recipe_interfaces.py`
with the lived duplication, design a free-function / composition-based seam,
and show a concrete migration. Generic LLM knowledge would propose a base
class — wrong here.

---

## 2. Adapter ↔ checkpointer plugin seam

**Tension:** Today each new adapter type (LoRA, DoRA, QAT-LoRA) requires
editing `torchtune/training/checkpointing/_checkpointer.py` so the
checkpointer knows which keys are adapter weights vs. base weights. State-dict
filter helpers live in `torchtune/modules/peft/_utils.py`. The dependency
runs the wrong way: checkpointer imports adapter knowledge.

**Prompt direction:** Design a registration-based seam where each adapter
class declares its own state-dict filter, base-weight merge, and HF/Meta
round-trip rules — and the checkpointer becomes adapter-agnostic. Show:
(a) the new contract (interface / protocol / registry); (b) what changes in
`FullModelHFCheckpointer`, `FullModelMetaCheckpointer`,
`FullModelTorchTuneCheckpointer`; (c) how LoRA and DoRA migrate; (d) how a
fictional new adapter (e.g., IA³) would be added end-to-end, including its
YAML config; (e) backwards compatibility for existing checkpoints.

**Why Hard:** answerer must trace the current coupling across
`_checkpointer.py`, `peft/_utils.py`, `peft/lora.py`, `peft/dora.py`, and at
least one recipe's `save_checkpoint`. Requires inverting a real dependency.

---

## 3. Distributed lifecycle vs. ParallelDims

**Tension:** `ParallelDims` (in `torchtune/training/_distributed.py`) covers
the topology (DP × TP × CP shape) but each distributed recipe inlines FSDP
wrap, TP plan application, CP shard, gradient accumulation, and explicit
`dist.barrier()` calls. There is no lifecycle owner.

**Prompt direction:** Propose a `DistributedTrainingContext` (or equivalent
non-inheritance abstraction — composition, context manager, free functions)
that owns the lifecycle: rendezvous → topology setup → model wrap → step →
barrier → teardown. Define which responsibilities cross from recipe into
context, which stay in the recipe (gradient accumulation? loss scaling?
optimizer state?). Show how `full_finetune_distributed.py` looks before and
after. Address: does this abstraction violate the no-inheritance policy?

**Why Hard:** requires reading at least two distributed recipes, plus
`_distributed.py`, plus the parallelism plan files in
`torchtune/models/*/_parallelism.py`. The answer must defend a specific
ownership boundary, not just refactor for refactoring's sake.

---

## 4. Declarative model spec vs. factory functions

**Tension:** Each model variant (e.g., `lora_llama3_8b`, `llama3_70b`) is a
hardcoded factory in `_model_builders.py`. YAML configs reference these by
`_component_:` path. Adding a 7B variant means writing a new factory *and*
a new YAML. Model architecture is half declarative (config) and half code
(builder).

**Prompt direction:** Either (a) propose a declarative architecture-spec
format (YAML/dataclass) that lets a user define new variants without writing
Python, including how it integrates with `config.instantiate()` and how
`_component_builders.py` consumes it; OR (b) argue why the current split is
correct and propose narrower guardrails (lint rules, schema validation,
generator tooling) that reduce the cost of adding a variant. Pick a side
and defend it; address LoRA-on-new-variant (which combines a base factory
with an adapter wrapper) explicitly.

**Why Hard:** must understand `config/_instantiate.py` semantics, the split
between `_component_builders.py` and `_model_builders.py`, and how LoRA
adapter factories layer on top.

---

## 5. Vision-encoder swap without dataset surgery

**Tension:** `SFTDataset` applies `message_transform → model_transform`.
Multimodal preprocessing (image patching, tile layout) lives in
`torchtune/modules/transforms/` and is coupled to a specific vision encoder
(e.g., CLIP patch_size, tile_size). Swapping to a different vision encoder
ripples back into dataset construction and recipe wiring.

**Prompt direction:** Design an interface that lets a recipe author swap
vision encoders without modifying the dataset class. Define the contract
between encoder and image transform (what does the encoder advertise — patch
shape, tile policy, dtype, normalization?), where negotiation happens
(builder time? recipe setup?), and how `SFTDataset.__init__` becomes
encoder-agnostic. Show the change in one concrete multimodal recipe and one
config file. Address sequence-packing interactions (`_packed.py`).

**Why Hard:** crosses `datasets/`, `modules/transforms/`, `models/*/`, and
recipe code. Generic knowledge of multimodal pipelines won't yield the
torchtune-specific contract.

---

## 6. Unified quantization seam

**Tension:** Three integration points: (a) **builder-time** — model
factories wrap modules at construction (e.g., NF4 weights via
`torchtune/modules/low_precision/nf4_linear.py`); (b) **recipe-time** —
`torchtune/training/quantization.py` applies torchao quantization after
load; (c) **QAT** — training-time fake-quant in `qat_*` recipes. Each point
has different config knobs and different checkpointer interactions.

**Prompt direction:** Propose either a single unified quantization seam OR
a principled three-way split with explicit invariants. The unified design
should let a config declare `quantization: {dtype: nf4, qat: true,
post_load: false}` once and have the rest of the system honor it.
Address: which lifecycle stage actually owns the binding, how
checkpointers learn the quantization mode at save/load, and how
non-quantizable layers opt out. Walk through one concrete recipe
(`qat_lora_finetune_distributed.py`) before and after.

**Why Hard:** requires reading three different quantization integration
points + at least one checkpointer + one QAT recipe. Generic answers about
"unified quantization" don't survive contact with the QAT fake-quant
training-loop interaction.

---

## 7. Adding a new fine-tuning paradigm (e.g., ORPO / KTO) end-to-end

**Tension:** Adding a new paradigm exercises every seam: new recipe script,
maybe a new loss in `modules/loss/`, possibly a new dataset format, possibly
new checkpointer behavior. The cost is a stress test for the architecture.

**Prompt direction:** Walk through adding ORPO (or KTO) end-to-end against
this codebase. For each step, identify: (a) which existing abstraction
absorbs the new code cleanly, (b) which forces a copy-paste-modify of an
existing recipe, (c) which leaks (e.g., loss type leaks into recipe arg
parsing, or dataset format leaks into checkpointer). From the trace,
identify the 2–3 highest-leverage seams to harden, ranked by frequency of
leakage observed.

**Why Hard:** answer is a code-grounded postmortem of an extension exercise.
Must cite specific files for each leak. Generic ORPO knowledge ≠ this answer.

---

## 8. Multi-worker orchestration vs. OrchestrationRecipeInterface

**Tension:** `OrchestrationRecipeInterface` exists in `recipe_interfaces.py`
with a literal TODO comment: *"we would use this function to stop
hardcoding resources."* It anticipates multi-worker orchestration
(inference + scoring + training workers, e.g., RLHF). But the protocol is
minimal (setup / run / cleanup), and resource allocation is unspecified.

**Prompt direction:** Design what the orchestration story should look like
when fully realized for a PPO-style RLHF setup with separate inference,
scoring, and training workers. Address: (a) what does the protocol need to
add beyond setup/run/cleanup? (b) how does worker resource allocation
(GPUs, ranks, comms groups) interact with `ParallelDims`, which today
assumes one training job owns all ranks? (c) where do worker lifecycle and
training lifecycle compose vs. conflict? (d) what's the migration path
from the current `ppo_full_finetune_single_device.py` to a multi-worker
distributed variant? Be explicit about which design choices are forced by
the no-inheritance policy.

**Why Hard:** combines orchestration, distributed topology, recipe
boundaries, and the TODO already in the code. Requires reading the existing
PPO recipe, the orchestration protocol, and `_distributed.py`.

---

## Selection notes (for the author picking one)

- **Highest "this can only be answered from the repo" score:** #2 (adapter
  ↔ checkpointer), #6 (quantization seam), #8 (orchestration).
- **Highest "long-context" demand:** #1 (must read multiple recipes), #7
  (touches every seam).
- **Cleanest single-tension focus:** #3 (distributed lifecycle), #5 (vision
  swap).
- **Most likely to overlap generic LLM knowledge** (be careful):  #4
  (declarative config) — generic "make it declarative" answers can shallowly
  succeed unless we force the answer to reckon with `config.instantiate()`
  and the LoRA factory layering.

Reference example for calibration: Task 5137 (guide p220–282).
