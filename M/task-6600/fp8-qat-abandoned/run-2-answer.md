# Run 2 — Tightened prompt + subagent answer

**Date:** 2026-05-20
**Subagent:** general-purpose (Claude Opus 4.7-class)
**Repo:** torchtune at BASE `213f38605`
**Estimated rubric pass rate:** ~55% (Medium band)
**Significance:** Shows what happens when breadcrumbs are removed. **Missed NF4 builder-time integration entirely** — that miss is the strongest rubric discriminator we have. Played the README "no longer actively maintained" card to scope down (which Run 3 was tweaked to block).

## The prompt sent

> I want to add FP8 QAT to torchtune — fake-quantize weights to FP8 during training, deploy as actual FP8 at inference. Single-device and distributed, ideally on top of LoRA. This is my first contribution.
>
> Walk me through what implementing this actually requires. I'm trying to follow the contribution rules — recipes stay readable as self-contained scripts, no implementation inheritance — and I want an honest read on whether the codebase shape makes this addition clean or whether I'd be adding pain that future contributors will inherit. If it's the latter, propose what to fix first.

## The subagent answer

# Adding FP8 QAT to torchtune: a contributor's reality check

Below is what implementing this actually looks like in the repo as it sits at HEAD, and where it will hurt. I'll split it into (1) what already exists you can build on, (2) the concrete work, recipe by recipe, (3) the structural costs that will fall on you because of how torchtune is organized, and (4) what to fix first if you'd rather invest a day up front than re-pay it forever.

I should flag the elephant in the room before going further: the README says *"Torchtune is no longer actively maintained: torchtune development wound down in 2025"* (`/home/user/torchtune/README.md` line 1). That matters for any "first contribution" calculus; the cost/benefit on cleanup work is different than on a healthy project. I'll still answer the architecture question honestly.

## 1. What you can build on

### FP8 training is already wired up (no-QAT form)

`torchtune/training/quantization.py` already imports torchao's float8 module and exposes:

- `convert_to_float8_training(model, fp8_recipe_name=...)` (lines 247-273): swaps `nn.Linear` with `Float8Linear` and supports `"tensorwise" | "rowwise" | "rowwise_with_gw_hp"`, defaulting to tensorwise with FSDP all-gather enabled. It already filters out the `"output"` projection via `module_filter_fn=lambda mod, fqn: fqn != "output"`.
- `is_fp8_tensorwise_scaling(...)` and `_validate_float8_tp_plan(...)` (lines 277-302): guard the TP combinations torchao supports.
- Model-side TP plans: `torchtune/models/llama3/_parallelism.py` has `_get_fp8_llama_tp_training_plan` using `Float8ColwiseParallel`, `Float8RowwiseParallel`, `PrepareFloat8ModuleInput`. Llama4's plan (`torchtune/models/llama4/_parallelism.py` line 183) explicitly raises on FP8.

The only recipe that wires this in today is `recipes/full_finetune_distributed.py`:
- Reads `enable_fp8_training` and `fp8_recipe_name` from config (lines 211-212)
- Calls `convert_to_float8_training` before TP / FSDP (line 627)
- Calls `precompute_float8_dynamic_scale_for_fsdp(self._model)` after the optimizer step under tensorwise scaling (lines 1057-1067)

Note: not a single shipped YAML in `recipes/configs/` sets `enable_fp8_training: True` (grep returns empty). It's CLI-override-only territory today, even for plain full FT.

### QAT is wired up for int4 / int8da-int4w

`torchtune/training/quantization.py` lines 26-29 import torchao's `Int4WeightOnlyQATQuantizer` and `Int8DynActInt4WeightQATQuantizer`. The dispatch tables (`_quantizer_to_mode`, `_quantizer_mode_to_disable_fake_quant`, `_quantizer_mode_to_enable_fake_quant`) and the helper pair `_get_disable_fake_quant` / `_get_enable_fake_quant` (lines 189-202) are how recipes do delayed fake-quant (see `recipes/qat_single_device.py` lines 531-550).

Each QAT recipe calls `quantizer.prepare(model)` to module-swap `nn.Linear` → torchao's `FakeQuantizedLinear`, which is *dtype-preserving* — that's the reason the existing `FullModelHFCheckpointer`, `FullModelMetaCheckpointer`, `FullModelTorchTuneCheckpointer` in `torchtune/training/checkpointing/_checkpointer.py` need *zero* QAT-specific code. State dicts round-trip as fp32/bf16 weights; the fake-quant op is purely a forward-time numeric simulation, and `recipes/quantize.py` (lines 88-95) does the actual int conversion *post-training* by calling `quantizer.convert(self._model)`.

### QAT+LoRA path

`torchtune/modules/peft/lora.py`:
- `QATLoRALinear` (lines 150-306) wraps base-weight & activation `FakeQuantizer`s around the frozen weight, leaves LoRA branch full-precision: `x -> fake_quantize(W_frozen) @ fake_quantize(x) + BAx` (line 218). Forward is at lines 249-265.
- `from_lora_linear` constructor for in-place swap (lines 267-306). Rejects bias and rejects `quantize_base=True` (LoRA + QLoRA + QAT not stackable).

`torchtune/training/quantization.py` `swap_lora_linear_with_qat(...)` (lines 205-244) walks the module tree and module-swaps `LoRALinear` → `QATLoRALinear`. `recipes/qat_lora_finetune_distributed.py` `_convert_model_to_qat` (lines 450-466) instantiates the quantizer, pulls `get_activation_fake_quantize_config()` / `get_weight_fake_quantize_config()` off it, and calls the swap. It refuses DoRA explicitly (line 456).

## 2. The actual work, recipe by recipe

You said: single-device + distributed, layered on LoRA. That's three recipes (`qat_single_device.py`, `qat_distributed.py`, `qat_lora_finetune_distributed.py`) plus, transitively, `recipes/quantize.py` for the final convert step. Here's what each needs.

### A. torchao plumbing (the small part)

You need an FP8-flavored QAT quantizer that fits the existing `_quantizer_to_mode` shape. In `torchtune/training/quantization.py`:

1. Either expose a torchao FP8 QAT quantizer (if torchao ships one — `Float8DynActFloat8WeightQATQuantizer` or similar; check the torchao version you're targeting) or write a thin wrapper that constructs the right `FakeQuantizeConfig(dtype=torch.float8_e4m3fn, ...)` pair and exposes `.prepare()`, `.convert()`, `.get_activation_fake_quantize_config()`, `.get_weight_fake_quantize_config()`. The last two are what `_convert_model_to_qat` calls (line 464-465 of `qat_lora_finetune_distributed.py`).
2. Register it in `_quantizer_to_mode["fp8-qat"]` / `…enable_fake_quant["fp8-qat"]` / `…disable_fake_quant["fp8-qat"]`, and add to `__all__`. The disable/enable can reuse `_enable_linear_fake_quant` (lines 56-71) as long as torchao's FP8 fake quantizer subclasses `FakeQuantizedLinear`.
3. Update the docstring of `get_quantizer_mode` (lines 167-186) and `recipes/quantize.py`'s `QuantizationRecipe` docstring (lines 27-46) to advertise the mode.

This is contained and copy-paste clean; nothing structural to argue about.

### B. Recipe surface (the painful part)

This is where torchtune's "copy-paste recipes, no inheritance" rule (`torchtune/recipe_interfaces.py` lines 14-23: *"This interface is not meant to be a vehicle for code reuse… Minimizing code duplication is not the goal. Recipe-writers are encouraged to copy-paste-modify"*) collides with what you're trying to do. Specifically, the QAT recipes and the FP8 training path are *different code in different files* today.

Two design choices, with their costs:

**Option B1 — fold FP8-QAT into the existing `qat_*` recipes.** Cleanest from the user's POV (one mental model: "QAT, pick your quantizer"). What you'd touch:
- `recipes/qat_single_device.py`: `_quantizer_mode` already gates the delayed-fake-quant logic (lines 531-550). If your FP8 quantizer registers `"fp8-qat"` in the mode tables, that loop just works.
- `recipes/qat_distributed.py` `_setup_model` (lines 590-748): the call sequence is meta-init → compile → TP → AC → `quantizer.prepare()` → FSDP shard → load state. FP8 in `full_finetune_distributed.py` runs *before TP* (line 627). torchao's FP8 QAT prep, if it's pure module-swap of `nn.Linear` → `FakeQuantizedLinear` like the other QAT modes, can stay where `quantizer.prepare(model)` is today (line 677) — *but* you need to verify the swap composes with the FSDP shard hooks and with `Float8ColwiseParallel`/`Float8RowwiseParallel` TP plans. The `_validate_float8_tp_plan` check (lines 277-295) is there because torchao FP8 TP plans only compose with tensorwise scaling; QAT TP support is an unknown you'll need to verify with a real run.
- `recipes/qat_lora_finetune_distributed.py`: `swap_lora_linear_with_qat` is dtype-agnostic — it just constructs `FakeQuantizer(activation_qat_config)` / `FakeQuantizer(weight_qat_config)`. If the FP8 quantizer's `get_*_fake_quantize_config()` returns a proper torchao `FakeQuantizeConfig`, this Just Works™ for forward semantics. But: this recipe also rejects DoRA (line 456), it rejects bias and `quantize_base=True` (`QATLoRALinear.from_lora_linear` lines 280-283 of `lora.py`), and there's no `compile` path (`recipes/qat_distributed.py` line 147-150 outright errors on compile for QAT). You inherit those restrictions.

**Option B2 — new dedicated recipes** (e.g. `fp8_qat_distributed.py`, `fp8_qat_single_device.py`, `fp8_qat_lora_finetune_distributed.py`). This is what the contribution rules push you toward, and it's what the project does in practice (six distributed QAT-adjacent scripts that share 80%+ of their text). Cost: you'd be adding ~2,800 LOC (1146 + 955 + 719 from the three QAT files, give or take). And every cross-cutting fix in QAT or LoRA from here on must be applied N times.

I would push back gently on B2 if I were reviewing your PR — even within the project's own rules, the existing `qat_*` recipes already parameterize the quantizer via `_component_`. Adding `"fp8-qat"` as another quantizer choice is exactly the seam those recipes were designed for. New recipes here would be cargo-culting the duplication idiom past its purpose.

### C. Things that need to happen unique to FP8 (and where they live)

- **FSDP all-gather scale precompute.** `full_finetune_distributed.py` lines 1057-1067 call `precompute_float8_dynamic_scale_for_fsdp` after the optimizer step under tensorwise scaling. None of the QAT recipes have this hook. You need to add it to `recipes/qat_distributed.py` and `recipes/qat_lora_finetune_distributed.py` train loops, gated on whether the quantizer is FP8 and the scaling is tensorwise. You'll want a public predicate, not just `is_fp8_tensorwise_scaling` — something like `is_fp8_qat_quantizer(quantizer)` in `torchtune/training/quantization.py`.
- **TP plan selection.** `recipes/full_finetune_distributed.py` line 641 passes `enable_fp8_training=self._enable_fp8_training` to the TP plan factory. Llama3's plan picks the `Float8*Parallel` styles when set (`torchtune/models/llama3/_parallelism.py` lines 76-120). The QAT distributed recipe calls `config.instantiate(self.tp_plan, model=model)` without that flag (line 633). If you want TP+FP8+QAT, you'll need to thread an `enable_fp8_training` (or `is_fp8_qat`) parameter through the QAT recipe's TP-plan instantiation, *and* you'll need to think hard about whether all-gather in FP8 is even sensible while you're still in the fake-quant simulation phase. The honest answer is "the existing FP8 training all-gather is a separate optimization that overlaps with what the FakeQuantizer is simulating" — combining them is a research question, not a wiring question. For a first PR I'd punt and raise on `tp_degree > 1 and quantizer_mode == "fp8-qat"`.
- **Llama4.** `torchtune/models/llama4/_parallelism.py` line 183 hard-rejects FP8 today (`raise ValueError("FP8 training is not supported for Llama4")`). Don't try to fix that in this PR.
- **Inference / convert step.** `recipes/quantize.py` is the post-training conversion entry point. Its `quantize()` method (lines 88-95) calls `self._quantizer.convert(self._model)` for any QAT mode. As long as your FP8 quantizer implements `.convert()` (torchao's QAT quantizers do), this path is free — but you'll need configs under `recipes/configs/quantization.yaml` (or similar) and at least one integration test.

### D. Configs

Each model family that gets FP8 QAT support needs a `*_fp8_qat_full.yaml`, `*_fp8_qat_lora.yaml` per device count. Look at `recipes/configs/llama3/8B_qat_full.yaml` and `recipes/configs/llama3/8B_qat_lora.yaml` for the template; only the `quantizer._component_` line changes. The recipe registry at `torchtune/_recipe_registry.py` then needs the new configs added under the `qat_*` recipe entries (lines 624-660 area).

### E. Tests

`tests/recipes/test_qat_distributed.py`, `test_qat_single_device.py`, `test_qat_lora_finetune_distributed.py` exist. New parametrized cases per FP8 quantizer; loss-curve sanity test (matches an existing QAT run within tolerance for the first few steps); convert-step round trip test. No tests today cover FP8 anywhere in `tests/` (search returns empty), so you're setting the precedent.

## 3. Honest read on whether this lands clean

It mostly lands clean as B1 (add a mode to existing recipes), *if* torchao's FP8 QAT quantizer behaves like its int QAT quantizers (dtype-preserving module swap into a class compatible with the `_enable_linear_fake_quant` predicate). The good design choices the codebase already made that help you:

- The quantizer-as-config-component pattern. `qat_*` recipes don't know about int4 vs int8 — they call `config.instantiate(quantizer_cfg).prepare(model)`. FP8 slots in here.
- Checkpointers are blissfully ignorant of QAT (verified: `grep -i "qat\|FakeQuant" torchtune/training/checkpointing/` returns nothing). Fake-quant is a forward-time transformation only; weights stay in their training dtype. This is real evidence that *the PEFT↔checkpointer coupling complaint people sometimes raise doesn't apply to QAT.* It applies to LoRA/DoRA (because adapter weights are real new parameters that need to be saved/loaded separately), and your `QATLoRALinear` will inherit that LoRA scaffolding without you touching the checkpointer.
- `swap_lora_linear_with_qat` is dtype-agnostic.

The places it lands *messily* are FP8-specific, not torchtune-specific:

- **The FSDP fp8 all-gather scale precompute hook.** It's a tensorwise-only thing, it has to be called once per optimizer step, and the QAT recipes don't have it. You'll be adding 5-8 lines in two more files. Manageable.
- **TP plan dispatch.** The asymmetry between `full_finetune_distributed.py` (passes `enable_fp8_training`) and `qat_distributed.py` (doesn't) is real. The cleanest fix is generalizing the TP plan factory to take a `precision_hint` argument or to consult the quantizer mode directly. Otherwise you're propagating an `enable_fp8_training` bool through three more recipes.
- **The semantic question of "FP8 fake-quant in fwd + FP8 all-gather of grads/weights for FSDP perf."** These are two different FP8 transforms. The existing `convert_to_float8_training` swaps `nn.Linear` → real `Float8Linear` (does actual fp8 compute + all-gather). QAT swaps `nn.Linear` → `FakeQuantizedLinear` (does bf16 compute, simulates fp8 numerics). You can't naively stack them. You probably want to *replace* the QAT prepare on a layer with an FP8 prep on the same layer for the post-warmup phase, or just commit to "QAT = simulation only, FSDP all-gather stays in bf16." Pick one in the design doc up front; reviewers will ask.

## 4. What I'd fix first if I were doing this

If the project were healthy, I'd say: don't ship FP8 QAT until two cleanups are in:

1. **Lift the per-recipe FP8 hooks into the library.** Make `torchtune/training/quantization.py` own:
   - A `maybe_precompute_float8_scales(model, quantizer_or_mode)` that no-ops unless the mode demands it. Today the logic is inlined into `full_finetune_distributed.py` lines 1057-1067 and would be re-inlined into three more recipes.
   - A `setup_fp8_training_or_qat(model, mode_or_quantizer, parallel_dims)` that does the prepare + the TP validation + returns a small struct telling the recipe what hooks to call. Today `full_finetune_distributed.py` line 617 has the prep and lines 624-626 has the CP guard, totally separately from where the QAT recipes' `quantizer.prepare()` lives.
   - A canonical `is_fp8_*` predicate set used everywhere instead of the magic string `"qat" in quantizer_mode` pattern.

2. **One TP-plan factory shape.** Today `_get_llama_tp_training_plan(model, enable_fp8_training=...)` takes the flag, but the QAT recipe instantiates the plan via `config.instantiate(self.tp_plan, model=model)` (line 633 of `qat_distributed.py`), which only forwards what's in the YAML. Either standardize on "the recipe always passes a `precision_hint` kwarg" or move precision selection inside the plan based on the model's actual linear types (FP8-aware detection on the modules). The current state means every new precision flavor multiplies recipe ↔ plan plumbing.

That said: given the README's "no longer actively maintained" notice, I would not invest the cleanup work first. Do option B1, take the small duplication hit (the FSDP scale-precompute hook in two extra files, an `is_fp8_qat_quantizer` helper, TP gated off with a clear error message), and submit a tight PR. If maintainers want the refactor, they'll ask. If they don't, you haven't sunk a week on speculative cleanup of a winding-down codebase.

### Concrete PR shape I'd aim for

- `torchtune/training/quantization.py`: add FP8 QAT quantizer class + mode registration + `is_fp8_qat_quantizer(...)` predicate + `maybe_precompute_fp8_scales(model, quantizer)` helper (~80 LOC).
- `recipes/qat_single_device.py`: no change beyond config registration (the existing delayed-fake-quant loop is mode-agnostic).
- `recipes/qat_distributed.py`: ~10 LOC — call the precompute helper in the optimizer-step branch around line 1027; raise on TP+fp8-qat for now near line 631.
- `recipes/qat_lora_finetune_distributed.py`: same ~10 LOC for the precompute helper.
- `recipes/quantize.py`: docstring update; the convert path is already mode-driven.
- Configs: one full + one LoRA YAML for one model family (Llama3 8B is the safe choice — has both FP8 TP plan support and existing QAT configs).
- Tests: extend the three `test_qat_*` files with a `quantizer="fp8-qat"` parametrization; add a smoke test for the convert step.

### Things to call out in the PR description

- TP not yet supported with FP8 QAT (explain the asymmetry).
- Llama4 not supported (preexisting FP8 limitation).
- Compile not supported (preexisting QAT limitation at `qat_distributed.py:147-150`).
- DoRA not stackable (preexisting limitation at `qat_lora_finetune_distributed.py:456`).
- `quantize_base=True` (QLoRA-style NF4 base) not stackable (preexisting at `lora.py:283`).

That gives reviewers a clear scope and lets the next contributor know which boxes are still empty without you having to fill all of them.

### Key files for reference

- `/home/user/torchtune/torchtune/training/quantization.py` — quantizer registry + FP8 training conversion
- `/home/user/torchtune/torchtune/modules/peft/lora.py` — `LoRALinear`, `QATLoRALinear`, `from_lora_linear`
- `/home/user/torchtune/recipes/qat_distributed.py` — distributed QAT (1146 LOC)
- `/home/user/torchtune/recipes/qat_single_device.py` — single-device QAT, includes delayed-fake-quant loop
- `/home/user/torchtune/recipes/qat_lora_finetune_distributed.py` — QAT + LoRA (955 LOC)
- `/home/user/torchtune/recipes/full_finetune_distributed.py` lines 211-212, 617-627, 1057-1067 — the existing FP8 training hooks you're mirroring
- `/home/user/torchtune/recipes/quantize.py` — post-training convert path
- `/home/user/torchtune/torchtune/models/llama3/_parallelism.py` — FP8 TP plan
- `/home/user/torchtune/torchtune/recipe_interfaces.py` — the "no inheritance, copy-paste-modify" charter
- `/home/user/torchtune/torchtune/_recipe_registry.py` — where new configs get listed
- `/home/user/torchtune/tests/recipes/test_qat_*.py` — your test surface
- `/home/user/torchtune/README.md` line 1 — project status notice (worth reading before sinking weeks in)
