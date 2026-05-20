# Run 4 — User-refined prompt (calibration test)

**Prompt under test:** user-refined version. Key deltas vs. Run 3:
- Naive framing: "first contribution to torchtune", "contribution rules don't feel intuitive"
- Typos: "FP8 QUART" (instead of QAT), "fakes quantize qeights"
- Asks for explicit "step by step plan"
- Adds "ideally on top of LoRA" (more concrete than Run 3)
- Keeps Run 3 tweaks: "next scheme next year needs custom fake-quant AND custom forward graph for adapter composition"
- Keeps Run 3 tweak: "assume the repo will be forked and maintained" (no-maintenance-escape)
- Adds "optimize for future developers to maintain the code" — explicit maintainability frame

**Subagent setup:** general-purpose, low context (no CLAUDE.md, no M/), absolute path to repo.

**Date:** 2026-05-20

---

## Subagent answer (verbatim)

Short version: torchtune doesn't have a single "quantization plugin" seam — there are three independent integration points already in the tree, and you'll touch all three. The work breaks into (a) a new fake-quant adapter `nn.Module`, (b) wiring it into the QAT mode tables in `training/quantization.py`, (c) a new pair of recipes (single-device + distributed) cloned from the QAT+LoRA family, and (d) a small post-training conversion step that turns the trained model into actual FP8 linears for inference. Below I cite the files you'll touch and call out the project rules that look unintuitive but matter.

---

## 1. Where FP8 / QAT / LoRA live today

You should read these end-to-end before writing code; they are the prior art you are extending.

- **Quantization integration surface:** `torchtune/training/quantization.py`. This single file is the "registry" — it has three integration patterns side by side:
  - PTQ wrappers (`Int4WeightOnlyQuantizer`, `Int8DynActInt4WeightQuantizer`) that call `torchao.quantization.quantize_` (lines 79–121).
  - QAT mode tables `_quantizer_to_mode`, `_quantizer_mode_to_disable_fake_quant`, `_quantizer_mode_to_enable_fake_quant` and the `get_quantizer_mode` lookup (lines 51–53, 94–97, 124–127, 167–202). This is the seam recipes read to know whether to call `disable_fake_quant`/`enable_fake_quant` during warmup.
  - QAT+LoRA bridge `swap_lora_linear_with_qat(...)` (lines 205–244) that walks the module tree and replaces every `LoRALinear` with `QATLoRALinear`, preserving weights including the `device=meta` case.
  - FP8 *training* (not QAT — bf16/fp32 master weights with FP8 matmul) lives in `convert_to_float8_training` (lines 247–273) and is a thin wrapper around `torchao.float8.convert_to_float8_training`. Note this is **dynamic FP8 in the forward pass for full-finetune**, *not* fake-quant — useful as a reference for how torchtune already plumbs FP8 through FSDP + TP, but a different code path than what you want.

- **The LoRA adapter that already does fake-quant on top of LoRA:** `torchtune/modules/peft/lora.py`. `LoRALinear.forward` at lines 128–147 implements `out = F.linear(x, W) + (α/r) * B(A(dropout(x)))`. `QATLoRALinear` (lines 150–266) is the existing reference design for "fake-quant the base path while leaving the adapter in higher precision":
  ```
  _x = activation_fake_quantizer(x)
  w  = weight_fake_quantizer(self.weight)
  out = F.linear(_x, w) + (alpha/rank) * lora_b(lora_a(dropout(x)))
  ```
  Crucially, the fake-quantizers are `torchao.quantization.qat.fake_quantizer.FakeQuantizer` instances configured by `FakeQuantizeConfig` (lines 222–247). Today `FakeQuantizeConfig` is keyed by `dtype=torch.int4/int8` — torchao's QAT API does accept `torch.float8_e4m3fn` / `torch.float8_e5m2`, so most of your work is plumbing, not new math. (Validate this against the torchao version pinned in `pyproject.toml` before you start — the recipe constructor at line 222 already asserts torchao ≥ 0.7.)

- **The reference QAT+LoRA recipe:** `recipes/qat_lora_finetune_distributed.py`. Read in particular:
  - `_convert_model_to_qat` (lines 450–466): instantiates the quantizer from YAML, pulls `quantizer.get_activation_fake_quantize_config()` / `get_weight_fake_quantize_config()` off it, then calls `swap_lora_linear_with_qat`. This is the contract your new quantizer class has to satisfy.
  - `_setup_model` (lines 468–600): instantiates the model on `meta`, calls `_convert_model_to_qat` while still on meta, then runs `set_trainable_params(model, get_adapter_params(model))` so only LoRA params have `requires_grad=True`. The QAT swap must run before `set_trainable_params` because `QATLoRALinear` is a strict subclass of `LoRALinear` and inherits `AdapterModule`.
  - Lines 575–584: `validate_missing_and_unexpected_for_lora` runs after weight loading. Your new module **must** keep `lora_a.weight` / `lora_b.weight` / `weight` parameter names so `get_adapter_state_dict` (`torchtune/modules/peft/_utils.py:115–133`, the `"lora" in x or "magnitude" in x` filter) and `get_merged_lora_ckpt` (`_utils.py:194–271`) keep working unchanged.

- **Checkpointer adapter coupling:** `torchtune/training/checkpointing/_checkpoint_client.py:23–26, 167–186, 362–384`. The checkpointer is hard-coded to detect adapters by the substring `lora` (or `magnitude` for DoRA). If you keep parameter names `lora_a`, `lora_b`, you inherit save/load/merge for free. If you invent new adapter parameter names, you will be editing the checkpointer, which is the tension I'd warn you about most.

- **Recipe registry / CLI:** `torchtune/_recipe_registry.py:642–663` is where `qat_distributed` and `qat_lora_finetune_distributed` are registered for `tune run` / `tune ls`. Any new recipe + config has to be appended here.

- **Post-training "real quantization" recipe:** `recipes/quantize.py`. Today this only handles the `Int8DynAct/Int4WO` quantizers — for FP8 you'll either reuse it (preferred) or add a small companion recipe that uses `torchao.quantization.quantize_` with an FP8 config to materialize the trained model as real FP8 for inference/serving.

---

## 2. The unintuitive contribution rules, and how to read them

You flagged these as not feeling intuitive, so let me restate them as they apply to your PR:

- **No implementation inheritance** (`torchtune/recipe_interfaces.py:13–14, 24, 66–67`, `CONTRIBUTING.md:175`). The recipes are protocols (`FTRecipeInterface`), not base classes. The docstring is explicit: *"torchtune strictly prohibits implementation inheritance"* and *"Minimizing code duplication is not the goal. Recipe-writers are encouraged to copy-paste-modify."* What this means for you: you do **not** subclass `QATLoRAFinetuneRecipeDistributed`. You copy the file, rename the class, and edit. Annoying, but it's how the repo is.
  - This rule has one principled crack: `QATLoRALinear(LoRALinear)` in `lora.py:150`. Modules are allowed to inherit because they're "modular blocks" (`CONTRIBUTING.md:174`). So a new `FP8QATLoRALinear(LoRALinear)` is consistent. Adding it as a separate class rather than a third arm of `QATLoRALinear` is the right call for forward-extensibility (your point about "next scheme next year").

- **Limit generalization** (`CONTRIBUTING.md:178`). Resist the temptation to refactor the QAT mode tables into a generic plugin protocol "while you're there." That refactor would be a separate PR and a different conversation; if you bundle it with FP8 QAT, the review will stall.

- **Clean interfaces / no intrusive configs** (`CONTRIBUTING.md:176–177`). Configs interact through builders. So your fake-quantizer takes flat args (`fp8_dtype`, `granularity`, `scaling_recipe`), not a `DictConfig`.

So the *idiomatic* way to add a new fake-quant scheme that needs its own forward graph is:
1. New module class in `torchtune/modules/peft/` (or a new `low_precision/` file if you want to reuse it outside LoRA later).
2. New quantizer class in `torchtune/training/quantization.py` that registers in the three mode tables.
3. New `swap_*` helper alongside `swap_lora_linear_with_qat`.
4. New recipe files (copy + edit), new YAML configs, new entries in `_recipe_registry.py`.

That's a lot of files for one feature. It's intentional. The payoff is what your question hints at: when the next scheme arrives next year, it lands as additive code, not as edits to a shared base class that has now collected three special cases.

---

## 3. Step-by-step plan

I'd split this into 5 PRs for review-ability; you can fold 1–2 together if you want.

### PR 1 — `FP8QATLoRALinear` module

**New file:** `torchtune/modules/peft/fp8_qat_lora.py` (don't shove it into `lora.py` — that keeps the diff to one file per scheme; aligned with "no implementation inheritance, prefer modular blocks").

```python
class FP8QATLoRALinear(nn.Module, AdapterModule):
    """
    LoRA linear with FP8 fake-quantization applied to the frozen base path.

    Forward: out = F.linear(fake_q_fp8(x), fake_q_fp8(W)) + (alpha/rank) * B(A(dropout(x)))

    Trains in bf16/fp32; the fake-quantizers simulate FP8 numerics (e4m3 / e5m2).
    After training, call `to_fp8_inference()` to produce a real-FP8 module for serving.
    """
```

Design notes that matter for the "next scheme" requirement:
- Inherit from `nn.Module` + `AdapterModule` directly. Do **not** inherit from `LoRALinear`. The existing `QATLoRALinear(LoRALinear)` re-uses the base path's plain `nn.Linear`-style weight, but the QAT case still calls `F.linear(_x, w)`. If your "next scheme" needs a custom forward graph (per your question — "custom forward graph for adapter composition"), inheriting from `LoRALinear` will force `super().__init__()` to register a base weight with a fixed layout. Implement the four parameters (`weight`, `lora_a`, `lora_b`, optional `bias`) directly; reuse `_lora_a_init_params` / `_lora_b_init_params` from `lora.py:309–320`.
- Take `activation_qat_config` and `weight_qat_config` as `Optional[FakeQuantizeConfig]` constructor args — same signature shape as `QATLoRALinear.__init__` (lines 198–247). This is what lets `swap_*` and the recipe code reuse the existing config-flow.
- Implement `adapter_params(self) -> ["lora_a.weight", "lora_b.weight"]` (matches `lora.py:116–126`) so `get_adapter_params`, `get_adapter_state_dict`, `get_merged_lora_ckpt`, and `validate_missing_and_unexpected_for_lora` all work without touching the checkpointer (`_utils.py:115–271`).
- Implement `from_lora_linear(cls, lora_linear, activation_qat_config, weight_qat_config)` analogous to `lora.py:267–306`, including the `device=='meta'` guard. This is the contract `swap_*` calls.
- Add a `to_fp8_inference(self) -> nn.Module` method (or a free function). After QAT is done you merge the adapters (`get_merged_lora_ckpt`, see `_utils.py:264–266`), then convert the merged `nn.Linear` to a real FP8 `Float8Linear` (you can reuse `torchao.float8` machinery; `torchtune/training/quantization.py:247–273` shows the existing FP8 conversion pattern). Keeping this as a method on the module is OK; making it a free function in `torchtune/training/quantization.py` is more consistent with the existing `convert_to_float8_training` style. I'd put it next to `convert_to_float8_training` and import it where needed.

**Export it:** add to `torchtune/modules/peft/__init__.py` (lines 7–35).

**Tests:** add `tests/torchtune/modules/peft/test_fp8_qat_lora.py` modeled on `tests/torchtune/modules/peft/test_lora.py`. Cover: (a) shape parity vs. `LoRALinear`, (b) `from_lora_linear` round-trips weights, (c) `adapter_params()` returns expected names, (d) `disabled=True` matches base path, (e) `to_fp8_inference()` produces a `Float8Linear` whose forward is numerically close to the QAT forward after `disable_fake_quant`.

### PR 2 — Quantizer + registry hookup in `training/quantization.py`

Add to `torchtune/training/quantization.py`:

```python
class Float8WeightActQATQuantizer:
    """Fake-quantize weights and activations to FP8 (e4m3 by default)."""
    def __init__(self, weight_dtype="float8_e4m3fn", activation_dtype="float8_e4m3fn", granularity="per_tensor"):
        ...
    def get_activation_fake_quantize_config(self): ...
    def get_weight_fake_quantize_config(self): ...
```

Why this shape: the recipe at `qat_lora_finetune_distributed.py:464–466` calls exactly these two getters. Matching that surface means **zero changes to the recipe scaffolding to swap quantizers** — the recipe is config-driven. Validate the FakeQuantizeConfig FP8 path works with the torchao version pinned in `pyproject.toml`; you may need to bump the floor (`raise ValueError("Need torchao version 0.X+")` at lines 46–49).

Register in the three mode tables (lines 51–53 + the pattern at 94–97):
```python
_quantizer_to_mode[Float8WeightActQATQuantizer] = "fp8-qat"
_quantizer_mode_to_disable_fake_quant["fp8-qat"] = _disable_linear_fake_quant
_quantizer_mode_to_enable_fake_quant["fp8-qat"] = _enable_linear_fake_quant
```
The existing `_enable_linear_fake_quant` / `_disable_linear_fake_quant` helpers (lines 56–71) just walk the tree looking for `FakeQuantizedLinear` and toggle `.enabled` on its fake-quantizers. As long as your `FP8QATLoRALinear` puts its `FakeQuantizer` instances at attributes named `activation_fake_quantizer` / `weight_fake_quantizer` (same as `QATLoRALinear` at lines 232, 245) you'll need one tiny tweak: either change the helper to also check for your new module type, or — cleaner — make the helpers check by attribute name, not class. I'd do the latter in this PR (`if hasattr(mod, "activation_fake_quantizer"): ...`). It's a 5-line behavior change and worth flagging in the PR description so reviewers understand it generalizes the existing seam without taking the bigger "design a quantization plugin protocol" hit.

Add `swap_lora_linear_with_fp8_qat(module, ...)` next to `swap_lora_linear_with_qat` (lines 205–244). Copy-paste-modify — yes, this duplicates ~20 lines. That is what `CONTRIBUTING.md:178` is asking you to do; resist the urge to factor out a generic `_swap_adapter_module` until there are three of these.

Add it to `__all__` (lines 35–43) and to the docstring of `get_quantizer_mode` (lines 167–186).

Update `recipes/quantization.md` with the new mode string and a one-paragraph example.

### PR 3 — `fp8_qat_lora_finetune_distributed.py` recipe + config

**Copy** `recipes/qat_lora_finetune_distributed.py` to `recipes/fp8_qat_lora_finetune_distributed.py`. The diff vs. the source should be:

- Class rename: `QATLoRAFinetuneRecipeDistributed` → `FP8QATLoRAFinetuneRecipeDistributed`.
- Import `swap_lora_linear_with_fp8_qat` instead of `swap_lora_linear_with_qat`.
- In `_convert_model_to_qat` (the source lines 450–466), call `swap_lora_linear_with_fp8_qat`.
- Optionally guard `"fp8-qat" not in quantizer_mode` instead of `"qat" not in quantizer_mode` (source line 460). I'd actually keep the broader check and rely on the registry to make sure only your FP8 QAT quantizer is selectable — but a tighter check fails loudly when someone uses the wrong YAML, which is friendlier.
- Everything else (FSDP wiring, `set_trainable_params`, `validate_missing_and_unexpected_for_lora`, checkpointing) is identical.

This will feel wrong — you're duplicating ~950 lines of recipe. It is the rule. The upside: when distributed FP8 QAT needs a quirk that pure-int QAT doesn't (e.g., interaction with `precompute_float8_dynamic_scale_for_fsdp` à la `full_finetune_distributed.py:1063–1067`), the fix lands in *one* recipe and can't accidentally break the int-QAT path.

**Configs:** add `recipes/configs/llama3/8B_fp8_qat_lora.yaml` and `recipes/configs/llama3_1/8B_fp8_qat_lora.yaml` (start with two; add more as users ask). Pattern off `recipes/configs/llama3/8B_qat_lora.yaml` and swap the `quantizer._component_` to your new class. Validate with `tune validate <config>`.

**Register:** add a new `Recipe(...)` entry in `torchtune/_recipe_registry.py:642–663` so `tune run fp8_qat_lora_finetune_distributed --config llama3/8B_fp8_qat_lora` works.

**Recipe test:** add `tests/recipes/test_fp8_qat_lora_finetune_distributed.py` modeled on `tests/recipes/test_qat_lora_finetune_distributed.py` (if it exists) or `tests/recipes/test_lora_finetune_distributed.py`. Marker `@pytest.mark.integration_test` plus `@gpu_test` per `CONTRIBUTING.md:48–53`. Skip cleanly if the runner doesn't have FP8-capable hardware (H100+ / sm90).

### PR 4 — single-device recipe

Copy `recipes/qat_single_device.py` → `recipes/fp8_qat_lora_finetune_single_device.py`. (No single-device QAT+LoRA recipe exists yet — see `qat_single_device.py:719` lines is QAT *without* LoRA. So you're also doing what `qat_lora_finetune_distributed` does, on a single device. Treat this PR as "QAT+LoRA single device, FP8 flavor only" — that's still scope-honest.)

Same wiring as PR 3 with the distributed bits stripped. Add config files, registry entry, and a recipe test. If you want to land FP8 QAT+LoRA broadly, you could also add an int-QAT single-device LoRA recipe — but I'd not bundle that here.

### PR 5 — Inference conversion path

This is the "deploys as actual FP8 during inference" piece. Two options, in order of preference:

**Option A (preferred):** extend `recipes/quantize.py` to handle the FP8 mode. The recipe (lines 22–127) already takes a YAML pointing at a quantizer and a checkpointed model. Add a branch: when `quantization_mode == "fp8-qat"`, (i) merge LoRA adapters via `get_merged_lora_ckpt(state_dict, rank, alpha)` (already in scope through `_utils.py:194`), (ii) instantiate the base model, (iii) call your new free function in `training/quantization.py` that converts each `nn.Linear` to a real `Float8Linear` for inference. (Look at `convert_to_float8_training`, lines 247–273, for the wrapper style; you probably want a sibling `convert_to_float8_inference` that uses `torchao.float8`'s static/PTQ FP8 config instead of the training-time one.) The output is a real-FP8 `state_dict` saved with `FullModelTorchTuneCheckpointer`, consumable by `recipes/generate.py` and `recipes/eleuther_eval.py`.

**Option B:** ship a separate `recipes/fp8_inference_convert.py`. Same logic, but it doesn't touch existing code. Worse because users now have two PTQ recipes to choose from.

Document in `recipes/quantization.md`.

---

## 4. The "make-it-survive-the-next-scheme" piece

You asked for the design to still make sense if next year a scheme arrives that needs both custom fake-quant *and* a custom forward graph for adapter composition. Here is what makes that easy in this plan:

1. **The module is independent.** `FP8QATLoRALinear` inheriting from `nn.Module + AdapterModule` (not from `LoRALinear`) means the next scheme can do the same: define a totally different forward graph, expose `adapter_params()`, and not be entangled with LoRA's forward.

2. **The `swap_*` pattern is per-scheme.** `swap_lora_linear_with_qat` and `swap_lora_linear_with_fp8_qat` are siblings. The next scheme adds a sibling. No central dispatch table to grow.

3. **The mode-table seam is the only shared registry**, and that's fine: it's three flat dicts of strings → callables in one file. The recipe just calls `get_quantizer_mode(quantizer)` and the matching enable/disable function. New schemes register; nothing else changes.

4. **Adapter state-dict handling is structural (substring `"lora"`/`"magnitude"`), not type-based.** As long as new schemes name their adapter parameters with `lora_*` or `magnitude` (or extend the filter), the checkpointer keeps working. Document this convention in the new module's docstring — it's the implicit contract.

5. **Recipes are duplicated, deliberately.** This is the part that will feel ugly. The defense is `recipe_interfaces.py:18–19` ("Minimizing code duplication is not the goal"). If three QAT-LoRA recipes accumulate and they share, say, a complex new gradient-scaling pre-step, *then* you raise refactoring as a separate proposal — by then there's evidence for what should be shared and what shouldn't.

The one thing this plan deliberately does **not** do, and that I'd push back on doing in your first contribution: introduce a generic "fake-quant adapter plugin" base class or factory. The codebase has actively rejected that direction (`CONTRIBUTING.md:175, 178`, `recipe_interfaces.py:13–14`). You'll get a better review by adding code than by trying to refactor the seam.

---

## 5. Tradeoffs and uncertainties I'd flag in the PR description

- **torchao version floor.** `torchtune/training/quantization.py:46–49` asserts torchao ≥ 0.7. FP8 `FakeQuantizeConfig` support and the FP8 PTQ convert function may require a newer floor — confirm against the torchao version pinned in your local install before claiming compatibility. If you have to bump the floor, mention it explicitly; it affects every user, not just FP8 ones.
- **FSDP + FP8 fake-quant.** torchtune's existing FP8 *training* path (`full_finetune_distributed.py:617–627`) gates on a specific PyTorch nightly (`torch.__version__ < "2.8.0.dev20250318"`). Your fake-quant path doesn't need that — the matmul is in bf16/fp32 and only the quantize step casts to FP8 numerics — but verify on the target torch version before claiming "distributed support."
- **Hardware gating.** Real FP8 inference needs sm89 (Ada) or sm90 (Hopper). The QAT step doesn't, since it's fake. Make the inference-conversion recipe fail with a clear message on older hardware; don't surface this in the QAT recipe.
- **TP plan.** The existing FP8 TP plan (`Float8ColwiseParallel` / `Float8RowwiseParallel` and `_validate_float8_tp_plan` at `quantization.py:277–295`) is only used by `full_finetune_distributed`. Your FP8 QAT recipe is bf16-training-with-fake-quant, so it should follow the *regular* (non-FP8) TP plan — be explicit about that in the docstring.
- **DoRA.** The existing QAT+LoRA recipe explicitly rejects DoRA (`qat_lora_finetune_distributed.py:454–456`). Do the same. Don't try to fix this in scope.

---

## Files you'll touch

New:
- `/home/user/torchtune/torchtune/modules/peft/fp8_qat_lora.py`
- `/home/user/torchtune/recipes/fp8_qat_lora_finetune_distributed.py`
- `/home/user/torchtune/recipes/fp8_qat_lora_finetune_single_device.py`
- `/home/user/torchtune/recipes/configs/llama3/8B_fp8_qat_lora.yaml` (and siblings)
- `/home/user/torchtune/tests/torchtune/modules/peft/test_fp8_qat_lora.py`
- `/home/user/torchtune/tests/recipes/test_fp8_qat_lora_finetune_distributed.py`

Edited:
- `/home/user/torchtune/torchtune/training/quantization.py` (new quantizer class, new `swap_*` helper, new mode-table entries, optionally generalize `_enable_linear_fake_quant` to check by attribute)
- `/home/user/torchtune/torchtune/modules/peft/__init__.py` (export `FP8QATLoRALinear`)
- `/home/user/torchtune/torchtune/_recipe_registry.py` (register the two new recipes)
- `/home/user/torchtune/recipes/quantize.py` (handle the `"fp8-qat"` mode for PTQ conversion to real FP8)
- `/home/user/torchtune/recipes/quantization.md` (docs)
- `/home/user/torchtune/docs/source/api_ref_*.rst` as appropriate (`CONTRIBUTING.md:81–83`)
