# Run 3 — Twice-tightened prompt + subagent answer

**Date:** 2026-05-20
**Subagent:** general-purpose (Claude Opus 4.7-class)
**Repo:** torchtune at BASE `213f38605`
**Estimated rubric pass rate:** ~75% (Medium band — tightening backfired)
**Significance:** Closest analog to the model answer the Step 5 eval will produce. Found extra couplings (`adapter_key_filter` substring, 35-site `adapter_cls`, QAT recipe drift) that neither Run 1 nor Run 2 found. **Still missed NF4 builder-time integration** — consistent miss with Run 2.

## The prompt sent

> I want to add FP8 QAT to torchtune — fake-quantize weights to FP8 during training, deploy as actual FP8 at inference. Single-device and distributed. This needs to compose with LoRA, and I want the design to still make sense if a sixth scheme shows up next year that needs both training-time fake-quant AND a custom forward graph for adapter composition. This is my first contribution.
>
> Walk me through what implementing this actually requires. I'm trying to follow the contribution rules — recipes stay readable as self-contained scripts, no implementation inheritance. I know the README says torchtune is wound down; assume for this question that the codebase will be forked and actively maintained going forward. Give me an honest read on whether the codebase shape makes this addition clean or whether I'd be adding pain future contributors will inherit, and if it's the latter, propose what to fix first.

## The subagent answer

# FP8 QAT in torchtune: implementation plan + honest read on the codebase shape

## TL;DR

Adding FP8 QAT touches **five seams** that don't share a coordinated abstraction: a torchao quantizer, an adapter class, a recipe-time module-swap helper, the checkpointer's adapter-state filter, and either two new recipes or invasive edits to the existing four QAT recipes. The mechanical work to plumb it through is doable in a week, but the **adapter-aware composition** (FP8-QAT × LoRA × FSDP/TP × inference export) lives across files that have to be edited in lockstep — and the "no inheritance, copy-paste-modify" norm guarantees that your "sixth scheme next year" makes the duplication worse. **Net call: it's clean enough to land FP8 QAT today following the existing pattern, but if you're forking for active maintenance, fix the seams below first or you're inheriting pain.**

## Where everything you need to touch lives

### The training-time fake-quant surface

`torchtune/training/quantization.py` is the single registry for QAT modes (`_quantizer_to_mode`, `_quantizer_mode_to_disable_fake_quant`, `_quantizer_mode_to_enable_fake_quant`). At BASE it knows about exactly four modes — `"8da4w"`, `"8da4w-qat"`, `"4w"`, `"4w-qat"` plus their `-module-swap` BC aliases. The two QAT quantizers come straight from `torchao.quantization.qat` and expose a `prepare(model)` / `convert(model)` interface plus `get_activation_fake_quantize_config()` / `get_weight_fake_quantize_config()` (the latter pair only used by the QAT+LoRA recipe).

Same file also already ships **real** FP8 training (`convert_to_float8_training` at line 247 — that swaps `nn.Linear` for torchao's `Float8Linear` and is wired only into `recipes/full_finetune_distributed.py` lines 211/617/627/1063 via `enable_fp8_training` and `fp8_recipe_name`). That helper is *not* fake-quant — it casts to fp8 dtype on the fly. You want a different code path.

The Llama3 TP plan in `torchtune/models/llama3/_parallelism.py` has a fork (`_get_fp8_llama_tp_training_plan` vs `_get_base_llama_tp_training_plan`) — confirmation that FP8 special-cases TP. Llama4 has the same fork at `torchtune/models/llama4/_parallelism.py:167+`.

### The LoRA composition surface

`torchtune/modules/peft/lora.py:150` defines `QATLoRALinear(LoRALinear)`. Its `forward()` (line 249) is `F.linear(fake_quant_act(x), fake_quant_weight(W))` + LoRA path. It instantiates `torchao.quantization.qat.fake_quantizer.FakeQuantizer` directly from two `FakeQuantizeConfig` objects. The swap-after-construction helper is `swap_lora_linear_with_qat` in `torchtune/training/quantization.py:205`, called from `recipes/qat_lora_finetune_distributed.py:466` inside `_convert_model_to_qat` (line 450). Note `_convert_model_to_qat` hardcodes "DoRA is not compatible with QAT" (line 455).

The model builders are the *other* PEFT entry point. Every model has its `_component_builders.py` doing the literal line `adapter_cls = DoRALinear if use_dora else LoRALinear` — I count **35 occurrences across the repo** (`llama3/`, `llama3_1/`, `llama4/`, `gemma/`, `gemma2/`, `qwen2/`, `qwen2_5/`, `qwen3/`, `phi3/`, `phi4/`, `mistral/`, `llama2/`). Any "third adapter class" has to be threaded through here or — as `QATLoRALinear` does — applied post-construction via a swap. The post-construction swap path is the only one that scales; the if/else doesn't.

### The checkpointer/adapter-state surface

`torchtune/modules/peft/_utils.py:115` is the load-bearing one:
```python
adapter_key_filter = lambda x: "lora" in x or "magnitude" in x
```
That hardcoded substring filter is what `get_adapter_state_dict` uses, which is called from `_checkpoint_client.py:170,365,537` and `_distributed.py:528`. Anything that isn't a `lora_*` or `magnitude` parameter gets classified as base-model and goes through the full HF-format conversion path. If your FP8 QAT introduces any new trainable scaling parameter or stateful fake-quantizer buffer (FP8 amax history is a real concern — `FakeQuantizer` from torchao keeps internal state), you have to decide whether it lives in the adapter blob or the base blob and you have to teach `get_adapter_state_dict` how to recognize it.

`get_merged_lora_ckpt` (`_utils.py:194`) is the inference-export step for LoRA — it does `state_dict[f"{module}.weight"] += (alpha/rank) * lora_b @ lora_a` and deletes lora keys. For FP8-QAT-LoRA you'd want to (a) merge LoRA into the high-precision base weight, then (b) actually cast/quantize that merged weight to FP8 for deployment. `recipes/quantize.py` exists for the QAT→convert step but it hardcodes torchao quantizer modes via `get_quantizer_mode` and there is **no** FP8 entry registered in `_quantizer_to_mode`. The convert step path is `"qat" in self._quantization_mode` → `self._quantizer.convert(self._model)` (line 92). You'll be adding a new mode and likely an `fp8_convert` step.

### The recipe surface

Four QAT recipes today: `qat_distributed.py` (1146 LOC), `qat_lora_finetune_distributed.py` (955 LOC), `qat_single_device.py` (719 LOC), plus `qat_lora_single_device` doesn't exist — note the gap. So you already need *two new* recipes (`qat_lora_single_device` doesn't exist, and your single-device path may need one if you don't fold it into the existing `qat_single_device`).

The quantizer is plumbed into recipes via `quantizer_cfg = cfg.get("quantizer", None)` → `config.instantiate(quantizer_cfg)` → `quantizer.prepare(model)` (in plain QAT recipes) or → `swap_lora_linear_with_qat(...)` (in QAT+LoRA). Both paths assume the torchao QAT `prepare()` API; FP8 doesn't have that today, you'd be writing it.

Delayed fake-quant (`fake_quant_after_n_steps`, qat_distributed:220, qat_single_device:532) uses the `_quantizer_mode_to_{enable,disable}_fake_quant` registry — your new mode needs entries there too. Note the registry's enable/disable helpers reference `FakeQuantizedLinear` (`training/quantization.py:63`) without importing it — currently a latent bug masked by the fact that `Int4WeightOnlyQATQuantizer.prepare()` swaps in `FakeQuantizedLinear` and the isinstance check works at runtime; for FP8 you'll either rely on the same torchao class or hit this dormant bug.

QAT also has `if cfg.get("compile", False): raise ValueError("Compile is not yet supported for QAT...")` in `qat_distributed.py:147`. FP8 training in `full_finetune_distributed.py` *does* support compile. Decide whether FP8-QAT inherits the QAT restriction or breaks it.

## The minimal implementation, mechanically

1. **torchao side (upstream prerequisite).** Either an `Float8QATQuantizer` exists in torchao (latest releases were heading in this direction) or you write one. It needs `prepare(model)`, `convert(model)`, and the `get_{activation,weight}_fake_quantize_config()` getters. Its `prepare` should module-swap `nn.Linear` for `FakeQuantizedLinear` with FP8 scaling configs. Until it lands upstream you can subclass it in `torchtune/training/quantization.py` next to `Int8DynActInt4WeightQuantizer`.

2. **Register it.** Add a `"fp8-qat"` entry to all three dicts in `torchtune/training/quantization.py` (`_quantizer_to_mode`, `_quantizer_mode_to_disable_fake_quant`, `_quantizer_mode_to_enable_fake_quant`). Fix or work around the `FakeQuantizedLinear` import issue while you're there.

3. **LoRA composition.** Either (a) reuse `QATLoRALinear` if FP8's `FakeQuantizer` plugs in via the same `FakeQuantizeConfig` interface — which is the cheap path — or (b) write `FP8QATLoRALinear(LoRALinear)` in `torchtune/modules/peft/lora.py` if FP8 needs a different forward graph. If (b), extend `swap_lora_linear_with_qat` to dispatch on quantizer type, or write `swap_lora_linear_with_fp8_qat`. The current `swap_lora_linear_with_qat` is already mode-blind — it takes two `FakeQuantizeConfig` objects and passes them through — so option (a) is probably enough.

4. **Adapter state-dict filter.** If your FP8 FakeQuantizer has trainable scales or stateful amax buffers, edit `torchtune/modules/peft/_utils.py:132`'s `adapter_key_filter`. Today it's a hardcoded substring match — turn it into something that asks each `AdapterModule` what to save, or at minimum extend the substring set. **This is the change with the longest blast radius**: 4 call sites in `_checkpoint_client.py` and `_distributed.py` rely on it.

5. **Recipes.** Two options:
   - **Cheap:** add `enable_fp8_qat`/quantizer detection into the existing `qat_distributed.py`, `qat_single_device.py`, `qat_lora_finetune_distributed.py`. Mirrors how `enable_fp8_training` was bolted into `full_finetune_distributed.py`. Pro: no new files. Con: makes those recipes harder to read, and you're not following the "self-contained" idiom.
   - **Idiomatic per `recipe_interfaces.py`:** copy-paste those three into `fp8_qat_distributed.py`, `fp8_qat_single_device.py`, `fp8_qat_lora_finetune_distributed.py`. Pro: each new file reads top-to-bottom. Con: you're now duplicating ~2800 LOC for a feature whose actual diff from existing QAT is ~30 lines.

   `recipe_interfaces.py:18` explicitly says *"torchtune strictly prohibits implementation inheritance"* and *"Minimizing code duplication is not the goal."* So the project's stated value system says option B. Honestly: pick A unless you have a contributor headcount that makes B sustainable.

6. **Configs.** Add `recipes/configs/<family>/<size>_fp8_qat.yaml` and `_fp8_qat_lora.yaml` per family you want to support, register them in `torchtune/_recipe_registry.py` (`Recipe` entries around lines 642–663 are the model).

7. **Inference / export.** Add an `Fp8WeightOnlyQuantizer` (or similar) to `torchtune/training/quantization.py` and register `"fp8"` in `_quantizer_to_mode`. The flow in `recipes/quantize.py:91-94` already branches on `"qat" in self._quantization_mode` → `convert()` vs `quantize()`, so as long as your QAT quantizer exposes `.convert()` and your inference quantizer exposes `.quantize()`, the existing recipe should just work.

8. **TP plan.** If FP8-QAT needs different all-gather behavior under TP (the real FP8 path does — see `Float8ColwiseParallel`/`Float8RowwiseParallel` in `_parallelism.py`), add a third TP-plan variant in `torchtune/models/llama3/_parallelism.py` and the same for Llama4. Or just declare TP unsupported in the initial PR — that's what `_validate_float8_tp_plan` in `training/quantization.py:277` is paving the way for.

9. **Tests.** `tests/recipes/test_qat_lora_finetune_distributed.py` is the template — small-scale integration test marked `@pytest.mark.integration_test`. Also unit-test the new quantizer in `tests/torchtune/training/test_quantization.py`.

## Honest read: is this clean?

**No.** Here are the load-bearing reasons, ranked by how badly they'll bite the "sixth scheme next year":

### 1. The adapter filter is a substring match on weight names (`_utils.py:132`)
```python
adapter_key_filter = lambda x: "lora" in x or "magnitude" in x
```
This is the *only* mechanism by which the checkpointer distinguishes adapter weights from base weights, and it's called from four places. Every new adapter (DoRA shipped `magnitude`; a future "sixth scheme" with a new scale parameter would need its own substring). For FP8 specifically: the FP8 FakeQuantizer may carry amax buffers that need to be either checkpointed-with-adapter or stripped on save, and there's no clean way to express that.

**Fix-first proposal:** make `AdapterModule.adapter_params()` (already a protocol in `torchtune/modules/peft/_utils.py:20`) the source of truth, and rewrite `get_adapter_state_dict` to walk the model and use those names, instead of substring-matching on the flat state dict. One change, removes a whole class of future bugs.

### 2. `swap_lora_linear_with_qat` is named for one specific combination
`training/quantization.py:205` is the *only* mechanism for "take a LoRA model and add training-time numerical-precision wrappers to it." It's already what FP8-QAT-LoRA wants to reuse. Today its name implies it only does int QAT, and `QATLoRALinear.from_lora_linear` (`lora.py:267`) accepts only `FakeQuantizeConfig` (torchao QAT's config object). If FP8 reuses `FakeQuantizeConfig` you're fine; if it needs anything different you'll fork `QATLoRALinear`. A *seventh* scheme that wants a different forward (your hypothetical) will definitely fork it, and the existing pattern says "write a new module + a new swap function" rather than "extend the existing one."

**Fix-first proposal:** rename to `apply_training_precision_to_lora(module, scheme_config)` and define a small protocol — `apply_to_layer(lora_linear) -> nn.Module` — that any precision scheme (int QAT, FP8 QAT, your future scheme) implements. The swap loop in `training/quantization.py:231-244` becomes scheme-agnostic. Today it's already structurally agnostic; this is just a rename + protocol decoration. Cheap.

### 3. The `_quantizer_to_mode` string registry is global mutable state
`training/quantization.py:51-53`. Modes are registered by mutating dicts at import time. Discovery is "grep for `_quantizer_mode_to_*`". For a long-lived fork with N quantization schemes, this becomes a global namespace with no validation that all three dicts agree. The `FakeQuantizedLinear` reference in `_enable_linear_fake_quant` (line 63) without an import is exactly the kind of latent break this style invites.

**Fix-first proposal:** a `QuantizationScheme` dataclass (`mode_name`, `quantizer_cls`, `enable_fake_quant_fn`, `disable_fake_quant_fn`, `supports_lora`, `convert_fn`), registered once per scheme. Drop the three parallel dicts. This is the smallest structural change with the biggest payoff for "sixth scheme next year."

### 4. Recipe duplication is doctrinal, not accidental
`recipe_interfaces.py:18` literally says the interface is "not a vehicle for code reuse" and that minimizing duplication "is not the goal." The four QAT recipes alone are ~2800 LOC of near-identical code — the FSDP setup, the gradient accumulation loop, the validation loop, the `fake_quant_after_n_steps` plumbing, the checkpoint client wiring, the optimizer-in-bwd handling all repeat. `qat_distributed.py` is 1146 LOC; the QAT-specific *logic* in it is the ~20 lines around `quantizer.prepare(model)` and the `_fake_quant_after_n_steps` block.

If you copy-paste those into two new FP8 QAT recipes, you've doubled the surface for any future cross-cutting bug fix (e.g. a logging field, a CP setup tweak, a checkpoint-resume edge case). The QAT recipes have already drifted: `qat_distributed.py` has a `compile=True` guard that raises (line 147), `qat_single_device.py` quietly supports compile (line 254), and only the single-device path implements `fake_quant_after_n_steps` enforcement in the training loop (line 532) — the distributed recipe records the config value but doesn't act on it.

**Honest take on "fix-first":** this one I don't recommend tackling in your first PR — it's the explicit project value system and you'll lose the social argument. But internally for your fork: a thin `TrainingLoopMixin` or `RecipeContext` dataclass that holds the loop state and yields it to the recipe would let your recipes shrink to ~200 LOC of orchestration each without "inheritance." Sequel discussion, not the FP8 PR.

### 5. Two PEFT-injection paths, neither pluggable
`adapter_cls = DoRALinear if use_dora else LoRALinear` in the model `_component_builders.py` (35 sites) is one path; post-construction `swap_lora_linear_with_qat` is the other. New adapters are added either by literally editing 35 files (DoRA's route) or by the swap pattern (QAT-LoRA's route). The swap pattern scales; the if/else doesn't. Your FP8-QAT can use the swap pattern and stay out of the model builders, but if anyone ever wants `lora_llama3_8b(use_fp8_qat=True)` they're back in 35-files-land.

**Fix-first proposal:** the `adapter_cls` factory in `_component_builders.py` should take an `adapter_factory` callable from the recipe, not a `use_dora: bool`. Same pattern as `loss` / `optimizer` instantiation — config drives a callable. This is invasive (touches every model builder) but pays for every future adapter.

## Recommendation

If you're forking and committing to maintain this, in priority order:

1. **Before FP8 lands:** fix `adapter_key_filter` to consult `AdapterModule.adapter_params()`. One file (`peft/_utils.py`), all four callsites benefit, tested by existing LoRA/DoRA/QAT-LoRA recipes.
2. **With FP8:** consolidate `_quantizer_to_mode` and friends into a `QuantizationScheme` registry. Adds maybe 30 LOC, removes the import-time mutation footgun, makes adding scheme #6 mechanical.
3. **With FP8:** rename `QATLoRALinear`/`swap_lora_linear_with_qat` to scheme-agnostic names and define the protocol. FP8-QAT-LoRA reuses it, sixth scheme reuses it.
4. **After FP8 lands:** open a separate RFC for the recipe-duplication story. Don't pick that fight in the FP8 PR.

If you do *just* the three small ones, you'll add FP8 QAT in ~600 LOC net (two recipes, configs, a quantizer registration, a fake-quant LoRA adapter variant if needed) and meaningfully reduce the cost of the next contributor adding scheme #6. If you skip them and follow the existing copy-paste shape, your PR will be ~3000 LOC, will pass review under the project's stated values, and you will personally regret it the third time someone asks you to add a new quantization mode.

## Key files (absolute paths) for reference

- `/home/user/torchtune/torchtune/training/quantization.py` — quantizer registry, FP8 training helper, QAT mode dicts
- `/home/user/torchtune/torchtune/modules/peft/lora.py` — `LoRALinear`, `QATLoRALinear`
- `/home/user/torchtune/torchtune/modules/peft/dora.py` — `DoRALinear` (the "third adapter" precedent)
- `/home/user/torchtune/torchtune/modules/peft/_utils.py` — `adapter_key_filter`, `get_merged_lora_ckpt`, `AdapterModule` protocol
- `/home/user/torchtune/torchtune/recipe_interfaces.py` — the "no inheritance" doctrine, lines 16-23
- `/home/user/torchtune/recipes/qat_distributed.py` — distributed QAT recipe (1146 LOC)
- `/home/user/torchtune/recipes/qat_single_device.py` — single-device QAT, owns `fake_quant_after_n_steps` enforcement
- `/home/user/torchtune/recipes/qat_lora_finetune_distributed.py` — QAT+LoRA template, `_convert_model_to_qat` at line 450
- `/home/user/torchtune/recipes/quantize.py` — post-training convert step
- `/home/user/torchtune/recipes/full_finetune_distributed.py` — only recipe with FP8 (real, not QAT) plumbing
- `/home/user/torchtune/torchtune/models/llama3/_parallelism.py` — TP-plan FP8 fork (template for FP8-QAT TP plan)
- `/home/user/torchtune/torchtune/training/checkpointing/_checkpoint_client.py` — calls `get_adapter_state_dict` at three sites
- `/home/user/torchtune/torchtune/_recipe_registry.py` — where new recipe entries go (around line 642)
