# Run 1 — Over-coached prompt + subagent answer

**Date:** 2026-05-20
**Subagent:** general-purpose (Claude Opus 4.7-class)
**Repo:** torchtune at BASE `213f38605`
**Estimated rubric pass rate:** ~80% (Easy–Medium band — too easy)
**Significance:** Closest to the structural template for `golden_answer.md`. Use this as the golden's outline.

## The prompt sent

(With explicit breadcrumbs — see `session-log.md` for what was coached.)

> I want to add FP8 QAT (quantization-aware training where weights are fake-quantized to FP8 during training, then deployed as actual FP8 at inference) to torchtune. I want it to work for both single-device and distributed setups, ideally on top of LoRA. This is my first contribution to the codebase.
>
> From poking around, I can see FP8 already exists somewhere in the repo (there's a `convert_to_float8_training` function and an `enable_fp8_training` config option) and QAT already exists for Int4/Int8 (there are dedicated `qat_*` recipes). But FP8 QAT specifically doesn't exist, and I genuinely don't know where it would go.
>
> Walk me through an end-to-end plan that respects the contribution rules — recipes stay readable as self-contained scripts, no implementation inheritance. As you go, please flag two things:
>
> 1. The places in the codebase where quantization currently gets integrated into a model's lifecycle. I have a feeling there's more than one and they happen at different points (module construction, after loading weights, mid-training, etc.). Where are they, and *why* do they happen at different points — what would break if you moved one to a different stage?
>
> 2. Given those existing integration points, where does FP8 QAT have to plug in? And honestly: does the current shape of the codebase make adding FP8 QAT painful enough that I should propose a redesign first, before adding a fifth quantization path? If so, what does a design look like that lets future schemes declare which lifecycle hooks they need, without violating the no-inheritance policy?

## The subagent answer

# FP8 QAT in torchtune: where it has to plug in, and whether to redesign first

Short version: yes, there are **four** distinct quantization integration points today, they happen at different lifecycle phases because each scheme touches different things (the parameter object, the module type, the forward graph, or the saved state dict), and FP8 QAT — because it's the first scheme that needs *both* fake-quant at training time and a real low-precision representation at inference time — is exactly the case that will make the existing seams crack. I'd recommend doing a small, targeted refactor of `torchtune/training/quantization.py` to introduce a "quantization scheme" descriptor object before you write a fifth recipe; details below.

I'm going to walk the lifecycle first, then come back to FP8 QAT.

## 1. Where quantization currently plugs into a model

Pinning these to BASE `213f38605`:

### (a) Builder-time, inside the model factory — QLoRA (NF4)

Path: `torchtune/models/llama3/_component_builders.py:129-149, 176-292`, hitting `torchtune/modules/low_precision/nf4_linear.py` (`FrozenNF4Linear`).

When you set `quantize_base=True` on a LoRA model builder (e.g. `lora_llama3_8b(..., quantize_base=True)` in `torchtune/models/llama3/_model_builders.py:112`, exposed to YAML as `qlora_llama3_8b`), the builder swaps `nn.Linear` for `FrozenNF4Linear` *at construction time*. The module already owns a real `NF4Tensor` weight by the time the recipe sees the model. `LoRALinear.__init__` itself also calls `to_nf4` on its base weight when `quantize_base=True` (`torchtune/modules/peft/lora.py:82-87`).

Why here? Because NF4 changes the *parameter object*, not the forward. The weight is a tensor subclass, the forward path is `linear_nf4(input, weight)`, and downstream code (FSDP sharding, `load_state_dict`, checkpoint gather) has to know it's looking at NF4 from the start. If you tried to do this *after* sharding, you'd have to re-wrap every `DTensor._local_tensor` — which is exactly the painful path `gather_cpu_state_dict` in `torchtune/training/_distributed.py:462-517` exists to undo.

### (b) Post-construction, pre-shard — FP8 mixed-precision training & Int4/Int8 QAT

Paths:
- FP8: `torchtune/training/quantization.py:247-273` (`convert_to_float8_training`), invoked from `recipes/full_finetune_distributed.py:617-627`.
- QAT: `torchtune/training/quantization.py:79-153` (the `Int8DynActInt4WeightQuantizer` / `Int4WeightOnlyQuantizer` / `*QATQuantizer` classes), invoked via `quantizer.prepare(model)` at `recipes/qat_distributed.py:677` and `recipes/qat_single_device.py:386`.

Both do a **module swap on the already-built model**: FP8 replaces every `nn.Linear` (except `output`) with `Float8Linear`; QAT replaces them with torchao's `FakeQuantizedLinear`. Crucially the recipe does this *after* `config.instantiate(cfg_model)` and *before* `parallelize_module` / `shard_model` / `load_from_full_model_state_dict`. Look at the ordering in `recipes/full_finetune_distributed.py:611-715` — model on meta → fp8 swap → TP → FSDP → load full state dict.

Why here and not earlier or later?
- Earlier (in the builder) would mean every model factory needs `enable_fp8_training=...` plumbing and a parallel set of `Float8Linear`/`FakeQuantizedLinear` builders. Right now the recipe owns the swap; the builder stays clean.
- Later (after `load_from_full_model_state_dict`) would break in two ways. (i) FSDP/TP sharding plans are configured by *module type* — `Float8ColwiseParallel` vs `ColwiseParallel` (`torchtune/models/llama3/_parallelism.py:76-92`, fed by `convert_to_float8_training` having already swapped the modules). If you swap after shard, the parallel plan can't know what to do. (ii) For QAT, the state-dict keys for `FakeQuantizedLinear` are a superset of `nn.Linear` (the fake quantizer registers its own buffers), so `load_state_dict` would need to be tolerant; doing the swap first means `strict=True` works for the base weights.

### (c) Mid-training — the delayed QAT fake-quant toggle

Paths: `recipes/qat_single_device.py:532-550` (the live runtime toggle), `torchtune/training/quantization.py:56-71, 96-97, 126-127, 143-148, 159-164, 189-202` (the registry of `enable`/`disable` hooks keyed by quantizer mode string).

Once the model exists with `FakeQuantizedLinear` modules in it, `fake_quant_after_n_steps` walks the model and flips `mod.activation_fake_quantizer.enabled = False/True`. This is a per-step decision inside `train()`. Note: this path is wired in `qat_single_device.py` only — the distributed `qat_distributed.py` reads `cfg.get("fake_quant_after_n_steps", None)` into `self._fake_quant_after_n_steps` at line 220 but never consumes it in the training loop. Also note `_enable_linear_fake_quant` at `torchtune/training/quantization.py:63` references `FakeQuantizedLinear` *without importing it* — latent `NameError` if you ever actually hit the delayed-toggle path on the distributed recipe. Worth fixing as part of any refactor.

Why mid-training? The fake quantizer's calibration / scales benefit from a warmup on un-faked activations. That decision is fundamentally a function of `global_step`; it has to live in the loop.

### (d) Post-training — actual quantization for inference

Paths: `recipes/quantize.py`, particularly `_setup_model:69-86` and `quantize:88-95`.

If the saved checkpoint came from QAT, `quantize.py` re-instantiates the QAT-prepared model with `quantizer.prepare(model)` (because the checkpoint has fake-quant buffers in its state dict, so the model shape must match), loads the state dict, then calls `quantizer.convert(model)` — which replaces fake-quantized modules with real low-precision modules. If the checkpoint came from a normal full finetune, `quantizer.quantize(model)` is called instead (post-training quantization).

Why a separate recipe? Because actual quantization changes parameter dtypes and is destructive — you don't want to bake it into the training recipe; the same fine-tuned checkpoint may want to be deployed at different precisions.

### What would break if you moved one of these

- Moving (a) NF4 into recipe-time: the recipe would need to redo NF4-aware sharding logic (`_gather_nf4_tensor`, the NF4 branch of `load_from_full_model_state_dict` in `torchtune/training/_distributed.py:399-442`) at a different point. It's not impossible — it's just that the choice was "let the module carry its quantization identity end-to-end."
- Moving (b) FP8 swap into the builder: every model would need an FP8-aware factory; you'd also pre-commit to FP8 *before* the recipe gets to validate `cp_degree`, dtype, or TP plan compatibility (the recipe currently raises on `cp_degree > 1` at `recipes/full_finetune_distributed.py:623-626`).
- Moving (b) QAT swap *after* `load_from_full_model_state_dict`: state-dict keys would mismatch on resume, since the resumed checkpoint has `*.activation_fake_quantizer.*` buffers.
- Moving (c) delayed fake-quant into setup: you'd lose the step-conditional, which is the whole point.
- Moving (d) into the training recipe: you'd conflate the training-time model (which must remain in higher-precision compute for gradients) with the deployment-time model (which is destructively low-precision).

### Bonus: the **fifth** integration point you'll bump into

There's a TP-plan seam in `torchtune/models/llama3/_parallelism.py:76-126` and `base_llama_tp_plan(model, *, enable_fp8_training: bool)`. FP8 isn't just a module swap — it carries its own `Float8ColwiseParallel` / `Float8RowwiseParallel` tensor-parallel styles, and `_validate_float8_tp_plan` (`torchtune/training/quantization.py:277-302`) gates which fp8 recipe names are compatible with which TP styles. Similarly there's a per-step `precompute_float8_dynamic_scale_for_fsdp` call in the training loop at `recipes/full_finetune_distributed.py:1059-1067`. So FP8 actually touches **three** lifecycle phases by itself: pre-shard module swap, TP plan resolution, and post-optimizer-step scale precompute.

---

## 2. Where FP8 QAT has to plug in

If you want fake-FP8 weights during training and real FP8 at deployment, you need hooks in every one of (b), (c), (d), and likely the TP-plan seam:

1. **Module swap before shard.** A new `Float8QATQuantizer` (or similar) that swaps `nn.Linear` with a fake-FP8 linear. torchao's `Float8Linear` *already* does the equivalent of "fake quantize"-style scale + matmul; depending on the recipe name you choose (`tensorwise` / `rowwise` / `rowwise_with_gw_hp`), the existing `convert_to_float8_training` may already be "QAT-shaped" — i.e. weights stay master-precision, the forward casts down and back up. You should verify against the torchao version pinned by torchtune that the resulting trained weights are exactly the master-precision weights and whether the scale stats need to be carried into deployment. If the answer is "yes, `Float8Linear` already does what we want for the training half," then FP8 QAT training is almost a no-op (just call `convert_to_float8_training` from the right recipe). If you need a different scheme (e.g. *symmetric per-tensor FP8 weight fake-quant matching a specific inference kernel*), you'll need a sibling of `Int4WeightOnlyQATQuantizer` registered in `_quantizer_to_mode` at `torchtune/training/quantization.py:51, 94-97, 124-127`.

2. **Delayed fake-quant.** Plug into the same `_quantizer_mode_to_disable_fake_quant` / `_enable_fake_quant` registry. Note the bug I mentioned at `quantization.py:63` (missing `FakeQuantizedLinear` import) — fix this in the same PR. If you go the `convert_to_float8_training` route, you'll need a different toggle, because `Float8Linear` doesn't have `activation_fake_quantizer.enabled`; you'd toggle whether the forward casts at all. That's a different shape, which is part of the case for the redesign below.

3. **Post-training quantization to real FP8.** Add an `Float8WeightQuantizer` (or equivalent) so `recipes/quantize.py` can take an FP8-QAT checkpoint and convert weights to actual FP8 tensors using whatever torchao FP8 inference path you're targeting. Register it via `get_quantizer_mode`.

4. **LoRA composition.** This is where it gets ugly. The existing QAT+LoRA path runs through `swap_lora_linear_with_qat` (`torchtune/training/quantization.py:205-244`), which replaces `LoRALinear` with `QATLoRALinear` (`torchtune/modules/peft/lora.py:150-265`). `QATLoRALinear.forward` hardcodes the int-style structure: `_x = self.activation_fake_quantizer(x); w = self.weight_fake_quantizer(self.weight); out = F.linear(_x, w) + (alpha/rank) * B(A(x))`. It assumes a `FakeQuantizer` object with a `.enabled` switch. To support FP8 you'd either need (i) a parallel `FP8LoRALinear` class — yet another peft module — or (ii) make `QATLoRALinear` parameterizable over the fake-quant strategy. Note also that LoRA + FP8 may want a different transformation altogether: you might want the base matmul to be a `Float8Linear` op while the LoRA `BAx` stays in bf16 (LoRA deltas are too small to FP8 sensibly). That's a forward-graph decision, not a swap.

5. **Distributed plumbing.** `base_llama_tp_plan(..., enable_fp8_training=True)` (`torchtune/models/llama3/_parallelism.py:99-126`) and the `precompute_float8_dynamic_scale_for_fsdp` step are FP8-specific and already exist for full finetune. They'd need to carry over to whichever recipe you create. The `cp_degree > 1` and `is_fp8_tensorwise_scaling` constraints (`recipes/full_finetune_distributed.py:623-626`, `torchtune/training/quantization.py:286-302`) apply regardless.

6. **Checkpointer awareness.** `FullModelHFCheckpointer` etc. don't currently know about `Float8Linear`'s scale buffers. For training-time-only fake FP8, this is moot — `Float8Linear` keeps real weights and recomputes scales — so the saved state dict should look like a normal model. For *post-conversion* FP8 weights you'd be in `recipes/quantize.py` anyway, and the output format is whatever torchao saves. Worth confirming, though, that `gather_cpu_state_dict` (`torchtune/training/_distributed.py:486-517`) handles whatever subclass `Float8Linear` puts on its weight — the function only knows about `NF4Tensor` today and falls back to a generic DTensor gather otherwise.

### Required new recipes (assuming no redesign)

Given the no-inheritance rule:

- `recipes/fp8_qat_distributed.py` — copy of `qat_distributed.py` with the QAT prepare replaced by `convert_to_float8_training` and the per-step scale precompute added.
- `recipes/fp8_qat_single_device.py` — copy of `qat_single_device.py` minus the distributed-specific bits, but FP8 single-device is much less interesting (no FSDP all-gather scale, possibly no perf benefit on most consumer GPUs).
- `recipes/fp8_qat_lora_finetune_distributed.py` — copy of `qat_lora_finetune_distributed.py` with a new `swap_lora_linear_with_fp8_qat` helper.
- Configs under `recipes/configs/<model>/...` and registry entries in `torchtune/_recipe_registry.py:643-665`.

That's roughly 3 × 700–1100 LOC of near-duplicated recipe code, plus three new module/function siblings, plus three new entries in `_quantizer_to_mode`.

---

## 3. Should you propose a redesign first?

My honest read: **yes, a small one, before you write a fifth quantization path.** Not a giant overhaul — and the no-inheritance rule actually points you toward the right shape.

The current `_quantizer_to_mode` / `_quantizer_mode_to_disable_fake_quant` / `_quantizer_mode_to_enable_fake_quant` triad at `torchtune/training/quantization.py:51-53` is already *trying* to be a lifecycle-hook registry; it's just keyed by a string and only knows about two hooks (enable/disable fake quant). FP8 needs more hooks (post-step scale recompute, TP plan selection), so the registry needs to grow up. A reasonable shape:

```python
# torchtune/training/quantization.py
@dataclass
class QuantizationScheme:
    name: str  # "4w-qat", "8da4w-qat", "fp8-mixed", "fp8-qat", "nf4-frozen", ...

    # phase (a/b): module-replace before shard. None = no swap (e.g. NF4 lives in builder).
    prepare_modules: Optional[Callable[[nn.Module, "SchemeArgs"], nn.Module]] = None

    # phase (b) variant for LoRA composition. None = scheme not LoRA-compatible.
    prepare_lora_modules: Optional[Callable[[nn.Module, "SchemeArgs"], nn.Module]] = None

    # phase (c): per-step hooks. Either could be None.
    on_train_step_start: Optional[Callable[[nn.Module, int], None]] = None  # for delayed fake-quant
    on_optimizer_step_end: Optional[Callable[[nn.Module], None]] = None  # for fp8 scale precompute

    # distributed: optional override for the TP plan factory.
    tp_plan_factory: Optional[Callable[[nn.Module, dict], dict]] = None

    # phase (d): conversion to deployable representation.
    convert_for_inference: Optional[Callable[[nn.Module], nn.Module]] = None

    # capability flags — recipe validates these upfront instead of hand-coding "if fp8 and cp_degree > 1: raise"
    supports_context_parallel: bool = True
    supports_lora: bool = False
    requires_fsdp: bool = False
```

Each existing scheme registers a `QuantizationScheme` instance:

- `int4_weight_only_qat` fills `prepare_modules` (the existing `quantizer.prepare`), `on_train_step_start` (the delayed-toggle), `prepare_lora_modules` (the existing `swap_lora_linear_with_qat` shim), `convert_for_inference` (the existing `quantizer.convert`).
- `nf4_frozen` (QLoRA) fills nothing — its swap happens in the builder. The scheme exists mainly so the recipe knows to call `_gather_nf4_tensor` etc.
- `fp8_mixed_precision` fills `prepare_modules` (= `convert_to_float8_training`), `on_optimizer_step_end` (= `precompute_float8_dynamic_scale_for_fsdp`), `tp_plan_factory` (= `_get_fp8_llama_tp_training_plan`).
- `fp8_qat` (your new one) fills whichever subset matches the chosen fake-FP8 strategy.

Recipes — single-device and distributed alike — call a thin loop:

```python
scheme = config.instantiate(cfg.quantization_scheme)  # may be None
if scheme and scheme.prepare_modules:
    model = scheme.prepare_modules(model, args)
# ... TP / FSDP / load state dict ...
# in train loop:
if scheme and scheme.on_train_step_start:
    scheme.on_train_step_start(self._model, self.global_step)
# after optimizer.step():
if scheme and scheme.on_optimizer_step_end:
    scheme.on_optimizer_step_end(self._model)
```

**Why this respects the no-inheritance rule.** Nothing here is a base class. `QuantizationScheme` is a *value*, not a parent — recipes still own their lifecycle and remain copy-paste-modifiable. Each recipe still has its own model setup function; it just calls a couple of optional hooks at fixed phase boundaries. The protocol in `torchtune/recipe_interfaces.py` is unchanged. This is the same pattern `ParallelDims` follows: a composable descriptor object, not an inheritance hierarchy.

**What this buys you concretely for FP8 QAT.**

1. You write *one* `QuantizationScheme(name="fp8-qat", ...)` and register it.
2. The three recipes that need it (`full_finetune_distributed`, `lora_finetune_distributed`, `qat_distributed`-equivalent) get the hooks for free — you don't write `fp8_qat_distributed.py`. You write a small patch to `qat_distributed.py` so it accepts *any* scheme that has a `prepare_modules` hook, instead of hard-coding `quantizer.prepare`. The "qat" prefix on the recipe name becomes informational.
3. You delete `_validate_float8_tp_plan` and `is_fp8_tensorwise_scaling` as standalone helpers; their checks live as `supports_*` flags on the scheme.
4. The fifth scheme is additive, not a recipe fork.

**What this does *not* fix.** It doesn't unify the four phases — they're still phases, because they intrinsically have to happen at different points (the order is dictated by FSDP/TP, not by torchtune). What it does is make the *set* of phases a scheme participates in declarative, instead of being expressed as which recipe file you launched.

**Order of operations I'd suggest in your PR series.**

1. **PR 1 (refactor, no behavior change):** introduce `QuantizationScheme`, port the existing four schemes (NF4 / Int4-QAT / Int8DynAct-Int4-QAT / FP8-mixed) to it, leave `_quantizer_to_mode` as a thin compatibility shim. Fix the latent `FakeQuantizedLinear` import bug in `_enable_linear_fake_quant`. Wire one recipe (`qat_single_device` is the simplest) through the new hook protocol. CI green.
2. **PR 2 (wire distributed):** wire `full_finetune_distributed`, `qat_distributed`, `qat_lora_finetune_distributed` through the protocol. This is mostly s/`quantizer.prepare`/`scheme.prepare_modules`/.
3. **PR 3 (FP8 QAT):** register `fp8_qat` scheme + add configs. Should land as one scheme, optional new `Float8QATLoRALinear` if you decide LoRA composition needs a custom forward, and a `convert_for_inference` plug. No new recipe files.

If the reviewers reject the refactor — fair, it's a structural change — fall back to the three new recipe files I listed in section 2 and accept the duplication. But I'd float the refactor as an RFC issue first; the dominant cost in this codebase is the next person needing to add a sixth scheme, and the `_quantizer_to_mode`/`fake_quant_after_n_steps`/`precompute_float8_dynamic_scale_for_fsdp` triad is already a registry trying to escape, so it's a cheap moment to formalize it.

### Things I'm not 100% sure about

- Whether torchao's `Float8Linear` *as-is* gives you a deployable FP8 inference path or whether you need a separate `quantize.py` step to flatten scales into the saved weights. The torchao pin in `pyproject.toml` and a quick test against `recipes/quantize.py` will tell you.
- Whether FP8 + LoRA wants `BAx` in master precision (my instinct) or FP8 (probably bad for accuracy). This is a research question; the redesign at least makes it a clean swap to try both.
- Whether `gather_cpu_state_dict` correctly handles `Float8Linear`'s weight subclass under FSDP. The NF4 case has explicit handling at `torchtune/training/_distributed.py:399-442`; FP8 currently isn't exercised through that path because there's no LoRA-FP8 path today. You may need to add a similar branch.

### Key file references for your PR

- `torchtune/training/quantization.py` (will be the locus of the refactor)
- `torchtune/modules/peft/lora.py:150-300` (`QATLoRALinear` — model for any `FP8LoRALinear`)
- `torchtune/modules/low_precision/nf4_linear.py` (model for a builder-time fake-FP8 linear, if you want one)
- `recipes/qat_distributed.py:666-678` and `recipes/qat_single_device.py:375-389` (the current `prepare` placement)
- `recipes/qat_single_device.py:531-550` (the delayed fake-quant pattern)
- `recipes/full_finetune_distributed.py:617-627, 1059-1067` (FP8 hooks today)
- `recipes/qat_lora_finetune_distributed.py:450-466` (the QAT+LoRA composition)
- `recipes/quantize.py:69-95` (post-training convert path)
- `torchtune/models/llama3/_parallelism.py:76-126` and `torchtune/training/quantization.py:277-302` (TP plan & FP8 validation)
- `torchtune/training/_distributed.py:362-460, 486-517` (state-dict gather/load with NF4-aware branches — your model for FP8-aware branches if needed)
- `torchtune/recipe_interfaces.py` (the no-inheritance protocol, unchanged by the proposed design)
- `torchtune/_recipe_registry.py:616-665` (where new recipes/configs would register if you go the duplication route)
