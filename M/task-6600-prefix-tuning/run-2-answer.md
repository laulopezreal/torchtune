# Run 2 — Prefix Tuning catalyst, prompt v1 (variance check)

**Date:** 2026-05-20
**Subagent:** general-purpose (Claude Opus 4.7-class), low context — same prompt as Run 1
**Repo:** torchtune at BASE `213f38605`
**Informal pass rate:** ~60-65% (Medium-Hard, slightly below Run 1)

## Discovery score: 9/13

Hit:
- AdapterModule protocol (`peft/_utils.py:19-36`) — quoted exactly
- adapter_key_filter substring at `_utils.py:132` — quoted lambda
- LoRALinear / DoRALinear wrap nn.Linear, prefix doesn't
- MultiHeadAttention.forward at `attention.py:262-303`
- TransformerSelfAttentionLayer used as attachment level
- KVCache at `kv_cache.py:54-116` — explicit
- _checkpoint_client.py call sites (170, 365, 537) — multiple cited
- convert_weights.py:240-309 tune_to_peft conversion
- DoRA's `to_empty` (dora.py:97-108) referenced as template for prefix's

Missed:
- **35-site adapter_cls if/else** — not surfaced (Run 1 listed 13 families partially)
- **RoPE on prefix tokens** — only "optionally hit pos_embeddings" in passing
- **PackedDataset position IDs** — not addressed
- **Loss masking** — not addressed
- generation/_generation.py — not explicitly cited (Run 1 had it)

## Design questions: 2/5 strong, 3 missed — SAME PATTERN AS RUN 1

| # | Question | Status |
|---|---|---|
| 1 | Attachment (TransformerSelfAttentionLayer vs MultiHeadAttention) | ✓ Strong — `prefix_kv` kwarg option vs subclass option |
| 2 | RoPE on prefix tokens? | ✗ Not addressed |
| 3 | KV cache: pre-fill or regen? | ✓ Strong — explicit option (a) eager prefill vs (b) per-step concat, with cost analysis |
| 4 | Sequence packing position-IDs | ✗ Not addressed |
| 5 | Loss masking | ✗ Not addressed |

## Penalty traps

- ✓ No base class proposal
- ⚠ **Path B borderline penalty 5008-style bundling** — proposes "refactor first, then ship cleanly" as a single Path B. Doesn't explicitly sequence as separate PRs the way Run 1 did. Could be interpreted as bundling refactor with feature.
- ✓ Respects copy-paste-modify policy explicitly (quotes recipe_interfaces.py:13-19)
- ✓ Does not claim checkpointer classes need editing — locates pain at `_checkpoint_client.py` call sites
- ✓ No naive "unify all"
- ✓ No vague "follow the pattern"

## Notable strengths

- **Path A vs Path B framing** at PR-strategy level (parallel to Run 1's attachment-point Options A/B/C)
- **"Near-miss interface" insight**: `adapter_params()` IS polymorphic, the rest isn't
- Names **6 places** that duplicate the LoRA-substring assumption (get_adapter_state_dict, _get_lora_modules, validate_*_for_lora, unconditional get_merged_lora_ckpt, _is_dora sniff, PEFT export)
- "Untouched, intentionally" section — shows scoping discipline
- Notes `validate_missing_and_unexpected_for_lora` has a deprecated path and proposes rename in same PR

## Cross-run convergence

| Item | Run 1 | Run 2 |
|---|---|---|
| AdapterModule protocol | ✓ | ✓ |
| adapter_key_filter substring | ✓ | ✓ |
| Attachment point analysis | A/B/C options | A/B options |
| KV cache pre-fill | ✓ (picks eager) | ✓ (offers both) |
| Checkpointer coupling | ✓ | ✓ (more sites cited) |
| PEFT export incompatibility | ✓ | ✓ |
| 35-site adapter_cls | partial | missed |
| **RoPE on prefix** | **missed** | **missed** |
| **Packing position IDs** | **missed** | **missed** |
| **Loss masking** | **missed** | **missed** |

**The three sequence-level discriminators are consistent misses across both runs.** This is the catalyst working as designed: strong answers go for "module fits abstraction" reasoning and miss "sequence semantics change" without prompting.

## Calibration verdict

- Both runs in 60-70% band on discovery + design
- Both runs miss the same 3 Hard-band discriminators
- Catalyst behavior is reproducible

**Recommendation:** lock prompt v1, move to rubric design with the sequence-level questions (RoPE / packing / loss masking) as critical-band criteria. A 3rd run is not needed for variance — we have it.

## Subagent answer (verbatim)

---

Short version: the PEFT plumbing in this repo is shaped almost entirely around "an adapter is a Linear-replacement that contains a `lora_a`/`lora_b`/`magnitude` weight." Prefix Tuning fits none of that. You can ship it cleanly, but only if you also do a small, well-scoped refactor of the adapter contract first. Otherwise you'll either string-match `"prefix"` into a half-dozen call sites or smuggle prefix tensors into the LoRA naming convention and leave a worse mess for the next contributor.

Below I lay out (1) the contracts you'll have to satisfy, (2) what Prefix Tuning specifically breaks, (3) two implementation paths (no-refactor vs. refactor-first), and (4) the small refactor I'd recommend doing first.

## 1. The current PEFT contract — what actually exists

The "interface" is informal but real. It is made of four cooperating pieces:

**a. The `AdapterModule` protocol** — `torchtune/modules/peft/_utils.py:19-36`. It's a `@runtime_checkable` `Protocol` with a single duck-typed method, `adapter_params(self) -> list[str]`. Each adapter-bearing `nn.Module` declares which of its own parameter names are trainable. `LoRALinear.adapter_params` returns `["lora_a.weight", "lora_b.weight"]`; `DoRALinear` returns those plus `"magnitude"` (`lora.py:116-126`, `dora.py:146-154`).

**b. Two whole-model scans that build on top of (a):**
- `get_adapter_params(model)` walks `model.named_modules()`, finds anything with a callable `adapter_params`, and returns the merged dict of trainable parameters (`_utils.py:39-65`).
- `set_trainable_params(model, adapter_params)` then sets `requires_grad` everywhere based on membership in that dict (`_utils.py:68-83`).

This pair is the only part of the PEFT contract that is actually polymorphic. It does **not** care whether your adapter is rank decomposition, magnitude vector, prefix tensor, or fourth thing. Good.

**c. The string-key filter** — `get_adapter_state_dict` (`_utils.py:115-133`):
```python
adapter_key_filter = lambda x: "lora" in x or "magnitude" in x
```
This is the canonical way the whole codebase separates "adapter state" from "base state." It is duplicated by intent in `validate_missing_and_unexpected_for_lora` (`_utils.py:367-369`) and in the merge utility `_get_lora_modules` (`_utils.py:136-162`). Every checkpointer save/load path goes through `get_adapter_state_dict`: `_checkpoint_client.py:170, 365, 537`. **Substring `"lora"` or `"magnitude"` is the de-facto adapter type system.**

**d. `get_merged_lora_ckpt`** (`_utils.py:193-271`). LoRA-specific math — `state_dict[f"{module}.weight"] += (alpha/rank) * lora_b @ lora_a`. The checkpoint client calls this unconditionally whenever an `adapter_config` is present (`_checkpoint_client.py:177-182, 370-372, 543-548`). It assumes the adapter is a residual rank-r perturbation of an existing `nn.Linear.weight`.

There are a few smaller LoRA-isms layered on:

- `LORA_ATTN_MODULES = Literal["q_proj", "k_proj", "v_proj", "output_proj"]` and `get_lora_module_names` (`_utils.py:16, 86-112`) — a LoRA-specific notion of "which sub-modules are wrapped."
- `validate_missing_and_unexpected_for_lora` (`_utils.py:315-417`) — still named "for_lora" but the new signature (`state_dict_keys=` path) is generic; the legacy path is deprecated. Good news for us.
- `tune_to_peft_adapter_weights` / `tune_to_peft_adapter_config` in `torchtune/models/convert_weights.py:240-309` — translate torchtune adapter state to HuggingFace PEFT's wire format. This is genuinely LoRA-only; PEFT itself has a separate `PrefixTuningConfig`/`PromptEncoder` family that isn't covered.

**e. Recipes own the configuration of all this.** Each LoRA recipe directly pokes config keys and stashes an `_adapter_config` dict (`recipes/lora_finetune_single_device.py:413-431`, `recipes/lora_finetune_distributed.py:492-523`):

```python
self._lora_rank   = cfg_model.lora_rank
self._lora_alpha  = cfg_model.lora_alpha
self._lora_attn_modules = list(cfg_model.lora_attn_modules)
self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
...
self._adapter_config = {"r": ..., "lora_alpha": ..., "target_modules": ..., "peft_type": "LORA"}

self.adapter_params = get_adapter_params(model)
self._is_dora = any(["magnitude" in k for k in self.adapter_params.keys()])
set_trainable_params(model, self.adapter_params)
```

And then in the distributed recipe (`lora_finetune_distributed.py:566-587`):
```python
for m in model.modules():
    if isinstance(m, AdapterModule) and not lora_weights_state_dict:
        m.to_empty(device=lora_device)
        m.initialize_parameters()
    if hasattr(m, "rope_init"):
        m.rope_init()
...
for m in model.modules():
    if hasattr(m, "initialize_dora_magnitude"):
        m.initialize_dora_magnitude()
```

That `hasattr(m, "initialize_dora_magnitude")` is the load-bearing hint of how the repo handles "this adapter needs a post-load initialization step." It's the same pattern DoRA quietly added.

Bottom line: there is one polymorphic seam (`adapter_params()`) and a pile of LoRA-string-matched seams (`"lora"`/`"magnitude"`, `get_merged_lora_ckpt`, PEFT export, recipe key wiring, `_adapter_config` shape). DoRA could ship because magnitude is still a *per-Linear* tensor that participates in a residual to `W`; Prefix Tuning is a different beast.

## 2. What Prefix Tuning specifically breaks

Prefix Tuning attaches `P_k, P_v ∈ R^{prefix_len × n_kv × head_dim}` per layer and concatenates them onto K/V at each attention call. No `nn.Linear` is wrapped. There is no residual to merge. The adapter lives next to (or inside) the attention module, not in place of a projection.

The mismatches in concrete order of how much pain they cause:

1. **Forward path.** Prefix tuning requires a hook *inside* attention to concatenate prefix K/V before SDPA. Look at `torchtune/modules/attention.py:262-303` — `k`/`v` are produced by `k_proj`/`v_proj`, optionally hit `pos_embeddings`, are transposed to `[b, n_kv, s, h_d]`, and go straight into `_attention_call`. There is no extension seam there. Adapter modules currently never see the K/V tensors. You'll need either to give `MultiHeadAttention` an optional `prefix_kv: Optional[nn.Module] = None` constructor arg (cleanest), or to subclass `MultiHeadAttention` (acceptable, mirrors how `LoRALinear` is a peer of `nn.Linear`). I'd push for the first: it's one new optional kwarg on one module and lets every model gain prefix tuning automatically via its `_component_builders` factory.

2. **KV cache.** This is the surprise gotcha. Inference goes through `KVCache.update` (`torchtune/modules/kv_cache.py:54-116`), which writes into pre-allocated buffers of shape `(batch, num_kv_heads, max_seq_len, head_dim)` and tracks `cache_pos`. Prefix vectors must be prepended on every forward, including cached forwards. Two clean options: (a) eagerly write the prefix into positions `[0, prefix_len)` of the cache at `setup_caches` time and shift `cache_pos` to start at `prefix_len`, or (b) concatenate prefix to the live K/V each step before SDPA, leaving the cache untouched. (a) is faster but requires teaching `setup_caches` about the prefix; (b) is simpler and self-contained but adds prefix_len work per token. Decide and document — neither is hard, but neither is free.

3. **The substring filter for adapter state.** `get_adapter_state_dict` ignores anything that doesn't contain `"lora"` or `"magnitude"` (`_utils.py:132`). If you name your parameters `prefix_k`/`prefix_v`, your trainable tensors will silently fall out of every save/load path and `get_adapter_state_dict` will return an empty-for-prefix dict. This bites at `_checkpoint_client.py:170, 365, 537`. The fix is in §4.

4. **`get_merged_lora_ckpt` is called unconditionally whenever `adapter_config` is set.** See `_checkpoint_client.py:177-182, 370-372, 543-548`. It iterates `_get_lora_modules` (which is at least scoped to keys containing `"lora"` or `"magnitude"`), so it won't *mangle* prefix tensors — but it requires `adapter_config["r"]` and `adapter_config["lora_alpha"]`, and the call happens whether you want it or not. For prefix tuning there is no meaningful "merge with base weights" (the prefix is part of the forward, not a delta on `W`), so the right answer is to skip this step entirely when the adapter type isn't a residual-on-weight kind.

5. **PEFT/HF export.** `tune_to_peft_adapter_weights` (`convert_weights.py:263-309`) is hardcoded to translate `lora_a`/`lora_b`/`magnitude` keys via `_TO_PEFT_KEYS` and to permute Q/K LoRA-B matrices for the HF rotary layout. None of that applies to prefix vectors. HF PEFT does have its own prefix-tuning serialization, but it's a different file format. For a first PR I'd raise/skip the PEFT export path for prefix-tuned models and save in the `.pt` "torchtune-format" branch (the checkpointer already takes that branch for Phi/Llama3.2-Vision/Llama4 at `_checkpointer.py:926-937`).

6. **Recipe wiring.** Every LoRA recipe hardcodes `cfg_model.lora_rank`, `cfg_model.lora_alpha`, `cfg_model.lora_attn_modules`, `cfg_model.apply_lora_to_mlp`, `cfg_model.apply_lora_to_output` (`lora_finetune_single_device.py:413-417` and the parallel block at `lora_finetune_distributed.py:492-496`). It then builds `_adapter_config = {"r":..., "lora_alpha":..., "target_modules":..., "peft_type":"LORA"}` and ships that dict through the entire checkpoint path. For prefix tuning, none of those keys make sense. You need either separate recipes (the recipe-duplication path the repo already endorses) or a more general adapter_config shape.

7. **`_is_dora` sniff.** `lora_finetune_single_device.py:430` does `any("magnitude" in k for k in self.adapter_params.keys())` to decide whether to call `initialize_dora_magnitude` afterwards. This is the existing pattern for "adapter needs post-load init." It's not a contract; it's a substring check. Prefix Tuning doesn't need post-load init from base weights, so you can leave this alone — but the smell is real.

8. **FSDP / `to_empty`.** Distributed model setup in `lora_finetune_distributed.py:566-577` iterates `isinstance(m, AdapterModule)` and calls `m.to_empty(device=lora_device); m.initialize_parameters()`. Both `LoRALinear.to_empty` and `DoRALinear.to_empty` have their own implementations (`lora.py:104-108`, `dora.py:97-108`) because they need to materialize internal `nn.Linear` submodules off meta device. Your `PrefixTuningAdapter` will be a single `nn.Parameter`-holder, so its `to_empty` is a one-liner — but you do need to write it, and you do need `initialize_parameters` (e.g. small Gaussian or zero-init). This is fine, it's the part of the design that actually works.

## 3. Two implementation paths

### Path A — "ship it now, leave a worse mess" (do not pick this)

The minimum path is:
- Add `torchtune/modules/peft/prefix.py` defining a `PrefixTuningAdapter(nn.Module, AdapterModule)` that holds `prefix_k, prefix_v` parameters and exposes `adapter_params() -> ["prefix_k", "prefix_v"]`.
- Add `prefix_kv` as an optional kwarg on `MultiHeadAttention.__init__` and call it inside `forward` before SDPA.
- Name the parameters `lora_prefix_k`/`lora_prefix_v` so the substring filter picks them up. *(Don't actually do this. This is exactly the kind of thing that creates archaeology debt for the next contributor.)*
- Special-case `get_merged_lora_ckpt` to no-op when `adapter_config["peft_type"] == "PREFIX"`.
- Write two new recipes: `prefix_finetune_single_device.py` and `prefix_finetune_distributed.py`, copy-pasted from the LoRA equivalents, with the `lora_*` config keys swapped for `prefix_*` and the `_adapter_config` dict reshaped.
- Add `prefix_llama3_8b` builder in each model family you care about, parallel to `lora_llama3_8b`.

This works. It costs you maybe 1,500–2,000 LOC across the two recipes and the model factories. The pain it leaves: the next contributor adding a fourth adapter (IA3, Llama-Adapter, prompt tuning) sees that `"lora"` is now sometimes-LoRA-sometimes-prefix-tuning and has to either pile on or do the cleanup you should have done.

### Path B — "small refactor first, then ship cleanly" (recommended)

The codebase has a near-miss interface (`AdapterModule.adapter_params`) that's already polymorphic. The refactor is to extend that interface with the two things the rest of the code actually needs:

1. **A typed `adapter_type` (or per-module `is_adapter_param` predicate)** so `get_adapter_state_dict` stops substring-matching `"lora"`.
2. **A separation between "adapter trainable parameters" and "post-training merge into base"** — because prefix tuning has the first but not the second.

Concretely (file paths are where the change lands):

- `torchtune/modules/peft/_utils.py:115-133` — replace the substring filter in `get_adapter_state_dict` with a walk over the model: collect every parameter name belonging to a module that satisfies the `AdapterModule` protocol, and use *that* set to filter the state dict. This is a 10-line change and is strictly more correct: it already matches what `get_adapter_params` does. The only reason the substring filter exists is that `get_adapter_state_dict` takes a `state_dict` (not a `model`) — so add an overload that takes the model, and keep the substring path as a fallback for now with a deprecation warning. Every call site (`_checkpoint_client.py:170, 365, 537`) is in a place where the model is in scope.
- `torchtune/modules/peft/_utils.py:193-271` — `get_merged_lora_ckpt` stays LoRA-specific. The right move is in the checkpoint client: gate the call on `adapter_config.get("peft_type") == "LORA"` (or equivalent). Compare to `_checkpoint_client.py:177-182` — today it's unconditional whenever `adapter_config is not None`. One conditional, three call sites.
- `torchtune/modules/peft/__init__.py` — re-export `PrefixTuningAdapter` so it's a first-class neighbor of `LoRALinear`/`DoRALinear`.
- `torchtune/training/checkpointing/_checkpointer.py:907-1002` — wrap the `tune_to_peft_adapter_weights`/`tune_to_peft_adapter_config` block in a check on `adapter_config["peft_type"]`. For PREFIX, take the same "torchtune .pt format" branch already used for unsupported model types (`_checkpointer.py:926-937`). Document the limitation explicitly.
- `torchtune/modules/attention.py` — add `prefix_kv: Optional[nn.Module] = None` kwarg to `MultiHeadAttention`. Inside `forward`, between line 282 (KV cache update) and 287 (GQA expand), if `self.prefix_kv is not None`, ask it for `(p_k, p_v)` and prepend along the sequence dim. Make sure the attention mask gets prefix_len columns of `True` prepended on the left. This is one module, ~15 lines, and it's an additive optional kwarg — every existing call site continues to work.
- `torchtune/modules/peft/prefix.py` (new) — `PrefixTuningAdapter(nn.Module, AdapterModule)` storing `prefix_k, prefix_v` of shape `[prefix_len, num_kv_heads, head_dim]`. Implement `adapter_params`, `initialize_parameters` (zero or small Gaussian; the Li & Liang reparameterization trick with a small MLP is optional and can be a v2), and `to_empty`. Optional `disabled` flag mirroring `LoRALinear.disabled` to play nicely with `disable_adapter` (`_utils.py:274-312`).
- `recipes/prefix_finetune_single_device.py` and `recipes/prefix_finetune_distributed.py` (new). Yes, they will duplicate ~700 LOC each from the LoRA recipes. That is the codebase's stated style — `recipe_interfaces.py:13-19` literally says: *"torchtune strictly prohibits implementation inheritance in the codebase. Minimizing code duplication is not the goal."* Make peace with the copy. The diff against `lora_finetune_*.py` should be:
  - Read `cfg_model.prefix_len` instead of `cfg_model.lora_rank/alpha/...`.
  - Build `_adapter_config = {"peft_type": "PREFIX", "prefix_len": ..., "target_layers": [...]}`.
  - Drop the `_is_dora` branch.
  - Keep `get_adapter_params` + `set_trainable_params` (those are polymorphic; they just work).
  - Keep `validate_missing_and_unexpected_for_lora` — the `state_dict_keys=` path (`_utils.py:366-390`) doesn't actually care about LoRA semantics, it just diffs against the model's state dict keys. The function name is misleading; rename it `validate_missing_and_unexpected_adapter` in the same PR (the LoRA name is already deprecated for callers, see `@deprecate_parameter` decorators at `_utils.py:315-323`).
- One model family to start (recommend `llama3`, since it has the most-exercised configs). Add `prefix_llama3_8b` to `_model_builders.py` and a `prefix_llama3` to `_component_builders.py`. The pattern from `lora_llama3` at `_component_builders.py:154-297` is your template: loop over layers, attach a `PrefixTuningAdapter` to each `TransformerSelfAttentionLayer.attn` by passing `prefix_kv=...` into `MultiHeadAttention`. Plus one YAML in `recipes/configs/llama3/8B_prefix.yaml`. This matches how DoRA was rolled out: one model first, others follow as separate PRs.

Cost of Path B vs Path A: maybe 200–400 extra LOC in shared code (the refactor) and one *fewer* model family touched in v1, in exchange for the next contributor being able to add IA3 by writing exactly one new module file plus their recipes, with zero changes to the checkpointer or shared utils.

## 4. The honest read

> "Does the codebase shape make this clean, or am I about to add pain future contributors will inherit?"

It's the second one, *if* you go Path A. The PEFT contract here was designed for "LoRA, plus things that look like LoRA." DoRA fits because its delta is still `W + (BA × magnitude)` — that's why `get_merged_lora_ckpt` could be extended in-place (`_utils.py:243-258`) rather than refactored. Prefix tuning is the first adapter that doesn't fit the residual-on-Linear shape. You will be the contributor who discovers this. The Path-B refactor is small (under a day's work for someone familiar with the repo) and it is the right thing to do exactly because the next not-LoRA-shaped adapter is the test of whether torchtune's "no implementation inheritance, copy-paste recipes" philosophy can scale.

The philosophy is fine for *recipes* — those are training loops and they really do drift along orthogonal axes (DPO, QAT, KD, multi-dataset). It is *not* fine for the adapter contract, which is genuinely shared infrastructure. Today, six separate places (`get_adapter_state_dict`, `_get_lora_modules`, `validate_missing_and_unexpected_for_lora`, the unconditional `get_merged_lora_ckpt` call, `_is_dora` sniffing, PEFT export) all duplicate the same implicit assumption — "an adapter param has `lora` or `magnitude` in its name." That's not the no-inheritance principle in action; that's a missing abstraction.

The five concrete refactor points (substring filter → polymorphic walk; `get_merged_lora_ckpt` conditional on adapter type; `MultiHeadAttention` gets one optional `prefix_kv` kwarg; PEFT export gated; rename `validate_*_for_lora`) keep recipes as self-contained scripts and don't introduce any base classes. They just make `AdapterModule` carry its own weight as the polymorphism boundary, instead of having callers string-match its conventions. That's the smallest possible change consistent with the project's design rules.

One last note: the README's "wound down" caveat means you won't have upstream review on the boundary calls. If you're forking, this is actually the right moment to do the refactor — there's no upstream to merge-conflict against, and you'll be in much better shape when IA3, Llama-Adapter, or prompt tuning shows up next.

### Files you'll touch (path B), in rough order

New:
- `/home/user/torchtune/torchtune/modules/peft/prefix.py`
- `/home/user/torchtune/recipes/prefix_finetune_single_device.py`
- `/home/user/torchtune/recipes/prefix_finetune_distributed.py`
- `/home/user/torchtune/recipes/configs/llama3/8B_prefix.yaml` (and `8B_prefix_single_device.yaml`)
- tests under `/home/user/torchtune/tests/torchtune/modules/peft/` mirroring the existing LoRA/DoRA tests

Modified (small, surgical):
- `/home/user/torchtune/torchtune/modules/peft/_utils.py` — generalize `get_adapter_state_dict`; rename `validate_missing_and_unexpected_for_lora` (keep alias).
- `/home/user/torchtune/torchtune/modules/peft/__init__.py` — re-export.
- `/home/user/torchtune/torchtune/modules/attention.py` — add `prefix_kv` kwarg + call site.
- `/home/user/torchtune/torchtune/modules/kv_cache.py` — only if you choose option (a) for caching (write prefix into cache at `setup_caches`).
- `/home/user/torchtune/torchtune/training/checkpointing/_checkpoint_client.py` — three call sites gating `get_merged_lora_ckpt` on adapter type.
- `/home/user/torchtune/torchtune/training/checkpointing/_checkpointer.py` — gate the PEFT export block on adapter type.
- `/home/user/torchtune/torchtune/models/llama3/_component_builders.py` and `_model_builders.py` — add `prefix_llama3` / `prefix_llama3_8b`.

Untouched, intentionally:
- `recipes/lora_finetune_*.py` — leave them alone. Path B's whole point is that the LoRA recipes don't need to know prefix tuning exists, because all the shared-infra changes are backwards compatible.
- `torchtune/models/convert_weights.py` — until someone wants HF PEFT prefix-format export, raise/skip is fine.
