# Run 1 — Prefix Tuning catalyst, prompt v1

**Date:** 2026-05-20
**Subagent:** general-purpose (Claude Opus 4.7-class), low context
**Repo:** torchtune at BASE `213f38605`
**Informal pass rate:** ~65-70% (Medium-Hard band)

## Discovery score: 10/13

Hit:
- AdapterModule protocol (`peft/_utils.py:19-36`) — quoted exactly
- adapter_key_filter substring at `_utils.py:132` — quoted lambda
- 35-site adapter_cls if/else — partial; lists the 13 model families but doesn't quote pattern
- LoRALinear / DoRALinear wrap nn.Linear, prefix doesn't
- MultiHeadAttention.forward at `attention.py:262-292` — K/V never escape, no hook
- TransformerSelfAttentionLayer used as attachment level
- KVCache shape `[bsz, n_kv_heads, max_seq_len, head_dim]` — quoted
- _checkpoint_client.py:177-181 merge-on-save coupling
- convert_weights.py:240-309 tune_to_peft conversion
- generation/_generation.py for prefix-at-decode-step

Missed:
- **RoPE on prefix tokens** — entire question not addressed
- **PackedDataset position IDs** — packing mentioned only re mask shape, not the per-sample-position-reset question
- **Loss masking** — not addressed

## Design questions: 2/5 strong, 1 partial, 2 missed

| # | Question | Status |
|---|---|---|
| 1 | Attachment (TransformerSelfAttentionLayer vs MultiHeadAttention) | ✓ Strong — three options A/B/C with explicit tradeoffs; picks C (`kv_augment` hook) |
| 2 | RoPE on prefix tokens? At what positions? | ✗ Not addressed |
| 3 | KV cache: pre-fill or regen? | ✓ Pre-fill at `setup_cache` time |
| 4 | Sequence packing | Partial — "disable packing for v1" but on mask shape, not position IDs |
| 5 | Loss masking | ✗ Not addressed |

## Penalty traps avoided

- ✓ No base class proposal (Protocol + composition)
- ✓ PR sequence separates refactor (PR 1) from feature (PR 3) — does not bundle
- ✓ Explicitly respects "Minimizing code duplication is not the goal"
- ✓ Does not claim checkpointer classes need editing — locates pain at `get_merged_lora_ckpt` + `tune_to_peft_*`
- ✓ No naive "unify all" — careful and discriminating
- ✓ No vague "follow the pattern" — every claim file:line cited

## Notable strengths

- **Option A / B / C analysis** for attachment point — exemplary Hard-band reasoning
- **PR sequence**: refactor → hook → feature, each independently reviewable
- Flags FSDP × small per-layer parameter uncertainty honestly
- Names HF PEFT's actual `PrefixTuningConfig` (peft_type: "PREFIX_TUNING", num_virtual_tokens)
- Concrete `PrefixAdapter` code skeleton

## Diagnosis

Strong on PEFT-abstraction reasoning. Weak on sequence-structure implications. The three misses (RoPE, packing positions, loss masking) confirm the verifier's hypothesis that strong answers go for "module fits the abstraction" reasoning and miss "sequence semantics change" reasoning unless the prompt forces them there.

This is **good news for rubric design**: RoPE / packing / loss-masking will be the discriminators between Medium-band and Hard-band answers.

## Subagent answer (verbatim)

---

This is a useful exercise because Prefix Tuning is the first PEFT method that genuinely doesn't fit the existing abstraction. Walking through it will tell you whether the seams are clean or whether you're papering over a real issue. Short answer: **the codebase shape will fight you**, the LoRA assumptions are baked into the adapter system at several layers, and you should fix one of them before landing prefix tuning rather than after. I'll walk through what implementing it actually requires, then say what to fix first.

## 1. Where prefix tuning collides with the current abstraction

### The `AdapterModule` protocol is module-shaped, prefix tuning is layer-shaped

The PEFT contract today is in `torchtune/modules/peft/_utils.py:19-36`:

```python
@runtime_checkable
class AdapterModule(Protocol):
    def adapter_params(self) -> list[str]: ...
```

This works because LoRA/DoRA wrap a single `nn.Linear` (`LoRALinear` at `torchtune/modules/peft/lora.py:27`, `DoRALinear` at `torchtune/modules/peft/dora.py:21`). The adapter *is* an `nn.Module` swapped in for a `q_proj`/`v_proj` etc. inside `MultiHeadAttention` (`torchtune/modules/attention.py:83-137` takes them as constructor args). `get_adapter_params` (`_utils.py:39-65`) walks `named_modules()`, finds anything that responds to `adapter_params()`, and pulls those parameters out by name.

Prefix tuning isn't shaped that way. It's a `[num_layers, prefix_len, num_kv_heads, head_dim]` tensor (or one tensor per layer) that gets *prepended to the K and V sequences* inside attention — it doesn't replace any linear. So the cleanest fit is one `PrefixAdapter(nn.Module)` per `TransformerSelfAttentionLayer`, holding learnable `k_prefix`/`v_prefix` parameters, registered as a sibling of `attn` on the layer.

That part is fine — `AdapterModule` is a duck-typed protocol, so as long as your new module exposes `adapter_params()` returning `["k_prefix", "v_prefix"]`, `get_adapter_params` and `set_trainable_params` (`_utils.py:68-83`) will pick it up correctly with **zero changes** in `_utils.py`. Good.

### `MultiHeadAttention` has no insertion point

The real wall is `torchtune/modules/attention.py:181-303`. `forward(x, y, mask, input_pos)` computes `k = self.k_proj(y); v = self.v_proj(y)` (lines 262-263), then immediately consumes them in `self._attention_call(q, k, v, ...)` (line 292). K and V never escape this function. There's no hook for "concatenate something to K/V before SDPA."

Three options, ranked by how much pain you push onto future contributors:

**Option A — fork MultiHeadAttention (bad).** Make `PrefixMultiHeadAttention(nn.Module)` that copies `MultiHeadAttention` and concats prefix tensors before `_attention_call`. Pros: no upstream API change. Cons: this is exactly the "850–1170 LOC of near-identical boilerplate" problem the codebase already has at the recipe level, now metastasizing into the modules layer. Every change to base attention (new mask kind, new pos embedding wrinkle, flex-attention plumbing) now happens in two places. You'd also need a `prefix_llama3_self_attention` parallel to `lora_llama3_self_attention` (`torchtune/models/llama3/_component_builders.py:300`) in every one of the **13 model families** that have a `lora_*` builder today (`gemma`, `gemma2`, `llama2`, `llama3`, `llama3_1`, `llama3_2`, `llama3_2_vision`, `llama4`, `mistral`, `phi3`, `qwen2`, `qwen3`, `clip`). That's where the pain really compounds.

**Option B — pre/post hooks on MultiHeadAttention.** Don't touch attention's source; the prefix module registers a `forward_pre_hook` on `attn` that swaps in a wrapped `k_proj`/`v_proj` that concats the prefix to its output, plus a hook that extends `mask` along the key axis. This works but is fragile: it depends on argument names, doesn't compose with flex attention's `BlockMask` (`attention_utils.py:185-252` — flex requires a BlockMask object, you can't just pad it), and breaks the KV cache path because `kv_cache.update` (`kv_cache.py:54-116`) does positional bookkeeping that assumes K and V are the "real" sequence.

**Option C — small, surgical change to MultiHeadAttention (best).** Add an optional `kv_augment: Optional[nn.Module] = None` ctor arg to `MultiHeadAttention.__init__`. After K/V are computed and shaped to `[b, n_kv, s, h_d]` (around `attention.py:273-274`), call `k, v, mask = self.kv_augment(k, v, mask)` if present, with a no-op default. The augment module owns its own state and the mask-extension logic. Prefix tuning becomes one `KVPrefixAugment` implementation; future adapters (P-Tuning v2, Llama-Adapter v2 gating, soft prompts injected as KV) plug in the same way. This is ~5 lines in `attention.py` and gives you a real seam.

The cost of C is one upstream change in a hot path that everyone already uses; the cost of A is replicating ~70 lines of `MultiHeadAttention` × 13 model families forever. C is the right call.

### KV cache + flex attention + GQA expansion are all in the way

Whichever option you pick, you have to handle all three of these:

1. **KV cache** (`torchtune/modules/kv_cache.py`). The cache is `[bsz, n_kv_heads, max_seq_len, head_dim]` (line 32). Prefix vectors should occupy positions before the prompt and never be evicted. Simplest correct design: bake the prefix into the cache once at `setup_cache` time, treat positions `[0, prefix_len)` as preloaded, and have `input_pos`/`cache_pos` start from `prefix_len`. This requires extending `KVCache.__init__` or adding a `set_prefix(k, v)` method. The `cache_pos` arithmetic (`kv_cache.py:104-114`) needs to be aware of the offset.

2. **Mask shape**. SDPA receives `mask[:, None, :, :]` of shape `[b, 1, s_q, s_kv]` (`attention_utils.py:196-211`). When you prepend `prefix_len` K/V tokens, every causal/packed mask needs `prefix_len` extra columns of `True` on the left. For boolean masks: an `F.pad(mask, (prefix_len, 0), value=True)`. For `BlockMask` / flex attention (`attention_utils.py:217-250`): you cannot easily mutate a `BlockMask` — you'd need to construct it including the prefix region from the start, which means the prefix length leaks into `packed_block_causal_mask` (`attention_utils.py:133`). Honest recommendation for v1: **disable sample packing when prefix tuning is on**, error clearly, and revisit flex support in v2. This is the same kind of compromise QAT-LoRA makes.

3. **GQA expansion** (`attention.py:287-290`). Prefix must be in `[b, n_kv_heads, prefix_len, head_dim]` shape, concatenated *before* the `expand` to `n_heads`, otherwise you waste memory by a factor of `q_per_kv`.

### Checkpointer is hardcoded to LoRA's state-dict keys

This is the design tension that bites hardest. The "adapter" plumbing is named generically (`ADAPTER_KEY = "adapter"`, `training/checkpointing/_utils.py:57`) but the filtering is keyword-based on `"lora"` and `"magnitude"`:

- `get_adapter_state_dict` (`_utils.py:115-133`): `adapter_key_filter = lambda x: "lora" in x or "magnitude" in x`. **Prefix params get silently dropped.**
- `_get_lora_modules` (`_utils.py:136-162`) and `get_merged_lora_ckpt` (`_utils.py:193-271`): same string-match.
- `validate_missing_and_unexpected_for_lora` (`_utils.py:324-417`): the "is this a lora key" check is the same.

There's exactly one well-named seam (`get_adapter_state_dict`) and its implementation is hardcoded to LoRA. That's the bug to fix before adding a third adapter type, not after. Otherwise every adapter type is going to keep editing this function and adding `or "prefix" in x or "<next-thing>" in x` until it's unmaintainable.

The checkpointer itself is mostly fine — `CheckpointClient` (`training/checkpointing/_checkpoint_client.py`) and `FullModelHFCheckpointer` (`training/checkpointing/_checkpointer.py:907-975`) handle `ADAPTER_KEY` generically when saving. The pain points are:

- `get_merged_lora_ckpt` is called on save in `_checkpoint_client.py:177-181` and `_checkpointer.py` line 543-546. For prefix tuning there's no "merge" — prefix vectors can't be folded into base weights. The recipe must skip the merge when the adapter type is prefix. Today there's no clean signal for "this adapter type is unmergeable."
- `tune_to_peft_adapter_weights` (`torchtune/models/convert_weights.py:263-309`) and `tune_to_peft_adapter_config` (line 240-260) assume LoRA-shaped keys (`lora_A`, `lora_B`, magnitude) and `peft_type: "LORA"`. HF PEFT does have a `PrefixTuningConfig` (`peft_type: "PREFIX_TUNING"`) with `num_virtual_tokens`, so this is implementable, but you'll need a parallel conversion path keyed off the adapter type.

### Recipes — the "copy-paste-modify" rule

`recipe_interfaces.py:13-19` is explicit: *"torchtune strictly prohibits implementation inheritance"* and *"Minimizing code duplication is not the goal."* So the contribution rule says: write `prefix_finetune_single_device.py` (~750 LOC) and `prefix_finetune_distributed.py` (~985 LOC), copied from `lora_finetune_*.py` and modified.

What modifications are actually needed inside the recipe? Look at `recipes/lora_finetune_distributed.py:492-598`:

- Lines 492-506: build `_adapter_config` with `peft_type: "LORA"`, rank/alpha, target_modules. For prefix you'd build something with `peft_type: "PREFIX_TUNING"`, `num_virtual_tokens`, and a list of layer indices the prefix is on.
- Line 523: `set_trainable_params(model, get_adapter_params(model))`. This is the one line that **already generalizes** because of the protocol. Good.
- Lines 566-577: special-case for `AdapterModule.to_empty()` + `m.initialize_parameters()`. Your prefix module just needs to expose those methods and it works as-is.
- Lines 585-587: `initialize_dora_magnitude()` is a per-adapter-type init step done by `hasattr` sniffing. For prefix you might need a similar init (e.g., re-encode some real tokens through the embedding to seed the prefix, à la Li & Liang's reparameterization). Same pattern — add a `_prefix_init` method, sniff by `hasattr`.
- Lines 589-598: `validate_missing_and_unexpected_for_lora` — needs a non-LoRA-specific variant.

The interesting question: of those ~750 lines of recipe, how many actually differ between LoRA and Prefix? Probably **30**. The protocol-driven seam at line 523 is doing the right thing; the rest is essentially identical and the codebase tells you to copy-paste it anyway. I'd respect the rule, write the duplicated recipes, and **leave the duplication issue alone** — it's a deliberate codebase-wide choice, and fighting it as a first contribution is the wrong battle.

Where you should *not* duplicate is the model builder. You'll need a `prefix_llama3()` in `torchtune/models/llama3/_component_builders.py` paralleling `lora_llama3()` (`_component_builders.py:154`) and the per-size factories like `prefix_llama3_8b` in `_model_builders.py`. This is consistent with the existing pattern and is the price of admission for any new adapter family.

## 2. What I'd actually build

Concretely, the diff is:

**New file: `torchtune/modules/peft/prefix.py`**

```python
class PrefixAdapter(nn.Module, AdapterModule):
    def __init__(self, num_kv_heads, head_dim, prefix_len, dtype=...):
        super().__init__()
        self.prefix_len = prefix_len
        self.k_prefix = nn.Parameter(torch.empty(num_kv_heads, prefix_len, head_dim))
        self.v_prefix = nn.Parameter(torch.empty(num_kv_heads, prefix_len, head_dim))
        self.disabled = False
        self.initialize_parameters()

    def adapter_params(self):  # protocol
        return ["k_prefix", "v_prefix"]

    def initialize_parameters(self):
        nn.init.normal_(self.k_prefix, std=0.02)
        nn.init.normal_(self.v_prefix, std=0.02)

    def forward(self, k, v, mask):
        # k,v: [b, n_kv, s, h_d]; concat prefix on dim=2; pad mask on key axis.
        if self.disabled:
            return k, v, mask
        b = k.shape[0]
        kp = self.k_prefix.to(k.dtype).unsqueeze(0).expand(b, -1, -1, -1)
        vp = self.v_prefix.to(v.dtype).unsqueeze(0).expand(b, -1, -1, -1)
        k = torch.cat([kp, k], dim=2)
        v = torch.cat([vp, v], dim=2)
        if mask is not None and not isinstance(mask, BlockMask):
            mask = F.pad(mask, (self.prefix_len, 0), value=True)
        return k, v, mask
```

**Small change: `torchtune/modules/attention.py`** — add `kv_augment: Optional[nn.Module] = None` to `__init__`, store it, call `k, v, mask = self.kv_augment(k, v, mask)` after the GQA-shaped K/V are computed and before `_attention_call`. Raise on `BlockMask` + augment for now. This is the one cross-cutting change you should propose as a separate PR and get reviewed first.

**Refactor (the precondition):** generalize the "is this an adapter key" predicate in `torchtune/modules/peft/_utils.py`. Replace the string match in `get_adapter_state_dict` (`_utils.py:132`), `_get_lora_modules` (`_utils.py:150-154`), and the `is_lora_key` lambda in `validate_missing_and_unexpected_for_lora` (`_utils.py:367`) with a single source of truth — either a class-level `ADAPTER_KEY_TOKENS = {"lora", "magnitude"}` registry that adapter modules contribute to, or (cleaner) a marker — every parameter belonging to an `AdapterModule` is identified by walking modules, not by string matching the state-dict key. The protocol already gives you what you need (`get_adapter_params` works that way at line 54-61); just use the same approach for `get_adapter_state_dict`. Then prefix tuning needs zero changes there.

Also separate "save adapter weights" from "merge adapter into base weights." Today `_checkpoint_client.py:177-181` always calls `get_merged_lora_ckpt`. Make that a property of the adapter config (`mergeable: bool`, or a registered merge function). Without this, the prefix recipe will need to special-case the save path, which is exactly the pattern this refactor would let you avoid.

**New model builders:** `prefix_llama3()` in the component builders (calls the regular `llama3()` but threads a `PrefixAdapter` into each layer's `MultiHeadAttention.kv_augment`), `prefix_llama3_8b()` factory, YAML at `recipes/configs/llama3/8B_prefix.yaml`. Same pattern for whatever other families you want to support out of the gate — I'd ship one family in the first PR.

**New PEFT conversion in `convert_weights.py`:** `tune_to_peft_prefix_config` and `tune_to_peft_prefix_weights`, mapping to HF's `PrefixTuningConfig` schema. Then in `_checkpointer.py:944-952` dispatch on `adapter_config["peft_type"]` instead of unconditionally calling `tune_to_peft_adapter_weights`.

**Recipes:** `recipes/prefix_finetune_single_device.py` and `recipes/prefix_finetune_distributed.py`, copy-pasted from the LoRA ones with the model-specific diffs above (adapter config, no merge on save, KV cache offset wiring).

**KV cache for inference:** `KVCache.set_prefix(k_prefix, v_prefix)` that writes prefix vectors into positions `[0, prefix_len)` and starts `cache_pos` at `prefix_len`. The generation loop in `torchtune/generation/_generation.py` needs to call this once after `setup_caches`.

## 3. Honest read on the design

The first PEFT method gets the architecture; the second method tests it; the third method exposes whether it's actually an abstraction or just two copies of the same thing. Right now torchtune is in the "two copies of the same thing" state — DoRA is shaped almost exactly like LoRA (same `lora_a`, `lora_b`, plus a `magnitude` scalar), and the adapter system was built around that shape. The `AdapterModule` *protocol* is genuinely extensible (the `adapter_params()` contract is clean and prefix tuning works through it unmodified). The *checkpointer integration* is not — it's three string-match call sites that grew up assuming LoRA-shaped keys, and the merge-on-save is hardcoded into the checkpoint client path.

If you land prefix tuning without fixing those, you're adding the third special-case to a system that already needed a refactor. If you fix `get_adapter_state_dict` / `validate_missing_and_unexpected` / the merge-on-save dispatch first (probably one ~150-line PR), prefix tuning becomes a clean addition that future adapter types ride for free.

**Order of operations I'd recommend, especially for a first contribution:**

1. **PR 1:** Refactor `peft/_utils.py` to use module-walk instead of string-match for adapter state dict and validation. Add `mergeable: bool` (or a merge-fn registry) to the adapter config flow in `_checkpoint_client.py` and `_checkpointer.py`. No behavior change for LoRA/DoRA — pure refactor with existing tests as the safety net. Easy to review, makes the rest possible.
2. **PR 2:** Add `kv_augment` hook to `MultiHeadAttention`. Small surface area, also no behavior change by default.
3. **PR 3:** `PrefixAdapter` module + `prefix_llama3` builder + one config + the two recipes + KV cache prefix support + tests. Now the only LoRA-shaped assumption left is `tune_to_peft_*`, which you dispatch on `peft_type`.

Each PR is reviewable on its own and doesn't require the reviewer to hold the whole picture in their head. This is also the order that minimizes blast radius for the rest of the codebase if something needs to be reverted.

One uncertainty worth flagging: I haven't verified what happens with FSDP + a per-layer parameter that's small relative to the layer (the prefix tensor is `prefix_len * num_kv_heads * head_dim` per layer — small). The current sharding policy in `recipes/lora_finetune_distributed.py:528-554` shards on `TransformerSelfAttentionLayer` boundaries, so the prefix lives inside an FSDP unit and should be sharded with the layer it belongs to. That's fine in principle but I'd test it explicitly — small parameters inside large FSDP units sometimes interact oddly with `to_empty` + meta-init paths. Look at how DoRA handles this in `DoRALinear.to_empty` (`dora.py:97-108`) for a template; you'll likely need an analogous `to_empty` override.

### Key file references

- Adapter protocol: `torchtune/modules/peft/_utils.py:19-36`
- LoRA module: `torchtune/modules/peft/lora.py:27-147`
- DoRA module: `torchtune/modules/peft/dora.py:21-192`
- The string-match problem (3 places): `torchtune/modules/peft/_utils.py:132`, `_utils.py:150-154`, `_utils.py:367`
- Merge-on-save coupling: `torchtune/training/checkpointing/_checkpoint_client.py:177-181`, `_checkpoint_client.py:543-546`, `_checkpointer.py:944-952`
- HF/PEFT conversion: `torchtune/models/convert_weights.py:240-309`
- Attention insertion point: `torchtune/modules/attention.py:262-292`
- KV cache: `torchtune/modules/kv_cache.py:11-116`
- Mask handling: `torchtune/modules/attention_utils.py:185-252`
- Recipe template to copy: `recipes/lora_finetune_distributed.py:472-598` (the model-setup block is where ~30 lines genuinely differ)
- "No inheritance, copy-paste-modify is the rule": `torchtune/recipe_interfaces.py:13-19`
