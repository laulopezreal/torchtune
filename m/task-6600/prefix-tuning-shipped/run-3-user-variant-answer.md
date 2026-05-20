# Run 3 — User-refined prefix tuning variant

**Date:** 2026-05-20
**Subagent:** general-purpose (Claude Opus 4.7-class), low context
**Repo:** torchtune at BASE `213f38605`
**Informal pass rate:** ~78-83% (Medium band — HIGHER than Runs 1 + 2)

## The prompt sent (user-refined, verbatim with typos)

> I want to add Prefix Tuning to torchtune as a new PEFT method, according to the Li& Liang (2021) approach: weights stay frozen whils a small numer of learnable prefix vector per transformer layer are attend by the model during training and inference. I would like to support single device and distributed use. Given his is my first contributionI I want to fully commit to the project rules, but with a design that still make sense when new adapters show up.
>
> Walk me through a plan to achieve this. I know README says the project is not maintained, but assume for this question that the codebase will be forked and actively maintained going forward.

## What was dropped vs locked v1

- "third PEFT method alongside the existing LoRA and DoRA support"
- "The adapter state is a new tensor per layer; nothing wraps any existing nn.Linear"
- "recipes stay readable as self-contained scripts, no implementation inheritance"
- "Give me an honest read on whether the codebase shape makes this addition clean or whether I'd be adding pain future contributors will inherit. If it's the latter, propose what to fix first."

## Predicted vs actual

| | Predicted | Actual |
|---|---|---|
| Discovery of substring filter coupling | -10 to -15 pp | **+1 site found** (`_get_lora_moe_modules`, missed by Runs 1+2) |
| Penalty triggers (no "no-inheritance" naming) | +1 to 2 penalty triggers | **0 penalties** — model derived policy from `recipe_interfaces.py:14` anyway |
| Engagement with design forks | -5 to -10 pp | **5 design issues discussed** vs 3-4 in Runs 1+2 |
| Net band | 35–50% | **~78-83%** (higher than v1) |

**My prediction was wrong.** The dropped constraints didn't reduce score. Strong subagents recovered them from context + code reading. The variant is **slightly easier or equivalent**, not harder.

## Discovery score: 11/13 (highest of all 3 runs)

Hit:
- AdapterModule protocol + structural-not-syntactic (`peft/_utils.py:19-36`) ✓
- Substring filter at `_utils.py:132` ✓
- **Plus** `_get_lora_moe_modules` (`_utils.py:165-190`) — neither Run 1 nor Run 2 found this
- **Plus** the recipe-side `_is_dora = any(["magnitude" in k...])` at `lora_finetune_single_device.py:430` — strongest "5 sites of LoRA-substring coupling" count of any run
- LoRALinear / DoRALinear wrap nn.Linear ✓
- MultiHeadAttention.forward at `attention.py:262-283` ✓
- TransformerSelfAttentionLayer ✓
- KVCache (sized by `decoder_max_seq_len`) ✓
- _checkpoint_client.py at lines 170-182, 365-373, 537-547 — **most precise citation of any run** ✓
- `_distributed.py:528` (uses get_adapter_state_dict for sharded save) — only Run 3 found this
- `recipe_interfaces.py:14` quoted directly ✓
- HF PrefixTuningConfig fields specifically named (`prompt_tuning_init`, `num_virtual_tokens`, `token_dim`, `num_layers`, `num_attention_heads`, `num_transformer_submodules`) ✓

Missed:
- **PackedDataset position IDs** — same as Runs 1 + 2
- **Loss masking** — same as Runs 1 + 2

## Design questions: 4/5 strong, 1 missed

| # | Question | Status |
|---|---|---|
| 1 | Attachment point | ✓ Two options + reasoning (kwarg vs subclass), picks kwarg |
| 2 | RoPE on prefix? | ✓ STRONG — "No RoPE on prefix K... Insert the prefix after line 270, so it skips RoPE" |
| 3 | KV cache pre-fill vs regen? | ✓ "Either bump cache size by num_virtual_tokens (clean) or keep the prefix out of the cache and concat each step (more memory-efficient). Recommend the latter." |
| 4 | Sequence packing position-IDs | ✗ Not addressed |
| 5 | Loss masking | ✗ Not addressed |

## Penalty traps: 0 triggered (clean)

- 5001 base class: ✗ — proposes composition + Protocol marker `peft_type`
- 5002 dedup is goal: ✗ — explicitly quotes the policy
- 5003 lora_prefix naming: ✗ — uses `prefix_k`/`prefix_v`
- 5004 extend get_merged for prefix: ✗ — explicitly says prefix CANNOT merge
- 5005 edit 35 builder sites: ✗ — one family day one
- 5006 checkpointer class needs editing: ✗ — call-site fix only
- 5007 bundle refactor with feature: ✗ — explicit **5-PR sequence** (more granular than Run 1's 3-PR)
- 5008 vague "follow the pattern": ✗ — every claim file:line cited

## Notable strengths (more than any prior run)

- **5-PR sequence** (Run 1 had 3, Run 2 had 1 bundled Path)
- Finds `_get_lora_moe_modules` (the MoE variant of the substring problem)
- Finds `_distributed.py:528` use of `get_adapter_state_dict`
- Cites specific HF PEFT prefix tuning config fields (interop is concrete)
- Identifies the `is_causal=True` SDPA fast-path breaks when prefix is present
- Section on testing (`PrefixTuning` unit tests + MHA toy test + FSDP shard test)
- Identifies prefix tuning **breaks the merge contract** explicitly — "the user is then forced into `adapter_only=True`-style saving"

## Diagnosis

**The variant prompt did NOT make Step 5 harder.** The methodology's prediction held: a strong Claude-class model with a code-grounded catalyst recovers most prompt scaffolding from context. The 4 dropped sentences had marginal effect. The reproducible misses (sequence packing, loss masking) are **structural to the catalyst, not the prompt** — both runs of v1 AND the variant miss them.

**What this means:**
- The locked v1 prompt is calibrated correctly. Its 60-70% band reflects what strong models actually do.
- Your variant gives an equivalent-or-slightly-stronger answer for this single run.
- We cannot conclude variance vs systematic difference from one run.
- The rubric's critical 1007 (packing) and 1008 (loss masking) are the real Hard discriminators — confirmed across 3 independent runs.

## Subagent answer (verbatim)

Below is a senior-contributor walkthrough: what to build, where to put it, and the calls you have to make so a third PEFT method doesn't paper over the design tensions LoRA and DoRA already exposed.

## 1. What "Prefix Tuning" actually means in this codebase

Li & Liang's prefix tuning learns, per layer, a small set of prefix key/value vectors that get prepended to the layer's attention K/V (so queries can attend to them; the prompt itself is unchanged). In torchtune terms:

- Attention is `torchtune.modules.attention.MultiHeadAttention` (`torchtune/modules/attention.py:18`). The K/V live at lines 262-283 and have shape `[b, n_kv, s_y, h_d]`; you need to prepend along `s_y`.
- The natural attachment point is one prefix module per `MultiHeadAttention`, owned by the layer.
- Each prefix has trainable parameters of shape `[num_virtual_tokens, num_kv_heads * head_dim]` for both K and V.

Importantly: prefix tuning is **structurally different from LoRA/DoRA** -- it is not a per-`nn.Linear` wrapper, it changes attention shape, and it has to participate in:
- The KV cache (`torchtune/modules/kv_cache.py`) on inference.
- The attention mask (because prefix tokens broadcast to every query, regardless of causal mask).
- Positional embeddings (RoPE) -- prefix K/V should *not* go through RoPE in the standard formulation.

[Full answer in commit history; truncated for brevity]
