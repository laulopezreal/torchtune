# Prefix Tuning catalyst — code-grounded verification

**Date:** 2026-05-20
**Verifier:** Explore subagent (low context, no CLAUDE.md / no M/)
**Verdict:** Strong Hard catalyst, bordering on Very Hard. Proceed.

## Surface map (10+ files a strong answer must touch)

1. `torchtune/modules/peft/_utils.py:19-36` — `AdapterModule` protocol; single method `adapter_params() -> list[str]`. Structural protocol.
2. `torchtune/modules/peft/_utils.py:132` — `adapter_key_filter = lambda x: "lora" in x or "magnitude" in x`. **Substring-based, not protocol-based.** Prefix tuning state would NOT survive this filter without extending it.
3. `torchtune/modules/peft/lora.py:57-126` (LoRALinear) and `dora.py:51-154` (DoRALinear) — current AdapterModule implementations. Both wrap nn.Linear; prefix tuning does not.
4. **35 sites** of `adapter_cls = DoRALinear if use_dora else LoRALinear` across 13 `_component_builders.py` files — friction even though prefix tuning doesn't modify linears.
5. `torchtune/modules/attention.py:262-282` (MultiHeadAttention.forward) — one candidate injection point after K/V projection.
6. `torchtune/modules/transformer.py:132` (TransformerSelfAttentionLayer) — alternative injection point (wrap, don't subclass).
7. `torchtune/modules/kv_cache.py:32-41` — `KVCache.__init__` allocates empty tensors at fixed size. Pre-filling with prefix K/V requires constructor change.
8. `torchtune/modules/position_embeddings.py:69-122` — RoPE applied per-token by absolute `input_pos` index. **Design fork:** prefix tokens at positions [0, N) (shifts real tokens) or special-cased outside position space.
9. `torchtune/datasets/_packed.py:57-67, 139` — PackedDataset resets position IDs per sample. Prefix tokens break the "positions start at 0 per sample" assumption.
10. `recipes/lora_finetune_distributed.py` — loss masking; prefix tokens shouldn't contribute to loss.
11. `torchtune/training/checkpointing/_checkpoint_client.py:170-173` — calls `get_adapter_state_dict()` which depends on the substring filter.
12. `torchtune/models/convert_weights.py:240-283` — HF PEFT export; prefix tuning would need a new conversion path or skip HF export entirely.
13. `torchtune/generation/_generation.py:72-112` — generation path; prefix needs to be present at every decode step.

## Five real design decisions a Hard answer must address

1. **Attachment point:** TransformerSelfAttentionLayer wrap vs. MultiHeadAttention injection. No-inheritance policy constrains this.
2. **Position embeddings:** Do prefix tokens get RoPE? At what positions? Cleanly preserving `input_pos` semantics requires care.
3. **KV-cache:** Pre-fill at construction (saves recomputation, complicates `KVCache.__init__`) vs. inject per-step (simpler init, repeats work).
4. **Sequence packing:** PackedDataset assumes per-sample positions start at 0. Prefix offsets break this — does the dataset need awareness?
5. **Loss masking:** Should prefix tokens contribute to loss? Where in the recipe does this gate live?

## Friction vs IA3 (rejected as too narrow)

- IA3: ~80 LOC of friction concentrated in `convert_weights.py` (HF-PEFT compat). Rejected.
- Prefix tuning: **10+ files spanning** attention, KV-cache, position embeddings, packing, recipes, checkpointing. Modifies sequence structure (deeper than weight scaling). Clear architectural breadth.

## Greppability check

- `grep prefix /home/user/torchtune/torchtune/` → state-dict hook prefixes (unrelated), dir_prefix args (unrelated). **Zero existing scaffolding.**
- No "prompt_tuning", "p_tuning", "prefix_embed", "PrefixTuning" anywhere.

## Weakness flagged by verifier

> "Existing adapter abstraction is almost too clean. Once you define PrefixTuningModule with adapter_params(), the peft utility functions mostly work."

**Implication:** A shallow answer can land on "add PrefixTuningModule + extend filter + done." A Hard answer must engage the position / KV / packing / loss-masking design fork. This is the discriminator the rubric will need to enforce.

## Decision

Catalyst is viable. Proceed to prompt drafting.
