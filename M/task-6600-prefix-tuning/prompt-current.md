# Prefix Tuning catalyst — Prompt v1 (initial draft for Run 1 test)

**ASCII verified.** ~155 words.

## Prompt

> I want to add Prefix Tuning to torchtune as a third PEFT method alongside the existing LoRA and DoRA support. Prefix tuning is the Li & Liang (2021) approach: a small number of learnable prefix vectors per transformer layer that the model attends to during training and inference, while base weights stay frozen. The adapter state is a new tensor per layer; nothing wraps any existing nn.Linear. Single-device and distributed. I want the design to still make sense when the next not-LoRA-shaped adapter shows up. This is my first contribution.
>
> Walk me through what implementing this actually requires. I'm trying to follow the contribution rules — recipes stay readable as self-contained scripts, no implementation inheritance. I know the README says torchtune is wound down; assume for this question that the codebase will be forked and actively maintained going forward.
>
> Give me an honest read on whether the codebase shape makes this addition clean or whether I'd be adding pain future contributors will inherit. If it's the latter, propose what to fix first.

## What each sentence does

- "third PEFT method alongside LoRA and DoRA" — anchors to the existing adapter family without naming `peft/` directory.
- "Li & Liang (2021) approach... learnable prefix vectors per transformer layer that the model attends to" — defines what's being built without naming any file or function in torchtune.
- "adapter state is a new tensor per layer; nothing wraps any existing nn.Linear" — flags the structural difference from LoRA. Without this, an answerer could miss that the existing AdapterModule protocol is parameter-name-based and might not fit.
- "Single-device and distributed" — forces cross-recipe reasoning.
- "still make sense when the next not-LoRA-shaped adapter shows up" — the future-proofing constraint that demands a structural design, not piecemeal helpers. Doesn't name "registry" / "protocol" / "factory".
- "This is my first contribution" — Task 5137 natural-voice cue.
- "recipes stay readable as self-contained scripts, no implementation inheritance" — names the policy; answerer must find `recipe_interfaces.py` to ground it.
- "forked and actively maintained going forward" — blocks the README "no longer actively maintained" escape (Run 2 used this for FP8 QAT).
- "honest read... clean or adding pain... propose what to fix first" — trace + diagnose + propose structure.

## Things deliberately NOT in the prompt

- File names (none).
- Words: KV cache, position embedding, RoPE, packing, mask, loss, checkpointer, `AdapterModule`, `adapter_params`, builder.
- The number of files needing edits.
- The 35-site if/else pattern.
- Hint that position embeddings or KV cache are involved.
- Hint that sequence packing matters.

## What we expect a strong answer to surface (the discoveries)

- `AdapterModule` protocol with the parameter-name-based `adapter_params()` method
- `adapter_key_filter` substring match (`peft/_utils.py:132`)
- 35-site `adapter_cls = DoRALinear if use_dora else LoRALinear` pattern
- KV cache pre-fill question (`kv_cache.py:32-41`)
- RoPE / position embedding question (`position_embeddings.py:69-96`)
- Sequence packing position-ID assumption (`_packed.py:57-67, 139`)
- Loss masking in recipes
- HF PEFT export incompatibility (`convert_weights.py:240-283`)

## What we expect a weak answer to do

- Treat it as "define PrefixTuningModule + extend the filter + done"
- Miss the position embedding fork
- Miss the KV cache pre-fill question
- Miss the sequence packing implication
- Propose an inheritance-based base class
