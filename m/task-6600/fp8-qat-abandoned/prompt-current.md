# Task 6600 — Current locked prompt

This is the prompt sent to the third calibration subagent (Run 3). The next session should treat this as the working prompt unless the calibration check in `session-log.md` indicates it needs revision.

**ASCII verified.** No non-ASCII characters (reviewer red flag per `m/instructions broken/INDEX.md`).

**Word count:** ~125 words. Comparable to Task 5137's reference (~50 words), longer because of the multi-part ask (trace + diagnose + propose + scope-discipline).

---

## Prompt

> I want to add FP8 QAT to torchtune — fake-quantize weights to FP8 during training, deploy as actual FP8 at inference. Single-device and distributed. This needs to compose with LoRA, and I want the design to still make sense if a sixth scheme shows up next year that needs both training-time fake-quant AND a custom forward graph for adapter composition. This is my first contribution.
>
> Walk me through what implementing this actually requires. I'm trying to follow the contribution rules — recipes stay readable as self-contained scripts, no implementation inheritance. I know the README says torchtune is wound down; assume for this question that the codebase will be forked and actively maintained going forward. Give me an honest read on whether the codebase shape makes this addition clean or whether I'd be adding pain future contributors will inherit, and if it's the latter, propose what to fix first.

---

## What each sentence is doing (for future revisions)

- *"I want to add FP8 QAT to torchtune — fake-quantize weights to FP8 during training, deploy as actual FP8 at inference."* — defines the feature with enough specificity that the model knows it's a NEW combination (FP8 + QAT) without naming the existing files.
- *"Single-device and distributed."* — forces cross-recipe reasoning.
- *"This needs to compose with LoRA"* — forces engagement with the adapter composition surface (`QATLoRALinear`, `swap_lora_linear_with_qat`).
- *"the design to still make sense if a sixth scheme shows up next year that needs both training-time fake-quant AND a custom forward graph for adapter composition"* — blocks the "just write piecemeal helpers" answer Run 2 produced. Forces a structural design.
- *"This is my first contribution."* — soft framing that triggers "explain me the codebase" mode (closer to Task 5137's natural-voice style).
- *"recipes stay readable as self-contained scripts, no implementation inheritance"* — names the policy without quoting it. The model has to find `recipe_interfaces.py` to ground it.
- *"assume for this question that the codebase will be forked and actively maintained going forward"* — blocks the README "no longer actively maintained" escape that Run 2 used to scope down.
- *"Give me an honest read on whether the codebase shape makes this addition clean or whether I'd be adding pain future contributors will inherit, and if it's the latter, propose what to fix first."* — the trace + diagnose + propose structure. The "if it's the latter" branching forces the model to make the judgment call, not punt.

## Things deliberately NOT in the prompt

- File names (none mentioned).
- The word "lifecycle" (would coach the right framing).
- "Quantization fragmentation" or any name for the tension.
- Any quote from `recipe_interfaces.py`.
- Names of integration points.
- The expected number of integration points.
- Any hint that NF4 / QLoRA is relevant (consistent miss in Runs 2 & 3 — accepted as a rubric discriminator).
