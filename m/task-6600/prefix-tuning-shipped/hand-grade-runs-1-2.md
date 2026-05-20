# Hand-grade of Runs 1 and 2 against rubric v2 (tightened)

**Date:** 2026-05-20
**Rubric:** `/home/user/torchtune/m/pytorch-torchtune-architecture/rubric.json` (36 criteria, post-tightening)
**Method:** strict pass/fail per criterion description

## Tightening applied vs v1

1. **Added 2 critical reasoning criteria:**
   - 1011 Sequence-length change + ≥2 downstream consequences
   - 1012 No-inheritance specifically blocks subclassing core attention modules
2. **Demoted 2 critical completeness to bonus** (recipe template, convert_weights — both grep-trivial):
   - Former 2005 → 3008
   - Former 2006 → 3009

## Summary

| Run | Critical reasoning | Critical completeness | Critical total | Bonus | Style | Penalty | Net |
|---|---|---|---|---|---|---|---|
| Run 1 | 6/12 (50%) | 5/5 (100%) | 11/17 (65%) | 7.5/9 (83%) | 2/2 | 0 | ~68% |
| Run 2 | 7/12 (58%) | 4/5 (80%) | 11/17 (65%) | 5/9 (56%) | 2/2 | 0 | ~63% |

Improvement vs v1 rubric: Run 1 72% → 68%, Run 2 70% → 63%.

## Critical reasoning fails (both runs)

| # | Criterion | Run 1 | Run 2 |
|---|---|---|---|
| 1005 | RoPE-on-prefix design fork | ✗ | ✗ |
| 1006 | KV cache pre-fill vs regen tradeoff | ✗ (no comparison) | ✓ |
| 1007 | PackedDataset position-IDs | ✗ (mask only) | ✗ |
| 1008 | Loss masking | ✗ | ✗ |
| 1011 | Sequence-length + ≥2 consequences | ✗ (mask only) | ✗ |
| 1012 | No-inheritance blocks core-attention subclassing | ✗ | ✗ (Run 2 explicitly accepts subclassing) |

**Five critical reasoning items consistently missed.** The 1012 strict fail on Run 2 is notable — Run 2 says "subclass MultiHeadAttention (acceptable, mirrors how LoRALinear is a peer of nn.Linear)" which is exactly the failure mode the criterion targets.

## What both runs DO get right

| # | Criterion | Both |
|---|---|---|
| 1001 | Polymorphic seam vs many LoRA-coupled seams | ✓ |
| 1002 | Substring filter + ≥3 duplications | ✓ |
| 1003 | MultiHeadAttention no K/V extension point | ✓ |
| 1004 | Attachment-point design with ≥2 options | ✓ |
| 1009 | No-inheritance honored in proposal | ✓ |
| 1010 | get_merged_lora_ckpt unconditional + no merge for prefix | ✓ |
| 2001 | peft/_utils.py | ✓ |
| 2002 | LoRALinear + DoRALinear + nn.Linear wrap | ✓ |
| 2003 | MultiHeadAttention + TransformerSelfAttentionLayer | ✓ |
| 2004 | KVCache + shape/cache_pos | ✓ |

## Calibration verdict

Hand-grade Run 1 68%, Run 2 63%. Step 5 typically 5–10pt stricter:
- **Projected Step-5 pass rate: 55–60%**
- Per INDEX.md Hard band: 0–35% (strict)
- Per session log: 35–55% (acceptable for greppable catalysts)

Current projection sits at the **upper edge of acceptable for Hard**. Slightly higher than the session-log target but defensible:

1. The catalyst has zero greppable scaffolding — strong models still find the right files because they reason about PEFT structure, not because they grep "prefix"
2. Five critical reasoning items reproducibly fail across both runs — the discrimination IS happening at the right place (design depth, not file naming)
3. Both runs avoid all 8 penalty traps — they're genuinely good answers

The remaining gap to 50–55% target would require either:
- Adding 1–2 more critical reasoning criteria that even strong answers reliably miss (diminishing returns — answers like Runs 1 and 2 are already missing 5 of 12)
- Tightening completeness further (risk: golden answer can't hit 100% critical, breaks rubric)

**Recommendation:** lock the rubric here, draft the golden answer that hits 100% critical, then run Step 5 to see where it actually lands. The golden's structural template is:

1. TL;DR (3 sentences)
2. Current PEFT contract — name the 7 cooperating pieces (polymorphic seam + 6 LoRA-coupled seams)
3. What prefix tuning specifically breaks — walk the 5 design forks (attachment, RoPE, KV-cache, packing, loss)
4. Sequence-length consequences — trace mask + position IDs + loss + tokens-seen + packing
5. Two implementation paths (refactor-first vs ship-it-now) with PR sequencing
6. The minimal refactor that lets future not-LoRA-shaped adapters land cleanly
7. Honest read: clean or painful? Verdict + fix-first proposals
8. Cross-cutting limitations to flag in PR description (DoRA+prefix, HF PEFT export, flex attention, FSDP small param)
9. Files to touch (new + modified + intentionally untouched)
