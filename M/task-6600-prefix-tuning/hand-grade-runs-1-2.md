# Hand-grade of Runs 1 and 2 against rubric v1

**Date:** 2026-05-20
**Rubric:** `/home/user/torchtune/M/pytorch-torchtune-architecture/rubric.json` (34 criteria)
**Method:** strict pass/fail per criterion description

## Summary

| Run | Critical reasoning | Critical completeness | Critical total | Bonus | Penalty | Net |
|---|---|---|---|---|---|---|
| Run 1 | 6/10 (60%) | 7/7 (100%) | 13/17 (76%) | 5.5/7 (79%) | 0 | ~72% |
| Run 2 | 7/10 (70%) | 6/7 (86%) | 13/17 (76%) | 3/7 (43%) | 0 | ~70% |

**Verdict:** Both runs converge at ~70–76%. Slightly above the session-log target of 50–55% for Hard band.

## Critical reasoning — pass/fail breakdown

| # | Criterion | Run 1 | Run 2 |
|---|---|---|---|
| 1001 | Polymorphic seam + many LoRA-string-matched seams tension | ✓ | ✓ |
| 1002 | Substring filter + ≥3 duplications | ✓ | ✓ |
| 1003 | MultiHeadAttention has no K/V extension seam | ✓ | ✓ |
| 1004 | Attachment-point design with ≥2 options compared | ✓ (A/B/C) | ✓ (kwarg vs subclass) |
| 1005 | **RoPE-on-prefix design fork** | ✗ | ✗ |
| 1006 | KV cache pre-fill vs regen with tradeoff | ✗ (no comparison) | ✓ (explicit (a)/(b)) |
| 1007 | **PackedDataset position-IDs** | ✗ (mask only) | ✗ |
| 1008 | **Loss masking** | ✗ | ✗ |
| 1009 | No-inheritance honored | ✓ | ✓ |
| 1010 | get_merged_lora_ckpt unconditional + no merge for prefix | ✓ | ✓ |

**The three sequence-level criteria (1005 RoPE, 1007 packing, 1008 loss) fail on both runs.** This is the catalyst's structural discrimination working.

## Critical completeness

| # | Criterion | Run 1 | Run 2 |
|---|---|---|---|
| 2001 | peft/_utils.py | ✓ | ✓ |
| 2002 | LoRALinear + DoRALinear + nn.Linear wrap | ✓ | ✓ |
| 2003 | MultiHeadAttention + TransformerSelfAttentionLayer | ✓ | ✓ |
| 2004 | KVCache with shape/cache_pos | ✓ | ✓ |
| 2005 | lora_finetune_*.py as template + 2 new recipes | ✓ | ✓ |
| 2006 | convert_weights.py tune_to_peft | ✓ | ✓ |
| 2007 | generation/_generation.py + prefix at decode | ✓ | ✗ |

## Bonus

| # | Criterion | Run 1 | Run 2 |
|---|---|---|---|
| 3001 | 35-site adapter_cls + post-construction attachment | partial (0.5) | ✗ |
| 3002 | hasattr post-load init pattern | ✗ | ✓ |
| 3003 | HF PrefixTuningConfig | ✓ | ✓ |
| 3004 | GQA expansion shape ordering | ✓ | ✗ |
| 3005 | flex BlockMask incompatibility | ✓ | ✗ |
| 3006 | FSDP × small per-layer + to_empty | ✓ | ✓ |
| 3007 | Sequenced 3+ PR plan | ✓ | ✗ (Path B bundles) |

Run 1 wins on micro-discriminators (GQA shape, BlockMask, PR sequencing). Run 2 wins on hasattr-pattern recognition.

## Penalties triggered

Neither run triggers any penalty. Both:
- Use Protocol + composition (no base class)
- Respect copy-paste-modify policy
- Avoid `lora_prefix` naming hack (Run 2 explicitly warns against it)
- Locate checkpointer pain at call sites / converter helpers, not the checkpointer classes
- Cite file:line throughout

## Diagnosis vs. target

Per session-log methodology:
- Target Run 1 hand-grade: ~60–70%
- Target Run 2 hand-grade: ~45–55% (Run 2 is typically the "second strongest")
- Step-5 typically 5–10pt stricter than hand-grade
- Acceptable Step-5 average: 35–55% (upper-Hard / lower-Medium)

Current state:
- Run 1: 72% — slightly above target
- Run 2: 70% — well above target

**The critical completeness section is what's keeping the score high** — both runs hit 86–100% of file-naming criteria. That's expected (these are core files that any thorough answer will find), but it inflates the net score.

## Options to tighten

1. **Add 2 more critical reasoning criteria** that both runs fail. Candidates:
   - "Identifies the sequence-length change implication AND traces ≥2 downstream consequences" (mask, position IDs, loss accounting, tokens-seen metric, packing) — Run 1 partial (mask only); Run 2 ✗
   - "Identifies that the no-inheritance policy specifically blocks subclassing core attention modules" — both runs ✗ (Run 2 explicitly considers subclassing as acceptable)
   - "Identifies the `_is_dora` substring sniff in recipe code as a 4th duplication of the LoRA-substring assumption, particularly insidious because it leaks into user-facing recipes" — Run 1 ✗; Run 2 ✓

2. **Demote 1–2 critical completeness to bonus** (recipe template, convert_weights — both grep-trivial for any thorough answer) — this rebalances toward reasoning-heavy critical.

3. **Combine 1005–1008 into a single all-or-nothing "sequence-level engagement" criterion** — runs already fail all four; combining doesn't change the math but tightens the rubric narrative.

4. **Accept the band**. Per session log, slightly above 50–55% hand-grade is "acceptable given catalyst limitations." Prefix Tuning is less greppable than FP8 QAT and produces more reproducible misses; the rubric does discriminate them at the design-question level.

## Recommendation

Apply Option 1 (add the 2 sequence-length-consequences + no-inheritance-blocks-core-attention criteria). Both runs fail both → drops critical reasoning to 6/12 (50%) and 7/12 (58%). Combined critical drops to ~62–65%. Net lands at ~62–68%. That's at the methodology's upper-Hard boundary.

Apply Option 4 if the user is satisfied with current 70–72%.
