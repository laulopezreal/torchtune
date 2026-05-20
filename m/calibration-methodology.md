# Calibration methodology — prompt-testing playbook

The reusable methodology we evolved during Task 6600 (`m/task-6600/session-log.md`).
Use this as the playbook for the next Hard Architecture task on this repo, or as a
starting point for adapting to other intents (RCA, Onboarding, PR Triage).

The official guide (Part 7 / `m/instructions broken/INDEX.md`) tells us *what* a Hard
prompt should ask. This file tells us *how to find out* whether a draft prompt actually
lands at Hard band — before sinking time into rubric design + Step 5 eval.

---

## Why test prompts with a subagent at all

The Step 5 model eval is the official source of truth for difficulty calibration. It's
also expensive (5–10 min per run, averaged across 3), grades against a rubric we
haven't written yet, and only runs after the prompt + golden + rubric are all drafted.

By that point, prompt-level problems (over-coaching, ungreppable catalyst, ambiguous
ask) have cost a full rubric-design cycle to discover. Subagent-testing is the
cheap pre-flight: ~5–10 minutes per run, no rubric needed, surfaces prompt-level
problems before any downstream work is sunk.

It's not a substitute for Step 5 — it's a way to *not waste* Step 5 runs on prompts
that obviously won't land.

---

## When to use it

Use the subagent test loop **after** these gates are passed:

1. Tension picked, verified in code (see "Catalyst verification" below).
2. Catalyst chosen (specific feature request that forces engagement with the tension).
3. Draft prompt written following Part 7 rules.

Don't bother testing before. A prompt that's targeting a non-tension or naming the
answer path will produce noise.

---

## The harness

### Subagent setup

```
agent_type: general-purpose
context_isolation: low — exclude CLAUDE.md, M/, anything containing our analysis
repo_state: BASE SHA of the task
instructions: "answer as a senior contributor would in a design doc or Slack DM"
```

The agent prompt should be a self-contained brief, not a partial conversation. It must:

- Say **what the developer is asking** (the prompt under test, verbatim).
- Say **where the repo is** (absolute path).
- Say **what files to ignore** (CLAUDE.md, M/, anything with prior analysis).
- Say **what kind of answer to produce** (senior-contributor design doc, repo-grounded,
  honest about uncertainty).
- Tell the agent **not to ask for clarification** — answer from the prompt as written.

Template:

```
You are answering a colleague's developer-facing question about the
<REPO_NAME> codebase. The repository is at <ABSOLUTE_PATH>. Investigate
the code there to ground your answer in specific files, functions, and behaviors.

IMPORTANT — things you must NOT read:
- <PATH_TO_CLAUDE_MD>
- <PATH_TO_M_FOLDER>

Use only the actual upstream <REPO_NAME> source.

Your goal: produce a developer-facing answer the way a senior contributor would
respond in a design doc or detailed Slack reply. Repo-grounded, specific about
file paths and function names, honest about tradeoffs and any uncertainty. Do
not ask for clarification — answer the question as written.

The developer's question follows. Answer it completely.

---

<PROMPT UNDER TEST>
```

### Why low context matters

Including CLAUDE.md, M/, or our prior analysis biases the subagent toward our
expected answer path. The whole point of the test is to see what a model that's
never seen our notes produces from the prompt + the repo alone. That's what
Step 5 will simulate; we want the pre-flight to match.

### Why general-purpose, not Explore

Explore is read-only and excerpt-based — it can't produce a synthesized
developer answer. The Step 5 eval grades a synthesized answer, so we test
with one. Use general-purpose (which can do multi-step exploration and
write a complete response) or Plan if you want a structured design-doc shape.

---

## The signals to read

Three categories. Note them down before launching the subagent so you have an
honest baseline to grade against.

### 1. Discovery — what should the model find?

Before you launch, write down the file+function citations a strong answer
**must** include. From the catalyst, work out the 4–8 specific spots in the
codebase the answerer has to land on. These come from the tension verification step.

Then, after the run: which did the model find? Which did it miss? Which did
it find that you didn't anticipate?

### 2. Reasoning — does the model engage with WHY?

A Hard answer requires the model to explain *why* something is structured the
way it is — not just where it is. For each piece of friction the prompt
exposes, the answerer should be able to justify the current arrangement
before proposing changes.

The phrasing to look for: *"what would break if you moved X."* A model that
only enumerates files and not justifications is producing a shallow answer.

### 3. Design — does the model honor the constraint?

If the prompt names a constraint (no-inheritance policy, must work with
some other subsystem, future-proof for scheme N+1), did the model honor it
in the proposal? Or did it propose something the constraint forbids?

This is the most direct calibration signal: a *strong* model that gets the
constraint right is showing depth. A weaker model that violates it
gives us penalty-criterion ammo for the rubric.

---

## Grading the run

For each run, score informally against the three categories above. A rough rubric:

| Category | Strong (≥80%) | OK (50–80%) | Weak (≤50%) |
|---|---|---|---|
| Discovery | Found all expected citations + bonus discoveries | Found 60–80% of expected, missed 1–2 | Found <60%; missed major surfaces |
| Reasoning | Justified at least half the discoveries with "what would break" | Some justification, mostly enumeration | Pure enumeration, no WHY |
| Design | Constraint-honoring + non-trivial mechanism | Constraint-honoring but piecemeal | Constraint-violating or no design |

Average the three. That's your estimated Step-5 pass rate.

### Calibration bands

- **Too easy (>70%):** the prompt over-coaches. Tighten.
- **Hard band (35–55%):** good. Lock the prompt, move to rubric design.
- **Too hard (<25%):** the prompt under-specifies, or the catalyst is wrong.
  Add starting context (per Part 7's "preserve answerability" rule) or
  reconsider the catalyst.

Note: aim for 35–55% (slightly above Hard's stated 0–35% band) because the
rubric does additional calibration once written. Subagent grading is informal;
formal grading by Step 5 against the rubric will be a few points stricter.

---

## Iteration rules

### Too easy → tighten

Specific moves, in order of safety:

1. **Remove named files from the prompt.** If you wrote `convert_to_float8_training`
   in the prompt, the model gets a free grep anchor. Strip file/function names.
2. **Remove pre-articulated answer steps.** If you wrote "look at module
   construction, then post-load, then mid-training" — that's the answer in
   disguise. Strip it.
3. **Remove design hints.** Words like "lifecycle hooks", "registration
   mechanism", "protocol" telegraph the right design shape.
4. **Remove the WHY question phrasing.** "Why do they happen at different
   points" pre-articulates the reasoning move. Let the model figure out
   it needs to ask that.

### Too easy → DON'T do these

These are anti-calibration moves we discovered the hard way:

- **Don't add Hard-sounding constraints to force depth.** "Must compose
  with X, future-proof for Y, also work under Z" prompts *more* engagement
  from competent models, not less. They use the extra constraints to
  demonstrate strength. (Task 6600 Run 3 caught this.)
- **Don't add file names to "force" the model to engage with them.**
  Naming a file is anti-Hard — Hard requires discovery.

### Too hard → relax

1. **Name the symptom or starting context.** Part 7's "preserve
   answerability" rule. If the model can't figure out where to start,
   give it the *kind* of feature being added, not the file to edit.
2. **Disambiguate the ask.** If the prompt has two plausible
   interpretations, the model picks the wrong one or hedges. Pin the
   interpretation.
3. **Add a constraint that grounds the search.** "Must work for
   single-device AND distributed" forces specific repo regions; "must
   work in production" doesn't.

### Too hard → DON'T do these

- **Don't add file names to "help" the model.** Over-correction; jumps
  straight back to Easy band.
- **Don't add the design pattern by name.** If you tell the model what
  the right abstraction is, you've answered the question.

---

## When to stop iterating

Three runs is usually enough:

- Run 1: baseline against the first draft. Tells you whether the catalyst
  is workable and which direction to iterate.
- Run 2: a tightened or relaxed version. Tells you whether direction-1
  iteration is monotonic.
- Run 3: a refined version, OR confirmation that further tightening
  isn't productive.

If after 3 runs you're not converging on Hard band, the issue is usually
**the catalyst, not the prompt**. The catalyst can be too greppable (any
competent model finds the right files in 2 keyword searches) or too
abstract (no concrete feature anchors the question). Reconsider the catalyst
before drafting a 4th prompt — and consider whether the rubric can do the
remaining calibration work the prompt can't.

---

## The "greppable catalyst" trap

The catalyst names the feature being added (e.g., "FP8 QAT", "IA3 support",
"new VLM with different vision encoder"). If the feature name produces a
clean keyword match against the codebase, **the prompt has a Hard ceiling
that the rubric — not the prompt — has to enforce**.

For Task 6600, "FP8 QAT" greps land any competent model on
`convert_to_float8_training` and the `qat_*` recipes in seconds. Three
prompt iterations couldn't move the score below ~55%, because the
greppability of the catalyst made discovery essentially free.

In that case, the calibration shifts: instead of trying to make the
*prompt* harder, design the *rubric* to grade specifically what the
model couldn't infer from grep alone:
- Missing integration points (Task 6600: NF4 builder-time consistently missed).
- Missing WHY justifications.
- Naive design proposals (piecemeal helpers vs. unified registry).
- Penalty traps (proposing base classes; claiming files are involved when they aren't).

Less-greppable catalysts buy more difficulty headroom for the prompt.
Greppable catalysts shift the calibration burden to the rubric.

---

## Catalyst verification step

Before drafting any prompt, **verify the friction is real in code**.
Verify with `grep`, `Read`, or a focused Bash session — don't trust
intuitions about how the codebase is shaped.

Concrete checks:

1. **Run the failure case mentally.** "If I were the user adding this
   feature, what files would I need to edit?" Trace the actual call sites.
2. **Verify the cited friction.** If the claim is "X is coupled to Y",
   open both X and Y and check.
3. **Look for the actual blast radius.** `grep -rn` for the names of
   the coupled symbols. Count the call sites.

Task 6600 caught a bad framing this way: I claimed "every new adapter
forces edits inside the checkpointer classes." Code check showed the
friction was ~80 LOC in `convert_weights.py`, not the checkpointer
classes. Without that check, we'd have drafted a prompt against a
non-tension and the rubric would have failed Validate Golden.

Time spent here pays back many times during prompt iteration and rubric
design.

---

## Decision tree (quick reference)

```
Is the friction verified in code?
├── No → verify before drafting any prompt
└── Yes → draft a prompt that names the catalyst but no files/answer paths

Run a subagent test with low context.

Estimated pass rate?
├── >70% → tighten the prompt (remove named files, pre-articulated phases, design hints)
├── 35–55% → lock the prompt, move to rubric design
└── <25%  → add starting context (Part 7 "preserve answerability")

After 3 runs without convergence:
├── Catalyst greppable? → accept current band, lean on rubric for calibration
├── Catalyst ambiguous? → reconsider catalyst
└── Constraint not biting? → tighten the constraint (not the discovery)

Once locked:
- Use the strongest subagent run as the structural template for golden_answer.md.
- Use missed pieces across runs as critical/bonus criteria.
- Use consistent traps (proposing base class, claiming wrong files involved,
  naive unification) as penalty criteria.
- Hand-grade Run 1 and Run N against the draft rubric. Target Run 1 ~65%, Run N ~50%.
```

---

## Pointers

- Hard prompt rules: `m/instructions broken/INDEX.md` § "Prompt rules" + "Hard-task hardening".
- Rubric mechanics: `m/instructions broken/INDEX.md` § "Rubric rules".
- Task 5137 structural template: `m/instructions broken/INDEX.md` § "Task 5137 as a structural template".
- Worked example of this methodology: `m/task-6600/session-log.md` (full journey),
  `m/task-6600/run-{1,2,3}-answer.md` (raw subagent outputs).
