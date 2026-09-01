# Appendix: "Meaning lives in the orthogonal complement" — a staged falsification

**Status: CLOSED (negative, Type B).** Do not spend more compute on this line.

**Question.** Is the semantic concept steerable through the component of the
hidden state orthogonal to the token-row shell — i.e. can we inject a single
hidden-state direction `d_per` (the residual of a thematic contrast after
removing its projection onto the rank-`r` token-shell subspace) and read it
back as *coherent semantic transport* in generated text?

**Geometry setup.** Token rows of the LM head `W` are normalized, PCA'd
(`svd_lowrank`, q=300). The shell rank is `r≈231` (90% energy). The
complement direction is

```
d_per = d - Uc (Ucᵀ d),   d = mean_state(target) - mean_state(neutral)
```

projected into the full residual space, with `shell-leak = ‖Uc(Ucᵀ d_per)‖ = 0.0000`.

**Controls** (all in the *correct* geometry, per prior review): random
control is a rotated random vector **projected into the complement**
(`shell-leak=0.0000`), not a full Haar draw (which leaks); `-d_per` sign
control at every α; class-defining vocabets (ANCHORS / ENUM_CLUSTER /
HELD_OUT); string-level HELD_OUT for text, token-ID logits for the per-α
diagnostic; a-priori coherence criterion (max_run<6, distinct-1>0.6).

---

## Stage 1 — cheap mechanistic logit scan (8 s, no sampling)

For a batch of fixed prefixes, at each dose, measure `ΔL_held`, `ΔL_anchor`,
`ΔL_neutral` on the **readout** for `+d_per`, random-complement, `-d_per`
vs the natural baseline.

```
α     +dper held   rand held   -dper held   held−neutral(+dper)
.04   +0.40        -0.02        -0.40         +0.445
.10   +1.00         0.00        -1.00         +1.112
.16   +1.60        -0.07        -1.60         +1.778
.22   +2.20        -0.10        -2.20         +2.445
.28   +2.80        -0.13        -2.80         +3.111
.34   +3.40        -0.16        -3.40         +3.778
.40   +4.00        -0.19        -4.00         +4.444
```

- `ΔL_held(+d_per) = 10.0·α` — perfectly linear, r=10.0.
- **Random ≈ 0** at every dose (≤0.19, no shell leak). **Sign control is an
  exact mirror.** → The effect is genuinely carried by `d_per`, not by
  generic perturbation.
- But `ΔL_anchor = 1.1 × ΔL_held` at every dose (e.g. +4.40 anchor vs +4.00
  held at α=0.40). **It is a royal-region readout direction, not a
  held-out-selective one.**

## Stage 2 — gradient screen, K=4 at the promising doses

```
α     +dper  supA   rand   -dper   dLogitH+  dLogitB+  dLogitN+  minHrank  topPentry
.28   1/4    1/4    0/4    0/4     +328      +381      -38        1        0.32
.34   0/4    1/4    0/4    0/4     +382      +444      -44        1        0.29
.40   0/4    0/4    0/4    0/4     +426      +495      -49        4        0.10
```

Per-step rank / nucleus-entry tracking (the decisive metric) shows:

- **`minHrank=1`** at α=.28/.34 — a HELD_OUT word (crown/reign/kingdom/…)
  reaches **rank 1** in sampling logits at some step. Not "rank 400".
- **`topPentry≈0.30`** — a HELD_OUT word is inside the top-p nucleus ~30% of
  steps, while random ≈ 0 (its `dLogitH+` deltas are *negative*, −30…−43).
- **Yet transport ≤ 1/4** across all doses; isolated single-word hits
  (matching baseline noise). Pre-registered falsification criterion
  (`exists α: R/4 ≥ .5 & R>Q & M<R`) fails everywhere → **NO-TRANSPORT**.

## The evidence hierarchy (what is established)

```
complement direction → logit movement → top-p access ⇏ coherent semantic transport
```

Multi-modal support:

| Observation | Reading |
|---|---|
| random-complement ≈ no held-out effect (0 leak) | not generic perturbation |
| `+d_per` / `-d_per` are near-exact mirrors | structured direction, not noise |
| HELD_OUT reaches rank 1 at α=.28/.34 | real readout influence |
| HELD_OUT enters the nucleus | causal access to selection mass |
| no coherent transport emerges | fails generative control |
| `ΔL_anchor > ΔL_held` at every dose | royal-region, not held-out-selective |
| raising α worsens HELD_OUT rank (→4) / topPentry (→.10) | anchor over-commitment |
| anchor suppression does not rescue transport | not an anchor-averaging artifact |

## Defensible statement (reviewer-approved wording)

> **Whatever information about the target concept is encoded in the
> orthogonal complement is not sufficient, in this intervention geometry, to
> produce reliable semantic transport.** The complement has causal access to
> the readout, but the accessible direction is not an independently
> controllable semantic variable.

What we deliberately do **not** claim: (a) "meaning does not live in the
complement" — the experiment cannot establish that; (b) "the complement is
merely noise" — the sign control and random-complement comparison
demonstrate structured, reproducible information.

## Conceptual payoff: representation ≠ readout ≠ control

The line separates three notions that are **not interchangeable**:

| quantity | complement direction `d_per` |
|---|---|
| representation | ✅ encoded (royal contrast present) |
| readout influence | ✅ sign-asymmetric, random-zero logit effect |
| generative semantic control | ❌ no coherent transport |

This motivates the pivot: *why does logit-space contrast produce semantic
transport when hidden-state directions into this complement do not?*

---

*Files:* `logit_scan.py` (Stage 1), `falsify_orth3.py` (Stage 2, incl. the
anchor-suppressed `supA` condition and per-step rank / nucleus tracking).
Superseded: `falsify_orth.py` (over-aggregated, contaminated random control),
`falsify_orth2.py` (slow, no KV cache).

---

# Part II — What transports: row(W) cannot represent the generative signal

## The pivot question (after the ortho-complement falsification)

The complement direction `d_per` fails to transport even with correct
controls. The *working* mechanism is **logit-space contrast** (per-sentence
z-scored next-token logit difference, top-200 masked, added each step).
The open question: **what distinguishes the working logit contrast from a
hidden-state contrast projected through the output map `W`?**

## 1. Pure vector geometry (fantasy task, Qwen2-1.5B)

```
cos(dL_z_full, W d_per)        = +0.854    (the DIFFUSE contrast == hidden dir)
cos(dL_static(top-200), Wd_per)= +0.111    (the SPIKES are ~orthogonal to it)
spearman(dL_z, Wd_per)         = +0.817
sign-agreement                 = 80.7%
top-k overlap (K=200)          = 76 / 200  (38%)
```

The top-k masking is **not a neutral sparsification**: it selects a logit
subspace nearly orthogonal to `W d_per`. The transport-carrying spikes live
in the 99% residual.

## 2. Causal decomposition (decomp_gen.py, fantasy, canonical config)

```
dL_static = dL_parallel + dL_perp      (dL_parallel = proj onto Wd_per)

||dL_parallel|| ≈ 11.4   (~11% of dL)
||dL_perp||     ≈ 101.5  (~99%)
||dL_full||     ≈ 102.1

             transport   minHrank
dL_parallel   0/6         216    (hidden-representable -> does NOT transport)
dL_perp       3/6           0    (readout residual      -> does transport)
dL_full       2/6           0    (= the residual, ~99%)
random(matched norm) 0/6   260    (not magnitude)
```

- `dL_perp` matches/exceeds `dL_full` (3/6 vs 2/6, with better coherence:
  maxrun 1.7 vs 14.5). Phrased carefully: the residual matched or exceeded
  the full intervention in this 6-seed test with better repetition metrics.
- `dL_parallel ≈ random` (0/6, held-out never rank<216). The hidden-reachable
  slice is generatively inert.
- The small parallel component appears to *hurt* coherence when added back
  (dL_full maxrun jumps to 14.5) — suspected generic repetition attractor,
  not semantic.

## 3. Subspace hierarchy (subspace_hierarchy.py) — the sharp boundary

Project `dL_static` onto progressively richer hidden-reachable subspaces
(all **rescaled to the reference norm** so dose is identical):

| subspace | dim | transport | cos(proj, dL_full) |
|---|---|---|---|
| L0 `W d_per` | 1 | 0/6 | +0.113 |
| L1 shell image `W(Uc)` | 231 | 1/6 | +0.097 |
| L2 full `row(W)` | 1536 | 0/6 | +0.195 |
| L3 full logit space (= ref) | 152k | **2/6** | +1.000 |

**Interpretation (outcome A, sharp representability boundary):**
even the *full* rank-1536 hidden-to-logit image projects onto the success
vector with cosine only **+0.20, and does not transport**. This is not a
direction-selection issue — the generatively useful signal is essentially
**orthogonal to all of row(W)**. The success vector lives in the residual
`row(W)⊥` relative to the model's linear readout.

## 4. Defensible claim (reviewer-approved wording)

> The semantic transport signal is present in logit space, but the tested
> hidden-state direction — indeed **any** hidden-state direction, i.e. the
> whole linear image `row(W)` — cannot reproduce the portion of that signal
> responsible for generation.

Or: **the generatively effective operation exploits structure created by the
nonlinear/selective post-readout construction** (per-sentence z-normalization
+ top-k mask on the *output distributions*), not a hidden-state direction.
This is the "exploits post-readout transformation" statement.

Caveats kept explicit: the residual may reflect token-specific
nonlinear/contextual effects, information unavailable to averaged hidden
states, the z-normalization procedure, higher-order interactions, and/or the
specific TGT/NEU construction. We do **not** claim a general "semantic
residual"; we claim the narrower result above for these experiments.

*Files:* `vec_compare.py` (geometry), `decomp_gen.py` (causal
decomposition), `subspace_hierarchy.py` (representability boundary).

---

# Part III — Mechanism matrix: which operation is causal?

## Question

The decomposition showed the transport signal is ~99% outside `row(W)`. But
is that *escape* itself the mechanism, or is it a side effect of something
else? Separate the candidate operations with a factorial:

- **Normalization**: raw / centered / z-scored / per-sentence z-sum
- **Mask**: none / top-25 / top-50 / top-200 (positive top-k selection)
- **Controls** (all at top-200, norm-matched to the working reference):
  random coords (own values), random coords (sorted top-200 magnitudes),
  real top-200 coords with permuted values, real coords with equal weight,
  and the full `row(W)` projection of the working vector.

Every condition is **rescaled to the same effective logit norm** (`N_REF`),
so dose cannot explain any difference.

## Phase A — row-space fraction vs K

```
K         R_row(perz)      residual (1-R)
1         0.011            98.9%
10        0.015            98.5%
25        0.018            98.2%
50        0.022            97.8%
100       0.027            97.3%
200       0.038            96.2%
500       0.065            93.5%
1000      0.097            90.3%
5000      0.252            74.8%
V (full)  0.999             0.1%
```

Masking *creates* the row-space escape: the top-200 vector is **96% outside
row(W)**; the unmasked vector is **99.9% inside**. `zs` and `perz` give
nearly identical fractions (normalization barely moves geometry).

## Phase B — the causal table (SEEDS=6, fantasy)

```
cond       transport  dLogP_H  dLogP_U  maxrun  dist1  R_row
raw        0/6   +0.44  -0.58    0.0  0.68  1.000
raw_t25    0/6   -7.53  -7.19   33.0  0.67  0.018
raw_t50    0/6   -7.57  -7.59   60.2  0.43  0.022
raw_t200   2/6   -2.03  -2.42   14.5  0.56  0.038   ← transports
cent_t200  1/6   -1.74  -2.38   14.8  0.47  0.038
zs_t200    1/6   -1.74  -2.38   14.8  0.47  0.038
perz_t200  2/6   -2.00  -2.49   14.5  0.55  0.038   ← transports
rand200    0/6   -5.76  -6.84   83.3  0.25  0.012   ← degenerate
magmatch200 0/6  -3.68  -3.81   52.7  0.35  0.011   ← degenerate
shuffle200 0/6   -1.92  -2.58   17.7  0.44  0.037   ← dead
equal200  0/6    -1.90  -2.54   25.7  0.43  0.038   ← dead
rowW_proj 0/6    +0.79  -0.29    0.0  0.62  1.000   ← dead (back in rowW)
```

## What is causal

1. **Normalization is disposable.** raw/cent/zs/perz at top-200 are all
   ~1-2/6. The working "per-sentence z" is not the special ingredient; the
   raw contrast works equally once top-k-selected.

2. **Top-200 *coordinate* selection is necessary.** t25/t50 are dead and
   degenerate (maxrun 33-60 repetition); no-mask (full row-space vector) is
   dead. Only K≈200 carries the winner set the counterfactual needs.

3. **Row-space escape is necessary but NOT sufficient.** All top-200
   conditions sit ~96% outside row(W) — including the four dead controls.
   Escape alone explains nothing.

4. **The missing ingredient is the *coordinated* object: correct
   vocabulary coordinates × ranked magnitudes × out-of-rowW.**
   - `rand200`/`magmatch200` (wrong coords): dead + degenerate.
   - `shuffle200` (right coords, permuted values): dead.
   - `equal200` (right coords, equal weight): dead (no ranking gradient).
   - `raw_t200`/`perz_t200` (right coords, ranked magnitudes): transport.

5. **`rowW_proj` is the clean negative control**: projecting the working
   vector into `row(W)` kills transport (0/6) while keeping maxrun 0 — the
   signal itself is inert inside the hidden-reachable subspace, active
   outside it.

## Mechanism statement (reviewer-approved)

> The generatively effective operation is: take the vocabulary readout
> contrast, **select its top-200 positive coordinates with their ranked
> magnitudes, and apply the resulting sparse vector as a logit offset**.
> This vector necessarily escapes `row(W)` (masking breaks representability),
> and it is that *coordinated* sparse object — not any individual axis
> (normalization, sparsity, magnitudes, row-escape) — that transports.
> Random/equal/permuted/sparse/row-constrained variants all fail.

*File:* `mechanism_matrix.py`.

---

## Part IV — K × λ causal surface: sparsity and row-space interact

The mechanism statement isolates a *coordinated* object, but does not yet
separate its two axes. We cleanly manipulate both while holding dose fixed:

**Design.** For each K, `d_K = topk_pos(perz, K)` defines the sparse
coordinate set. We decompose `d_K = P_row d_K + P_\perp d_K` and set
`vec(K, λ) = (1-λ)·(d_K − P_row d_K) + λ·P_row d_K`, renormalized to the
canonical working norm `N_REF = 102.076` in every cell (fixed dose).
`λ=0` is the pure out-of-row residual; `λ=1` the pure row(W) projection.

**Compact diagnostic** (SEEDS=4, NTOK=60):

```
K     λ=0       λ=0.5     λ=1
100   0/4       0/4       0/4
200   2/4       1/4       0/4      <- localized transport
500   0/4       0/4       1/4
```

The effect is a **localized island around K=200**, not a generic function
of row-space distance. At K=200, cos(dK, dL_ref) ≈ 0.98-1.0 and replacing
the residual by its row-space projection monotonically suppresses
transport (2/4 → 1/4 → 0/4).

**Full replication** (SEEDS=30, NTOK=120), K=200 only:

```
λ       transport   medMinR   R_row   cos_ref
0.00    4/30        2         0.000   +0.981
0.25    4/30        1         0.004   +0.992
0.50    3/30        0         0.038   +1.000
0.75    6/30        0         0.263   +0.942
1.00    0/30        23        1.000   +0.195
```

**K-window replication.** A fine-grained discovery sweep (SEEDS=4, NTOK=60,
K in 25..500) showed a sharp transition between K=125 and K=150 (held-out
rank: 222 -> 6), followed by a broad operating regime through approximately
K=300:

```
 K      transport  medMinR  R_row  cos->dLfull
 25     0/4        305      0.018  +0.425
 50     0/4        274      0.022  +0.569
 75     0/4        269      0.025  +0.671
100     0/4        262      0.027  +0.754
125     0/4        222      0.030  +0.826
150     1/4          6      0.033  +0.890   <- sharp onset (rank 222 -> 6)
175     0/4          8      0.035  +0.947
200     1/4         11      0.038  +1.000   <- max cos alignment
225     2/4         17      0.041  +0.953
250     1/4         21      0.043  +0.913
300     1/4         25      0.048  +0.849
400     1/4         35      0.057  +0.758
500     0/4         45      0.065  +0.695
```

A preregistered confirmation at K={150,200,250}, SEEDS=30 (NTOK=120)
reproduced the effect: transport occurred in **2/30, 3/30, and 3/30**
seeds respectively, with median best held-out ranks of **1, 0, and 1**.

**Thus K=200 should NOT be characterized as a unique behavioral optimum.**
Rather, it lies near the center of a reproducible sparse operating window
K in 150-250 (behaviorally indistinguishable transport at this sample size),
while also producing the maximum cosine alignment with the reference signal
(cos=1.000). Onset ~K=150 (sharp); dilution begins beyond ~250-300. The
result supports a **sparse-regime interpretation**, not a special fixed K.
Whether the window shifts with prompt/concept/seed is an open question.


**Causal reading (careful, not overclaimed).** The clean claim is *not* a
smooth monotonic dose-response — λ=0.75 peaking at 6/30 is within binomial
noise (n=30). The causal variable is **existence vs. absence of the
out-of-row component**: **λ=1, the pure row-space projection, is the unique
collapsed cell** (0/30, medMinR→23, cos_ref→+0.195), while every λ<1 that
retains a *nonzero* residual transports at ~3-6/30 with medMinR≈0 and
cos_ref≈0.94-1.0.

This is a **behavioral discontinuity at the row-space boundary**, not a
continuous dose-response and not a nonlinearity in the interpolation
(the λ mixture itself is linear; only the transport behavior jumps at
λ=1). Projecting the sparse chosen vector back into `row(W)` (the only
`λ=1` manipulation) crosses out of the reachable set and destroys the
effect; any retained residual fraction preserves it, at a roughly flat
3-6/30 plateau. This is itself a negative control: an *amount* of
out-of-row escape does not predict efficacy — either the coordinate/
ranking pattern escapes `row(W)` (works) or it is pulled fully inside
(fails).

**Net mechanism (updated).**

> The generatively effective object is a sparse (K≈200), positively-masked,
> ranked-magnitude vocabulary contrast vector, applied as a logit offset.
> Its coordinate/ranking pattern is the semantic carrier; escaping `row(W)`
> (so it is not cancelled by the hidden-reachable subspace) is *necessary*;
> but the *amount* of escape, beyond merely nonzero, does not scale efficacy.

**Net mechanism (updated).**

> The generatively effective object is a sparse (K≈200), positively-masked,
> ranked-magnitude vocabulary contrast vector, applied as a logit offset.
> Its coordinate/ranking pattern is the semantic carrier; escaping `row(W)`
> (so it is not cancelled by the hidden-reachable subspace) is *necessary*;
> but the *amount* of escape, beyond merely nonzero, does not scale efficacy.

---

## Part V — Semantic vs. lexical (`neighbor_probe.py`, SEEDS=30)

The K×λ surface and mechanism matrix established *what* the intervention
does. This section asks **whether it transports a concept or merely forces
lexical coordinates**. We hold out three probe classes, all filtered so
their tokens are **NOT** among the boosted top-200 coordinates (zero direct
additive boost):

```
class   UNSTEERED          STEERED
lex     0/30  rank 45      27/30  rank 0      <- boosted coords: huge lift
sem     0/30  rank 188     0/30  rank 172     <- unboosted semantic neigh: flat
unr     18/30 rank 0       16/30 rank 1       <- unboosted unrelated: flat
```

- Positive control (LEX = tokens decoded from the actual boosted top-200):
  emission 0/30 → 27/30, best rank 45 → 0. The harness lifts precisely what
  it directly boosts.
- Semantic neighbors outside the mask (fiend/wraith/ghoul/apparition/
  sorcery/incantation/tyranny, all single-token, all NOT in top-200):
  **0/30 → 0/30, best rank 188 → 172.** No generalization to unboosted
  concept coordinates.
- Unrelated frequency-matched control: unchanged (18/30 → 16/30).

**Net claim (honest).** The technique is *sparse ranked **lexical** logit
steering*: it selectively raises a coordinated set of vocabulary coordinates
and generation enters that lexical region. It does **not** demonstrably
transport a *concept* to unboosted coordinates. This is a cleaner, more
defensible mechanism than semantic transport — and the honest way to describe
the method to a reviewer.

*Files:* `mechanism_matrix.py` (K×λ surface), `neighbor_probe.py` (lexical-vs-semantic).

---

## Part VI — Dynamic vs. static contrast (`dynamic_contrast.py`, SEEDS=30)

Is the intervention a genuine adaptive steering mechanism, or a fixed
lexical bias injected into generation? We compare, keeping K=200, norm,
injection step (SW0=20), sampling, seeds, and token budget identical:

| condition | what changes |
|---|---|
| STATIC | `dL` computed once, applied every step (reference) |
| DYN_PREFIX | recompute target-vs-neutral contrast from live prefix each step |
| DYN_SELF | contrast current self next-token distribution vs neutral each step |
| NONE | no steering (baseline) |

```
condition     transport  medMinR  medMaxR  medDist1  medCosVsStatic
STATIC        3/30        0        0        0.530     (reference)
DYN_PREFIX    1/30        2        2        0.430     +0.955
DYN_SELF      0/30       71        0        0.336     +0.013
NONE (base)   0/30       51        0        0.677     -
```

**Reading.** Recomputed-at-prefix concept contrast is *almost identical* to
the static vector (cos +0.955) and does not improve transport (1/30 vs 3/30).
A self-referential contrast (the model's own next-token distribution) is
*orthogonal* to the concept direction (+0.013) and inert (0/30, rank 71).

**Conclusion.** The successful intervention does **not require dynamic
adaptation**, and the effective intervention behaves as a **stable,
context-insensitive lexical bias under this setup**. Recomputing the same
target-vs-neutral contrast from each prefix yields a direction highly
similar to the static vector (cos 0.955) and does not improve transport; a
self-adaptive contrast derived from the current generation state yields a
nearly orthogonal direction (+0.013) and fails. No dynamic recomputation
reproduces or improves the static result.

*File:* `dynamic_contrast.py`.

---

## Part VII — Generalization: concept/prompt dependence (`generalize.py`)

Is the sparse ranked steering a general technique or specific to the fantasy
contrast? Test A (3 concepts x 2 prompts x {NONE + 5 K}, SEEDS=3, NTOK=40,
baseline-corrected):

```
concept  prompt  NONE base  K=100 150  200  250  300
FANTASY  beach   0/3  r238  0/3   0/3  0/3  1/3  1/3
FANTASY  town    0/3  r53   0/3   2/3  2/3  2/3  1/3   <- clean winner
SPACE    beach   0/3  r143  0/3   0/3  0/3  0/3  0/3   <- dead
SPACE    town    0/3  r40   0/3   0/3  1/3  0/3  0/3   <- dead
PIRATE   beach   1/3! r6    2/3   2/3  1/3  1/3  1/3   <- baseline-contam
PIRATE   town    1/3! r23   0/3   0/3  1/3  0/3  1/3   <- ~baseline
```

- **PIRATE held-out words appear 1/3 unsteered** (ship/captain/sea/gold are
  natural vocabulary) — its apparent transport is baseline contamination.
- **SPACE is genuinely dead** (0/3 baseline, 0/3 steered).
- **FANTASY/town is the clean winner** (0/3 -> 2/3, rank ->0); the beach
  prompt suppresses it -> prompt-dependence.

**Test B (honestly): what — if anything — discriminates?** Vector
diagnostics (DIAG mode) of each contrast:

```
concept   m1 max|topK|/sum   m2 max|topK|/max|all|   R_row(200)   cos(K,200)
FANTASY   0.0073             0.8470                  0.038         1.000
SPACE     0.0083             1.0000                  0.060         1.000
PIRATE    0.0083             1.0000                  0.060         1.000
```

*(reconciled in `metric_reconcile.py`; N_REF-identical rescaling)*

- **m1 (true share of top-K magnitude held by the largest coord) is ~0.008
  for ALL THREE concepts** — none is a single-coordinate spike. The earlier
  interpretation of "maxAbsFrac=1.00" as concentration was WRONG: that
  column was m2, which only measures whether the single most-extreme value
  of the full z-scored contrast happens to fall inside the selected top-200
  (SPACE/PIRATE: yes; FANTASY: no — its global extreme is outside). This is
  a *placement* fact, not a concentration fact.
- R_row and cos do not discriminate either (nearly identical).

**Corrected verdict.** The "single-coordinate spike" reading of the
diagonstics was WRONG (retired in this revision): the true concentration
metric m1 = max|topK|/sum|topK| is ~0.008 for ALL three contrasts, so
SPACE/PIRATE are not one-coordinate structures. What m2 = max|topK|/max|all|
captures is placement (is the contrast's single most-extreme value inside
the selected top-200): FANTASY 0.847 (extreme OUTSIDE), SPACE/PIRATE 1.000
(extreme INSIDE). That *does* separate the winner from the losers in this
set, as does (weakly) R_row (0.038 vs 0.060) — but with only 3 contrasts
these are at most candidate predictors, not validated laws (see Part IX).
The observable behavioral difference (baseline contamination) also
distinguishes PIRATE; Test C probes the causal ranking question directly.

*Files:* `generalize.py` (Test A + DIAG).

---

## Part VIII — Ranking causality (`rank_causality.py`, Test C)

Does the *ranked coordinate structure* have causal leverage, or is any
sparse injection enough? Retention/ablation ladder at SEEDS=3, NTOK=40,
all vectors norm-matched to N_REF:

```
FANTASY (quiet-town prompt):
  NONE           0/3  rank 53    <- baseline
  top1           0/3  rank 141
  top5           0/3  rank 141
  top10          0/3  rank 141
  top20          0/3  rank 141
  top50          0/3  rank 53
  full200        2/3  rank 0     <- works
  full_minus_t1  2/3  rank 0     <- remove LARGEST coord: still works
  full_minus_rd  2/3  rank 0     <- remove random coord: still works
  rand50         0/3  rank 141   <- random coords dead
  top50_shuf     0/3  rank 53    <- shuffled magnitudes dead

PIRATE (beach prompt):
  NONE           1/3  rank 6     <- baseline itself transports (contam.)
  top1..top50    0/3  rank 6
  full200        1/3  rank 3     <- AT baseline, no steering gain
  full_minus_t1  1/3  rank 3
  full_minus_rd  1/3  rank 3
  rand50/shuf    0/3  rank 6
```

**Reading.** For the clean winner (FANTASY):
- full200 beats baseline (2/3 vs 0/3).
- Deleting the single largest coordinate (t1) or a random one does NOT
  reduce transport (2/3) — the signal is distributed over ~200 ranked
  coordinates; no single load-bearing token.
- Small retention (top1..top50) is dead — the sparse ranked structure needs
  the full ~150+ window, matching the K-sweep.
- rand50 / top50_shuf are dead — coordinate identity and magnitude
  association are both causal.

**PIRATE confirms the baseline-contamination diagnosis:** full200 = NONE =
1/3 — no genuine intervention gain to mediate; the apparent transport is
natural vocabulary. Consistent with Part VII.

*File:* `rank_causality.py`.

**Test D — independent 30-seed confirmation** (FANTASY / town / K=200 /
NTOK=40, prespecified winner, baseline NONE, fixed seeds 0-29):

```
condition  transport  median minRank  meanRank
NONE       0/30       160              (stuck 50-400)
full200    5/30       4                9.1
```

```
full200 per-seed (seed: minRank)  [transport on seeds 0,1,3,5,26]
  0: 0   1: 0   2: 0   3: 0   4: 10   5: 0   6: 15   7: 0   8: 4   9: 3
 10:17  11:26  12: 7  13:15  14: 4  15: 3  16: 2  17:36  18:20  19:10
 20:15  21: 2  22:44  23: 3  24: 9  25: 1  26:0   27:17  28: 9  29: 0
```

- **Transport 5/30 vs 0/30** (baseline is a clean null floor). The 5-vs-0
  split has p~0.05 (exact binomial one-sided).
- **Rank distribution shifts massively**: median held-out rank 160 -> 4;
  8/30 seeds reach rank 0 vs 0/30 at baseline. Even non-emitting steered
  seeds lift the target vocabulary to low rank.
- Statistically consistent with the cheap-screen 2-3/30 at SEEDS=3.

**Confirmed clean progression:** contrast diagnostic -> successful concept ->
causal coordinate ranking -> causal magnitude ordering -> distributed
full-window mechanism -> independent 30-seed confirmation (0% baseline vs
**Confirmed clean progression:** contrast diagnostic -> successful concept ->
causal coordinate ranking -> causal magnitude ordering -> distributed
full-window mechanism -> independent 30-seed confirmation (0% baseline vs
16.7% steered, with rank 160->4).

---

## Part IX — Prior-art calibration and novelty statement

Directly relevant prior work was reviewed (3 librarian literature searches,
Sep 2026). Key facts governing how strong a novelty claim we may make:

### 1. Sparse steering per se is NOT novel
Contrastive Activation Addition (Turner et al., arXiv:2308.10248) introduced
contrast-derived steering vectors. Sparse Activation Steering (Bayat et al.,
arXiv:2503.00177), SAE-SSV (He et al., EMNLP'25, 2025.emnlp-main.112), and
CAusal Steering via Sparse Mediation (CAS-BiPO; Doan et al., EACL'26 Finds,
2026.findings-eacl.57) all demonstrate that *some* sparse coordinate or
subspace selection retains most steering effect. "We steer with a subset of
coordinates" is established.

### 2. The flagged "critique" paper is actually a POSITIVE sparse result
The clawRxiv paper the literature review flagged ("Sparse Activation Steering
with Mean Differences", clawrxiv.io/abs/2604.02039, 2026 — non-standard
venue, verify before citing) is **not a critique of sparsification**: it
reports that ENR-selected top-k works, keeping k=64/4096 -> 91.4% of dense
effect, with a k-sweep showing axis-dependent windows (refusal tolerates
k=32; diffuse axes such as formality need k>=256). It does NOT run any of
our three control types (magnitude-shuffle, equal-weight, largest-coordinate
ablation). Its point is that sparse RAW-coordinate steering *preserves*
efficacy while reducing collateral damage.

### 3. The largest overlap risk is a different paper (arXiv:2604.08524)
"What Drives Representation Steering?" (Cheng/Wiegreffe/Manocha, 2026)
sparsifies raw refusal vectors 90-99% and runs **largest-coordinate retention
("bottom-k" baseline)** and **random-dropout** baselines: random dropout
retains refusal-ASR up to ~40% sparsity; bottom-k up to ~80%. This is the
paper a reviewer will invoke against us. Crucially it does NOT run
magnitude-shuffle, equal-weight, or single-largest-coordinate-ablation; and
it keeps top-k (drops the *rest*), the complement of our largest-coordinate
retention test. It is also measured on a behavioral outcome (refusal ASR),
not vocabulary transport.

### 4. The controls that are still unique to this work
Across the reviewed literature (ActAdd Appendix H partial-vector curves;
Subramani et al. 2022 intrinsic dimension; Mayne et al. 2024 SAE-reconstruction
failure; ROAST energy argument arXiv:2602.14143; SWAI token-level shuffle
arXiv:2601.10960; AUSteer; depth-wise/layer-level equal-weight comparisons;
and the `What Drives` paper above) — **no work was found that runs, on a raw
residual-stream contrast vector: (a) magnitude-shuffle at fixed coordinates,
(b) equal-weight vs ranked-weight, or (c) deletion of the single largest
coordinate with survival reporting.** Test C does exactly these. The
ActAdd Appendix H result (non-monotonic partial vectors; ~60-80% of dims
needed; for one prompt 70% of dims beats 100%) is the closest prior
observation of a window-like effect — we must cite and extend it, not ignore
it.

### 5. Metric caveat (honesty)
Our outcome is **vocabulary transport** (held-out words entering generated
text). "Steerable but Not Decodable" (arXiv:2604.02608) shows function
vectors can steer behavior while projecting to *incoherent* token
distributions. So our top-k failure is demonstrated for the vocabulary-
transport metric; whether it generalizes to downstream behavioral metrics is
open. We claim the phenomenon for the metric we measured, no more.

### Novelty statement (defensible)
> To our knowledge, prior work has not established that the efficacy of a
> contrast-derived raw activation steering vector can depend on a
> **distributed magnitude-ranked coordinate window** (~150-300 coords) being
> preserved (top-1..50 failing), nor demonstrated that dependence through
> the coordinated battery of retention, random-coordinate, magnitude-
> shuffling, equal-weight, and largest-coordinate-ablation controls that
> Test C applies — where single-coordinate deletion survives while magnitude-
> shuffling and equal-weighting kill the effect. Sparse steering in SAE/
> learned-mask spaces (SAS/SAE-SSV/CAS-BiPO) and raw-vector sparsification
> (arXiv:2604.08524) are adjacent but operate under different selection
> criteria, outcome metrics, and — critically — without these assignment
> controls.

*Citations:* arXiv:2308.10248, arXiv:2503.00177, 2025.emnlp-main.112,
2026.findings-eacl.57, clawrxiv.io/abs/2604.02039 (verify venue),
arXiv:2604.08524, arXiv:2602.14143, arXiv:2601.10960, arXiv:2604.02608.
Venue verification recommended for all 2026 arXiv preprints before any
external reference.
