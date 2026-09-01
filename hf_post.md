---
license: cc-by-4.0
language:
  - en
tags:
  - steering-vectors
  - activation-steering
  - logit-steering
  - interpretability
  - mechanistic-interpretability
  - causal-analysis
  - llm-control
pipeline_tag: text-generation
model_type: transformer
base_model: Qwen/Qwen2-1.5B
inference: false
library_name: transformers
---

# Beyond Top-K: Distributed Ranked Structure in Contrastive Logit Steering

**Author:** N. Trillard — **September 2026**

Steering vectors computed for the **`Qwen/Qwen2-1.5B`** causal language model
(bf16, 152k vocabulary, hidden dim 1536, RMSNorm). These are the exact
`dL = zscore(mean_tgt − mean_neu) · top200` vectors used in the
experiments whose results are documented below and in the
[GitHub repository](https://github.com/ntrillard/logit-steering). During
generation, the stored vector is applied with `α = 2.0`, after step
`SW0 = 20`, with nucleus sampling `p = 0.9`.

---

## Links

| Asset | URL |
|---|---|
| Technique repo (code + writeup) | https://github.com/ntrillard/logit-steering |
| Historical / negative-control lineage (sphere investigation) | https://github.com/ntrillard/transformer-geometry |
| Base model (this card's vectors steer it) | https://huggingface.co/Qwen/Qwen2-1.5B |
| Main writeup (Parts I–IX) | [writeup-orthogonal-complement.md](https://github.com/ntrillard/logit-steering/blob/master/writeup-orthogonal-complement.md) |

**Base model** used throughout: `Qwen/Qwen2-1.5B`
([model card](https://huggingface.co/Qwen/Qwen2-1.5B)); earlier discovery work
also used the Instruct variant `Qwen/Qwen2-1.5B-Instruct`.

---

## What this is

A target-vs-neutral **vocabulary-logit contrast** — per-sentence z-scored,
averaged, re-z-scored, **positive-masked to its top-200 ranked coordinates**,
scaled to its working norm `N_REF ≈ 102` — added as a logit offset during
generation (`α = 2.0`, applied after step `SW0 = 20`, nucleus p = 0.9).

**The claim (supported, honest scope):**

> In the tested setting, effective vocabulary transport appears to depend
> on retaining a distributed, magnitude-ranked window of coordinates from the
> vocabulary-logit contrast.

> **In the tested FANTASY/town setting, retaining only the top 1–50 magnitude
> coordinates fails, while a distributed ranked window of roughly 150–300
> coordinates succeeds.**

The apparent paradox is causal: **removing the largest coordinate does not
remove the effect, while retaining only the largest coordinates does**.
These results rule out a single load-bearing coordinate and disfavor several
simple sparsification explanations in the tested setting. They are consistent
with the effect depending on the **distributed ranked structure of the
window**.

**Honest limits:** the intervention is **sparse ranked lexical logit
steering** — it forces the selected vocabulary coordinates; it does **not**
demonstrably transport a concept to unboosted coordinates, and it behaves as
a **largely stable lexical bias under the tested static and dynamic
recomputation procedures** (dynamic recomputation gives no
gain).

---

## Key results (all on Qwen2-1.5B, SEEDS = 30 unless noted)

### 1. Causal factorial (mechanism matrix, SEEDS = 6)
`rand200` / `magmatch200` / `shuffle200` / `equal200` / `rowW_proj`: **0/6**
(no observed transport).
`raw_t200` / `perz_t200` (correct coordinates × ranked magnitudes × out-of-row):
**2/6** (transport). Normalization (raw/centered/z/perz) **did not
materially change the observed result** once the top-k coordinates and
their ranked magnitudes were fixed.

### 2. K shows a sparse operating window, not a unique optimum
Fine sweep (SEEDS=4, NTOK=60): in the tested sweep, efficacy appeared to
emerge around **K≈150** (held-out rank 222→6) and persist through **K≈300**,
with dilution at larger K. 30-seed confirmation:

| K | transport | medMinR | R_row | cos→dLref |
|---|---:|---:|---:|---:|
| 150 | 2/30 | 1 | 0.033 | +0.890 |
| **200** | 3/30 | 0 | 0.038 | **+1.000** |
| 250 | 3/30 | 1 | 0.043 | +0.913 |

K=200 has the highest alignment with the reference vector in this sweep;
it is not a unique behavioral optimum.

### 3. K × λ surface: behavioral discontinuity at the row-space boundary

| λ | transport | medMinR | R_row | cos_ref |
|---|---:|---:|---:|---:|
| 0.00 | 4/30 | 2 | 0.000 | +0.981 |
| 0.25 | 4/30 | 1 | 0.004 | +0.992 |
| 0.50 | 3/30 | 0 | 0.038 | +1.000 |
| 0.75 | 6/30 | 0 | 0.263 | +0.942 |
| **1.00** | **0/30** | **23** | 1.000 | +0.195 |

The pure row(W) projection (λ=1) is the **only tested dead cell** in this
λ sweep; every λ<1 condition retained a nonzero out-of-row component, and
each produced at least some observed transport in this 30-seed run. In this
experiment, complete removal of the out-of-row component coincided with
complete loss of observed transport; the magnitude of the remaining
out-of-row component did not predict efficacy. **Note on reading
this table:** the behavioral counts (3–6/30 for λ<1) are stochastic and
effectively flat — the clean, reproducible invariant is the **geometric
boundary**, not monotonic efficacy in λ.

### 4. Semantic vs. lexical (neighbor_probe)

| probe | UNSTEERED | STEERED |
|---|---:|---:|
| LEX (boosted top-200 coords) | 0/30, rank 45 | **27/30, rank 0** |
| SEM (unboosted semantic neighbors) | 0/30, rank 188 | 0/30, rank 172 |
| UNR (unboosted unrelated) | 18/30, rank 0 | 16/30, rank 1 |

**Lexical forcing, not semantic transport.**

### 5. Static vs. adaptive (dynamic_contrast)

| condition | transport | medMinR | cos vs static |
|---|---:|---:|---:|
| STATIC | 3/30 | 0 | (reference) |
| DYN_PREFIX (recompute per prefix) | 1/30 | 2 | **+0.955** |
| DYN_SELF (self next-token contrast) | 0/30 | 71 | **+0.013** |

The recomputed contrast barely moves and does not help ⇒ stable lexical bias.

### 6. Ranking causality (retention/ablation ladder) + 30-seed confirmation

| condition | transport (SEEDS=3) | SEEDS=30 (Test D) |
|---|---:|---:|
| NONE (baseline) | 0/3 | **0/30, rank 160** |
| top1 / 5 / 10 / 20 / 50 | 0/3 | 0/30 (rank ~141–271) |
| **full200** | **2/3** | **5/30, rank 4** |
| full200 − largest coord | 2/3 | 4/30 |
| full200 − random coord | 2/3 | 5/30 |
| rand50 / top50_shuf | 0/3 | 0/30 |

- No single load-bearing token (deleting the **single largest** coordinate survives).
- Small top-k retention, random coordinates, and **shuffled magnitudes** all fail
  ⇒ **the observed effect depends on both coordinate identity and the
    association between coordinates and their ranked magnitudes, in the
    tested setting.**
- Test D: **5/30 transport vs. 0/30 baseline**; median held-out rank
  **160 → 4**; **8/30** seeds reach rank 0 (vs. 0/30 baseline).

### 7. Generalization (Test A, baseline-corrected, SEEDS=3)

| concept | prompt | NONE base | best steered |
|---|---:|---:|---:|
| FANTASY | town | 0/3 | **2/3** (K=150–250, rank 0) |
| FANTASY | beach | 0/3 | ~0–1/3 |
| SPACE | both | 0/3 | **0/3** (no observed transport) |
| PIRATE | both | **1/3** | ~1/3 (baseline-contaminated) |

Transport is **concept- and prompt-dependent**; some contrasts are immune.
An earlier "single-coordinate spike" distinguisher was **discarded after
`metric_reconcile.py` failed to support it**; no cheap vector metric cleanly
predicts success at K=200 in this 3-concept set.

---

## Novelty calibration (from 3 targeted literature searches, Sep 2026)

- **Not novel:** sparse steering itself — CAA (Turner et al., arXiv:2308.10248);
  SAS (Bayat et al., arXiv:2503.00177); SAE-SSV (He et al., EMNLP'25,
  2025.emnlp-main.112); CAS-BiPO (Doan et al., EACL'26 Finds,
  2026.findings-eacl.57).
- **Adjacent / must-cite:** ActAdd **Appendix H** (non-monotonic partial-vector
  retention; for one prompt 70% of dims > 100% — closest prior window
  observation); **arXiv:2604.08524** ("What Drives Representation Steering?",
  raw-vector sparsification with largest-coordinate-retention and random-dropout
  baselines — biggest overlap risk; measures refusal-ASR, not vocabulary
  transport, and runs none of the assignment controls below).
- **Potentially novel in the tested setting:** *top-1..50
  magnitude-coordinate retention of a raw contrast vector fails while a
  distributed ranked ~150–300-coordinate window succeeds*, with
  magnitude-shuffle, equal-weight, and largest-coordinate-ablation controls.
  To our knowledge, the combination of these controls applied specifically
  to raw contrast vectors and vocabulary transport has not been directly
  tested in the cited prior work.
- **Metric caveat:** results are for **vocabulary transport**. "Steerable but
  Not Decodable" (arXiv:2604.02608) shows steering can work while token
  projection is incoherent; generalization to behavioral metrics is open.

**Not claimed:** a validated universal screening law (generalization tested
on only 3 contrasts), a robust high-yield steering *algorithm* (5/30
absolute transport rate), or semantic
(concept-level) transport.

---

## Files in this posting

- `dL_fantasy_top200.pt` — the working contrast vector for the FANTASY/town
  cell used in all headline results (α=2.0, N_REF≈102).
- `dL_*_top200.pt` / `dL_*_topK.pt` — additional concept vectors (SPACE, PIRATE,
  royal) and K-window variants (150/200/250/300), norm-matched to N_REF.
- `README.md` (this card), `writeup-orthogonal-complement.md`, and the raw
  per-seed logs (`logs/*`) so every number above is checkable.

## Reproduce

```bash
git clone https://github.com/ntrillard/logit-steering
cd logit-steering

SEEDS=30 python3 mechanism_matrix.py 0                                  # causal factorial
LAMBDA=1 KLIST=200 LLIST=0,0.25,0.5,0.75,1 SEEDS=30 python3 mechanism_matrix.py 0
SEEDS=30 NTOK=120 python3 neighbor_probe.py                             # lexical-vs-semantic
SEEDS=30 python3 dynamic_contrast.py                                    # static vs adaptive
SEEDS=3 NTOK=40 python3 generalize.py                                   # concepts × prompts
CONCEPT=FANTASY SEEDS=30 NTOK=40 COND=NONE,full200 python3 rank_causality.py  # Test D

# model: Qwen/Qwen2-1.5B (HF transformers, bf16, device_map=auto, no quantization)
```

---

## Scope & license

This is a research finding (mechanistic analysis of steering vector
coordinate structure), not a production safety control. It demonstrates a
reproducible, causally constrained phenomenon on one model family; it is not
a general steering law. CC BY 4.0.