# Sparse ranked logit steering on Qwen2-1.5B: which coordinates actually matter?

Results from a causal investigation of vocabulary-logit steering on Qwen2-1.5B. All code, scripts, and per-seed logs are in the repository; the numbers below are directly reproducible from it.

**Repository:** [github.com/ntrillard/logit-steering](https://github.com/ntrillard/logit-steering) — writeup in `writeup-orthogonal-complement.md` (Parts I–IX)
**Related (sphere/geometry lineage, negative controls):** [github.com/ntrillard/transformer-geometry](https://github.com/ntrillard/transformer-geometry)
**Method details:** [writeup-orthogonal-complement.md](https://github.com/ntrillard/logit-steering/blob/master/writeup-orthogonal-complement.md)

## Setup

- Model: Qwen2-1.5B (bf16). Contrast of vocabulary-logit readouts on target-topic vs neutral sentences, per-token z-scored, averaged, re-z-scored.
- Steering vector `dL = zscore(mean_tgt − mean_neu) · top200` (top-200 positive coordinates with ranked magnitudes), norm-matched; applied as a logit offset (`α=2.0`, after step 20, nucleus sampling `p=0.9`).
- **Transport** = generated text contains a held-out target word (stem-matched, case-normalized) and is non-degenerate (no token run ≥6, type/token > 0.6). `medMinR` = median over seeds of the minimum held-out token rank; rank 0 = top token. 30 seeds per condition unless noted.

## Results

**1. K window (30-seed confirmation):**

| K | transport | medMinR | cos→dLref |
|---|---:|---:|---:|
| 150 | 2/30 | 1 | +0.890 |
| 200 | 3/30 | 0 | +1.000 |
| 250 | 3/30 | 1 | +0.913 |

Transport appears around K≈150 and persists through K≈300; dilution beyond. K=200 is the maximum-alignment point (cos→dLref = 1.000), not a unique behavioral optimum.

**2. K × λ causal surface (λ = blend from residual toward the row(W) projection):**

| λ | transport | medMinR | R_row | cos_ref |
|---|---:|---:|---:|---:|
| 0.00 | 4/30 | 2 | 0.000 | +0.981 |
| 0.25 | 4/30 | 1 | 0.004 | +0.992 |
| 0.50 | 3/30 | 0 | 0.038 | +1.000 |
| 0.75 | 6/30 | 0 | 0.263 | +0.942 |
| 1.00 | 0/30 | 23 | 1.000 | +0.195 |

Every λ<1 condition retains a nonzero out-of-row component and shows some transport; the pure row(W) projection (λ=1) shows none in 30 seeds. Absolute rates are low (3–6/30) and do not increase monotonically with the out-of-row component.

**3. Causal factorial (SEEDS=6):** `rand200` / `magmatch200` / `shuffle200` / `equal200` / `rowW_proj` → 0/6 (no transport). `raw_t200` / `perz_t200` (correct coordinates × ranked magnitudes × out-of-row) → 2/6. Normalization (raw/centered/z/perz) did not materially change results once top-k coordinates + ranked magnitudes were fixed.

**4. Lexical vs semantic (SEEDS=30):**

| probe | UNSTEERED | STEERED |
|---|---:|---:|
| LEX (boosted top-200) | 0/30, rank 45 | 27/30, rank 0 |
| SEM (unboosted semantic neighbors) | 0/30, rank 188 | 0/30, rank 172 |
| UNR (unboosted unrelated) | 18/30, rank 0 | 16/30, rank 1 |

Effect is lexical forcing, not semantic transport.

**5. Static vs adaptive (SEEDS=30):** STATIC 3/30; DYN_PREFIX (recompute per prefix) 1/30 (cos vs static +0.955); DYN_SELF 0/30 (cos +0.013). Recomputation barely changes the vector and does not help.

**6. Ranking causality / ablations (Test C + Test D):**

| condition | SEEDS=3 | SEEDS=30 (Test D) |
|---|---:|---:|
| NONE (baseline) | 0/3 | 0/30, rank 160 |
| top1/5/10/20/50 | 0/3 | 0/30 (rank ~141–271) |
| full200 | 2/3 | 5/30, rank 4 |
| full200 − largest coord | 2/3 | 4/30 |
| full200 − random coord | 2/3 | 5/30 |
| rand50 / top50_shuf | 0/3 | 0/30 |

- Test D: 5/30 transport vs 0/30 baseline, median held-out rank 160 → 4, 8/30 seeds reach rank 0 (baseline 0/30).
- No single load-bearing coordinate: deleting the single largest coordinate survives.
- The observation depends on both coordinate identity and the coordinate↔ranked-magnitude association.

**7. Generalization (Test A, baseline-corrected, SEEDS=3):**

| concept | prompt | NONE base | best steered |
|---|---:|---:|---:|
| FANTASY | town | 0/3 | 2/3 (rank 0) |
| FANTASY | beach | 0/3 | ~0–1/3 |
| SPACE | both | 0/3 | 0/3 (no observed transport) |
| PIRATE | both | 1/3 | ~1/3 (baseline-contaminated) |

Transport is concept- and prompt-dependent; some contrasts show no observed transport. No cheap vector metric cleanly predicts success in this 3-concept set.

## Notes

- Statistical status: exact Fisher tests on per-seed counts. LEX 27/30 vs 0/30 is decisive (p<0.001). The λ=0.75 cell is nominal two-sided p=0.024 but not significant after Holm correction across the 4 tested λ cells. Headline window-vs-baseline contrasts (5/30 vs 0/30, two-sided p=0.052; pooled 8/90 vs 0/30, p=0.199) are suggestive, not significant at n=30. Pilot findings on one model family.
- Prior art: sparse steering is not new (CAA, arXiv:2308.10248; SAS, arXiv:2503.00177; SAE-SSV, 2025.emnlp-main.112; CAS-BiPO, 2026.findings-eacl.57). ActAdd Appendix H reports a partial-vector window observation (70% of dims > 100% for one prompt). "What Drives Representation Steering?" (arXiv:2604.08524) runs bottom-k (retain-largest-coordinates) and random-dropout baselines with refusal ASR metrics. "Steerable but Not Decodable" (arXiv:2604.02608) shows function-vector steering can work when token projection is incoherent.

**Reproduce:** all commands in the repository README; scripts `mechanism_matrix.py`, `neighbor_probe.py`, `dynamic_contrast.py`, `generalize.py`, `rank_causality.py` with env-configured SEEDS/K/LAMBDA/COND.