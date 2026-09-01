# Sparse ranked logit steering on Qwen2-1.5B — which coordinates actually matter?

A casual, results-first look at vocabulary-logit steering on Qwen2-1.5B: take a contrast of vocab-logit readouts for a topic vs neutral sentences, add the top-200 coordinates (with magnitudes) as a logit offset during generation, and see what actually does the work. Everything below is reproducible from this repo — scripts, per-seed logs, writeup included.

**Writeup (Parts I–IX):** `writeup-orthogonal-complement.md`
**Original logit-space contrast discovery (v4/v5):** `writeup-sentence-concept.md`
**Related (sphere/geometry lineage, the negative/control chapter):** [github.com/ntrillard/transformer-geometry](https://github.com/ntrillard/transformer-geometry)

## The short version

- **Top 1–50 biggest coordinates only? Nothing.** No transport, every time.
- **A distributed window around 150–300 coordinates?** Real transport. Cleanest 30-seed run: **5/30**, median held-out rank **160 → 4**, 8 seeds reach rank 0 (baseline 0/30).
- **Delete the single largest coordinate?** Still works — no single load-bearing coordinate.
- **Random coords, shuffled magnitudes, equal weights?** All dead.
- **Project into row(W) (what the model can actually represent)?** 0/30 — dead. Any nonzero out-of-row component? Some transport. The pure row-space projection is the one reliably-dead cell.
- **Lexical, not semantic:** boosted top-200 words come back (27/30 at rank 0); unboosted semantic neighbors don't move (0/30).

The useful signal isn't the biggest coordinates, and it isn't one magic token — it's a broad, rank-ordered spread where *which coordinates* and *how big each is* both matter. Shuffle the magnitudes → dead. Keep only the big ones → dead. Push it fully into representable space → dead.

## Key numbers

**K window (30-seed confirmation):**

| K | transport | medMinR | cos→dLref |
|---|---:|---:|---:|
| 150 | 2/30 | 1 | +0.890 |
| 200 | 3/30 | 0 | +1.000 |
| 250 | 3/30 | 1 | +0.913 |

Transport appears around K≈150 and persists through K≈300, dilution beyond. K=200 is the max-alignment point (cos→dLref = 1.000), not a unique behavioral optimum.

**K × λ surface (λ = blend toward the row(W) projection; λ=1 = pure row-space):**

| λ | transport | medMinR | R_row | cos_ref |
|---|---:|---:|---:|---:|
| 0.00 | 4/30 | 2 | 0.000 | +0.981 |
| 0.25 | 4/30 | 1 | 0.004 | +0.992 |
| 0.50 | 3/30 | 0 | 0.038 | +1.000 |
| 0.75 | 6/30 | 0 | 0.263 | +0.942 |
| 1.00 | 0/30 | 23 | 1.000 | +0.195 |

Every λ<1 condition keeps a nonzero out-of-row component and shows some transport; the pure row(W) projection shows none in 30 seeds. Rates are low (3–6/30) and don't climb monotonically with the out-of-row component.

**Lexical vs semantic (SEEDS=30):**

| probe | UNSTEERED | STEERED |
|---|---:|---:|
| LEX (boosted top-200) | 0/30, rank 45 | 27/30, rank 0 |
| SEM (unboosted semantic neighbors) | 0/30, rank 188 | 0/30, rank 172 |
| UNR (unboosted unrelated) | 18/30, rank 0 | 16/30, rank 1 |

**Ablation ladder (Test C + Test D):**

| condition | SEEDS=3 | SEEDS=30 (Test D) |
|---|---:|---:|
| NONE (baseline) | 0/3 | 0/30, rank 160 |
| top1/5/10/20/50 | 0/3 | 0/30 (rank ~141–271) |
| full200 | 2/3 | 5/30, rank 4 |
| full200 − largest coord | 2/3 | 4/30 |
| full200 − random coord | 2/3 | 5/30 |
| rand50 / top50_shuf | 0/3 | 0/30 |

**Generalization (Test A, baseline-corrected, SEEDS=3):**

| concept | prompt | NONE base | best steered |
|---|---:|---:|---:|
| FANTASY | town | 0/3 | 2/3 (rank 0) |
| FANTASY | beach | 0/3 | ~0–1/3 |
| SPACE | both | 0/3 | 0/3 (no observed transport) |
| PIRATE | both | 1/3 | ~1/3 (baseline-contaminated) |

Transport is concept- and prompt-dependent; some contrasts show no observed transport. No cheap vector metric cleanly predicts success in this 3-concept set.

Other bits: recomputing the vector per prefix barely moves it (cos vs static +0.955) and doesn't help — acts like a stable lexical bias, not an adaptive controller. Normalization (raw/centered/z/perz) doesn't matter once the top-k coordinates + ranked magnitudes are fixed.

## Files

```
writeup-orthogonal-complement.md   # main writeup, Parts I–IX (the real detail)
writeup-sentence-concept.md        # original logit-space contrast discovery (v4/v5)
mechanism_matrix.py                # causal factorial + K×λ surface (LAMBDA=1 mode)
neighbor_probe.py                  # semantic-vs-lexical probe
dynamic_contrast.py                # static vs adaptive
generalize.py                      # concept/prompt generalization (DIAG mode)
rank_causality.py                  # retention/ablation ladder (COND= env)
metric_reconcile.py                # metric sanity check
decomp_gen.py, subspace_hierarchy.py, vec_compare.py    # decomposition / geometry
falsify_orth{2,3}.py, logit_scan.py                      # falsification line
pivot2_topk{,phaseA}.py, pivot_contrast.py               # sparsity falsification (superseded)
gen_geom.py                        # reference implementation (MENU logit)
```

## Run

```bash
# causal factorial + K×λ surface
SEEDS=30 python3 mechanism_matrix.py 0
LAMBDA=1 KLIST=200 LLIST=0,0.25,0.5,0.75,1 SEEDS=30 python3 mechanism_matrix.py 0

# semantic-vs-lexical probe
SEEDS=30 NTOK=120 python3 neighbor_probe.py

# static vs adaptive
SEEDS=30 python3 dynamic_contrast.py

# generalization
SEEDS=3 NTOK=40 python3 generalize.py

# 30-seed Test D
CONCEPT=FANTASY SEEDS=30 NTOK=40 COND=NONE,full200 python3 rank_causality.py
```

Defaults everywhere: model `Qwen/Qwen2-1.5B`, 30 seeds, nucleus p=0.9.

## Notes

- **Stats status:** exact Fisher tests on per-seed counts. The lexical probe is decisive (27/30 vs 0/30, p<0.001). The λ=0.75 cell is nominal two-sided p=0.024 but not significant after Holm correction across the 4 tested λ cells. Headline window-vs-baseline contrasts (5/30 vs 0/30, two-sided p=0.052; pooled 8/90 vs 0/30, p=0.199) are suggestive, not significant at n=30. Pilot findings on one model family — treat accordingly.
- **Prior art:** sparse steering isn't new (CAA `arXiv:2308.10248`; SAS `arXiv:2503.00177`; SAE-SSV `2025.emnlp-main.112`; CAS-BiPO `2026.findings-eacl.57`). ActAdd Appendix H reports a partial-vector window observation (70% of dims > 100% for one prompt). "What Drives Representation Steering?" (`arXiv:2604.08524`) runs bottom-k (retain-largest-coordinates) and random-dropout baselines with refusal-ASR metrics. "Steerable but Not Decodable" (`arXiv:2604.02608`) shows function-vector steering can work when token projection is incoherent.
- **Transport definition:** generated text contains a held-out target word (stem-matched, case-normalized) and is non-degenerate (no token run ≥6, type/token > 0.6). `medMinR` = median over seeds of the minimum held-out token rank; rank 0 = top token.

**Evaluation metric details and per-seed logs:** see `writeup-orthogonal-complement.md` and the `logs/` referenced there.