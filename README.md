# Logit Steering — Sparse Ranked Vocabulary-Coordinate Control

Causal investigation of **sparse ranked logit steering** in a small LM
(Qwen2-1.5B): a semantic contrast becomes generatively effective when it is
restricted to a sparse set of vocabulary coordinates (~K=200) while preserving
their relative logit ordering and magnitude.

## Scientific lineage

This repo is the **technique repo**. The mechanism was discovered through a
**sphere/geometry investigation** (`github.com/ntrillard/transformer-geometry`
— preserved intact as the historical/control line, not rewritten). The
trajectory:

```
sphere hidden-direction steering  ->  falsified as the mechanism
      |
      +--> logit-space contrast (v4/v5, gen_geom)  ->  works
                 |
                 +--> causal dissection: sparse coords x ranked mags x row(W)-escape
                 |         |
                 |         +--> K x lambda causal surface (behavioral discontinuity
                 |              at the row-space boundary; lambda=1 unique dead cell)
                 |
                 +--> semantic-vs-lexical probe: LEXICAL forcing, not semantic
                      transport (neighbor_probe.py)
```

The sphere work is the **discovery/negative-control chapter**; the logit
intervention is the technique this repo characterizes.

## The claim

> The generatively effective operation is: take the vocabulary readout
> contrast, **select its top-200 positive coordinates with their ranked
> magnitudes, and apply the resulting sparse vector as a logit offset**.
> This vector necessarily escapes `row(W)` (masking breaks representability),
> and it is that *coordinated* sparse object — not any individual axis
> (normalization, sparsity, magnitudes, row-escape) — that transports.

**Honest scope:** the intervention is *sparse ranked **lexical** logit
steering*. It selectively raises a coordinated set of vocabulary coordinates;
generation enters that lexical region. It does **not** demonstrably transport
a concept to unboosted coordinates (`neighbor_probe.py` negative).

## Negative/control chapters

- The hidden-sphere direction: falsified as the mechanism — see
  `writeup-orthogonal-complement.md` (Part I–V).
- `row(W)`-escape: *necessary but not sufficient* — rand200/shuffle200/
  equal200 all sit ~96% out of `row(W)` yet fail.
- Behavioral discontinuity at the row-space boundary: the pure row(W)
  projection is the unique dead cell (0/30) in the K×λ causal surface.

## Layout

```
writeup-orthogonal-complement.md   # main mechanism writeup (Parts I–V)
writeup-sentence-concept.md        # orig. logit-space contrast discovery (v4/v5)
mechanism_matrix.py                # causal factorial + K×λ surface  (LAMBDA=1 mode)
neighbor_probe.py                  # semantic-vs-lexical probe (Part V)
decomp_gen.py                      # causal decomposition generator
subspace_hierarchy.py              # representability boundary
vec_compare.py                     # vector geometry (cos/spearman/overlap)
falsify_orth{2,3}.py, logit_scan.py # closed falsification line
pivot2_topk{,phaseA}.py, pivot_contrast.py # sparsity falsification (superseded)
gen_geom.py                        # working reference implementation (MENU logit)
```

## Run

```bash
# causal factorial + K×λ surface
SEEDS=30 python3 mechanism_matrix.py 0
LAMBDA=1 KLIST=200 LLIST=0,0.25,0.5,0.75,1 SEEDS=30 python3 mechanism_matrix.py 0

# semantic-vs-lexical probe
SEEDS=30 NTOK=120 python3 neighbor_probe.py
```

## Key tables

**K-sweep** (SEEDS=6): transport localized at K≈200 — K<100 dead,
K=200 canonical, K>200 dilutes (cos→dLref falls 0.695→0.131).

**K×λ causal surface** (SEEDS=30, K=200):

| λ | transport | medMinR | R_row | cos_ref |
|---:|---:|---:|---:|---:|
| 0.00 | 4/30 | 2 | 0.000 | +0.981 |
| 0.25 | 4/30 | 1 | 0.004 | +0.992 |
| 0.50 | 3/30 | 0 | 0.038 | +1.000 |
| 0.75 | 6/30 | 0 | 0.263 | +0.942 |
| 1.00 | 0/30 | 23 | 1.000 | +0.195 |

**Semantic vs lexical** (SEEDS=30):

| probe | UNSTEERED | STEERED |
|---|---:|---:|
| LEX (boosted) | 0/30, rank 45 | 27/30, rank 0 |
| SEM (unboosted semantic) | 0/30, rank 188 | 0/30, rank 172 |
| UNR (unboosted unrelated) | 18/30, rank 0 | 16/30, rank 1 |

---
N. Trillard — September 2026.