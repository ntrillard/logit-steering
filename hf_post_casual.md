# I wanted to see what actually matters in a steering vector, so I broke its coordinates one way at a time

Scratched an itch this week: build a vocabulary-logit steering vector, then find out which coordinates are actually doing the work. Nothing fancy — one small model, a LOT of seeds.

**Setup, in one line:** take a contrast of vocab-logit readouts on a target topic vs neutral sentences (per-token z-scored, averaged, re-z-scored), keep the **top-200 positive coordinates with their ranked magnitudes**, and add that as a logit offset during generation (α=2.0 after step 20, nucleus p=0.9). Qwen2-1.5B, 30 seeds per condition.

**Everything to reproduce:** [github.com/ntrillard/logit-steering](https://github.com/ntrillard/logit-steering) (all code + per-seed logs), writeup in `writeup-orthogonal-complement.md`. Older sphere/geometry lineage: [github.com/ntrillard/transformer-geometry](https://github.com/ntrillard/transformer-geometry).

## The TL;DR

- **Top 1–50 biggest coordinates only? Nothing.** Zero transport, every time.
- **A distributed window around 150–300 coordinates?** Real transport — and the cleanest run (30 seeds) gave 5/30, median held-out rank dropping from 160 → 4, 8 seeds hitting rank 0.
- **Delete the single largest coordinate?** Still works.
- **Random coordinates, shuffled magnitudes, equal weights?** All dead.
- **Project the vector into row(W) (the part the model can actually represent)?** 0/30 — completely dead. Give it *any* out-of-row component? Some transport. The pure row-space projection is the one thing that reliably kills it.
- **It's lexical, not semantic:** the boosted top-200 words come back strong (27/30 at rank 0), but unboosted semantic neighbors don't budge (0/30).

## The key numbers

K-sweep (30-seed confirmations):

| K | transport | medMinR | cos→dLref |
|---|---:|---:|---:|
| 150 | 2/30 | 1 | +0.890 |
| **200** | 3/30 | 0 | **+1.000** |
| 250 | 3/30 | 1 | +0.913 |

λ sweep (blend residual → row(W) projection; λ=1 = pure row-space):

| λ | transport |
|---|---:|
| 0.00 | 4/30 |
| 0.25 | 4/30 |
| 0.50 | 3/30 |
| 0.75 | 6/30 |
| **1.00** | **0/30** |

Probe: lexical vs semantic

| probe | unsteered | steered |
|---|---:|---:|
| LEX (boosted top-200) | 0/30, rank 45 | **27/30, rank 0** |
| SEM (unboosted neighbors) | 0/30, rank 188 | 0/30, rank 172 |

## So what is this?

The useful signal is **not** concentrated in the biggest coordinates, and it's not one magic token either. It looks like a decently broad, rank-ordered spread of coordinates, where which-coordinate *and* how-big-both matter. Shuffle the magnitudes → dead. Keep only the big ones → dead. Project it into representable space → dead.

Also worth knowing:

- It's **stochastic and concept-dependent.** FANTASY/town works, SPACE does nothing (clean null), PIRATE is contaminated at baseline (its held-out words already show up unsteered). Low rates — real but small effect, one model family.
- **Static vector ≈ recomputing it per prefix** (cos +0.955, slightly worse). It's acting like a stable lexical bias, not an adaptive controller.
- Some of this echoes prior work — ActAdd's Appendix H shows partial vectors can beat the full one (70% of dims > 100% for one prompt), and the 2026 "What Drives Representation Steering?" paper runs bottom-k / random-dropout baselines. Not claiming the window is brand-new; the *control battery* (magnitude shuffle, single-largest deletion, equal weight) applied to a raw vocab contrast is the bit I didn't find elsewhere.

That's it. Fun finding, honest limits, all numbers checkable from the repo. Questions welcome.