#!/usr/bin/env python3
"""rank_causality.py — Test C: does the RANKED coordinate structure have
causal leverage?

The diagnostic (Test B) says the useful property of a contrast is distributed
ranked mass (FANTASY maxAbsFrac=0.85) vs. a single-coordinate spike (SPACE /
PIRATE maxAbsFrac=1.00). This script makes that causal:

Retention sweep  : top1, top5, top10, top20, top50, full200
                   (keep ONLY the top-K positive coordinates, rescale to N_REF)
Ablation         : full200_minus_top1  (remove the single largest coordinate;
                   should KILL a concentrated contrast, barely touch a
                   distributed one)
Controls         : rand50 (random coords, own values), top50_shuf (right
                   coords, permuted values), NONE baseline.

Prediction for a DISTRIBUTED contrast (FANTASY):
  full200 ~ 2/3 ; top50/20/10/5/1 weak->dead (K window); full_minus_top1
  still works (one coord of 200 is dispensable); rand50/top50_shuf dead.

Prediction for a CONCENTRATED contrast (PIRATE):
  full ~ baseline-contaminated; top1 ~ full (it IS the whole contrast);
  full_minus_top1 -> dead (removing the dominant coordinate removes the
  contrast).

Run (cheap screen):
  CONCEPT=FANTASY SEEDS=3 NTOK=40 python3 rank_causality.py
  CONCEPT=PIRATE  SEEDS=3 NTOK=40 python3 rank_causality.py
"""
import os, re, time
from collections import Counter
import torch, transformers

SEEDS   = int(os.environ.get('SEEDS', '3'))
NTOK    = int(os.environ.get('NTOK', '40'))
NUCLEUS = float(os.environ.get('NUCLEUS', '0.9'))
ALPHA   = float(os.environ.get('ALPHA', '2.0'))
SW0     = int(os.environ.get('SW0', '20'))
K       = int(os.environ.get('K', '200'))
MODEL   = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
CONCEPT = os.environ.get('CONCEPT', 'FANTASY')
SEEDBASE = int(os.environ.get('SEEDBASE', '0'))

# map to the same concept cells as generalize.py
CONCEPTS = {
    'FANTASY': dict(
        PROMPT='The sun rose over the quiet town',   # clean-winner prompt
        TGT=('A dragon circled the ruined towers of the ancient kingdom|'
             'A knight drew his sword against the fire-breathing beast|'
             "The wizard's spell shattered the castle gates"),
        NEU=('The waves crashed gently on the beach|The sand was cool to the '
             'touch|The sun was warm over the water'),
        HELD_OUT='creature creatures evil monsters beast horror lurking nightmare demons'),
    'PIRATE': dict(
        PROMPT='The waves crashed gently on the beach',  # PIRATE's stronger cell
        TGT=('The pirate captain sailed his ship across the sea|'
             'The crew buried the chest of gold on the island|'
             'The sailor spotted a ship on the horizon'),
        NEU=('The sheep grazed in the meadow|The cows walked to the barn|'
             'The farmer harvested the wheat'),
        HELD_OUT='ship captain crew treasure island gold sail sea loot'),
}

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    t0 = time.time()
    cfg = CONCEPTS[CONCEPT]
    print(f'rank_causality CONCEPT={CONCEPT} prompt={cfg["PROMPT"]!r} '
          f'SEEDS={SEEDS} NTOK={NTOK}', flush=True)

    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    norm = model.model.norm
    V = model.lm_head.weight.shape[0]
    eos_id = int(tok.eos_token_id)

    def logits_of(s):
        t = tok(s, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        with torch.no_grad():
            return model(t).logits[0, -1, :].float().cpu()

    tgt = [x.strip() for x in cfg['TGT'].split('|') if x.strip()]
    neu = [x.strip() for x in cfg['NEU'].split('|') if x.strip()]
    nm = torch.stack([logits_of(s) for s in neu]).mean(0)

    perz_sum = None
    for s in tgt:
        Ls = logits_of(s)
        c = Ls - nm
        c = (c - c.mean()) / (c.std() + 1e-6)
        perz_sum = c if perz_sum is None else perz_sum + c
    perz = perz_sum / max(1, len(tgt))
    perz = (perz - perz.mean()) / (perz.std() + 1e-6)

    # diagnostic: share of top-K magnitude held by the single largest coord
    topK_idx = perz.argsort(descending=True)[:K]
    maf = (perz[topK_idx].abs().max() / perz[topK_idx].abs().sum()).item()
    print(f'  maxAbsFrac (top-{K}) = {maf:.3f}   '
          f'({"distributed" if maf < 0.95 else "single-coordinate spike"})', flush=True)

    def rescale(v):
        n = v.norm()
        return v * (102.076 / n) if n > 1e-9 else v  # N_REF = canonical dose

    top_idx = perz.argsort(descending=True)
    full = torch.zeros(V); full[top_idx[:K]] = 1.0; full = perz * full

    def topk_retain(k, positive=True):
        sel = top_idx[:k] if positive else top_idx[-k:]
        v = torch.zeros(V); v[sel] = perz[sel]
        return rescale(v)

    def ablation(idx):
        v = full.clone(); v[idx] = 0.0
        return rescale(v)

    # build a specific condition vector on demand (lazy) so only one full-dim
    # vector is live during generation - avoids the RAM blowup that stalled
    # the first run when all 11 were materialized at once.
    gi = torch.Generator().manual_seed(42 + 0)
    ridx = torch.randperm(V, generator=gi)[:50]
    perm = torch.randperm(50, generator=torch.Generator().manual_seed(77))
    v50 = perz[top_idx[:50]]

    def cond_vec(name):
        if name == 'NONE':
            return None
        if name == 'full200':
            v = torch.zeros(V); v[top_idx[:K]] = perz[top_idx[:K]]
        elif name == 'top50_shuf':
            v = torch.zeros(V); v[top_idx[:50]] = v50[perm]
        elif name == 'top1' or name == 'top5' or name == 'top10' or name == 'top20' or name == 'top50':
            k = int(name[3:])
            v = torch.zeros(V); v[top_idx[:k]] = perz[top_idx[:k]]
        elif name == 'full_minus_t1':
            v = torch.zeros(V); v[top_idx[:K]] = perz[top_idx[:K]]
            v[top_idx[0]] = 0.0
        elif name == 'full_minus_rd':
            v = torch.zeros(V); v[top_idx[:K]] = perz[top_idx[:K]]
            v[int(ridx[0])] = 0.0
        elif name == 'rand50':
            v = torch.zeros(V); v[ridx] = perz[ridx]
        elif name == 'top50_shuf':
            v = torch.zeros(V); v[top_idx[:50]] = v50[perm]
        else:
            raise ValueError(name)
        return rescale(v)

    h_ids = set()
    for w in cfg['HELD_OUT'].split():
        sp = tok(' ' + w, add_special_tokens=False).input_ids
        if len(sp) == 1: h_ids.add(int(sp[0]))

    def score(txt):
        toks = [w.strip('.,!?;:()[]"\'').lower() for w in re.findall(r"\S+", txt)]
        if not toks: return 0, 0.0
        maxr = cur = 0; prev = None; cnt = Counter()
        for t in toks:
            cnt[t] += 1
            if t == prev: cur += 1; maxr = max(maxr, cur)
            else: cur = 1; prev = t
        dist1 = len(cnt) / len(toks)
        low = txt.lower()
        h = sum(1 for w in cfg['HELD_OUT'].split() if re.search(r'\b' + w + r'\w*', low))
        return h, (h >= 1 and maxr < 6 and dist1 > 0.6)

    def gen(vec, seed):
        torch.manual_seed(seed)
        ids = tok(cfg['PROMPT'], add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        past = None; out_ids = []; minr = 10**9
        dd = None if vec is None else vec.to(DEV)
        with torch.no_grad():
            for step in range(NTOK):
                vc = {}
                hk = norm.register_forward_hook(lambda m, i, o: vc.__setitem__('o', o[0, -1, :].float().clone()))
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
                hk.remove()
                if past is None: past = out.past_key_values
                L0 = out.logits[0, -1, :].float()
                on = (step >= SW0)
                L1 = L0 + (dd if (on and dd is not None) else 0.0)
                order1 = L1.argsort(descending=True).tolist()
                pos1 = {tid: k for k, tid in enumerate(order1)}
                if h_ids: minr = min(minr, min(pos1[i] for i in h_ids))
                p = torch.softmax(L1, 0)
                q = p.clone(); ooo = q.argsort(descending=True)
                kk = int((q[ooo].cumsum(0) <= NUCLEUS).sum()) + 1
                msk = torch.zeros_like(q); msk[ooo[:kk]] = 1
                qq = (q * msk); qq = qq / qq.sum()
                nxt = int(torch.multinomial(qq, 1))
                if nxt == eos_id: break
                out_ids.append(nxt)
                ids = torch.tensor([[nxt]], device=DEV)
        txt = tok.decode(out_ids)
        s_h, ok = score(txt)
        return ok, minr

    for name in ['NONE', 'top1', 'top5', 'top10', 'top20', 'top50',
                 'full200', 'full_minus_t1', 'full_minus_rd', 'rand50', 'top50_shuf']:
        R = 0; mrs = []
        vec = cond_vec(name)   # lazy: only one full-dim vector live at a time
        for s in range(SEEDS):
            ok, mr = gen(vec, SEEDBASE + s)
            R += int(ok); mrs.append(mr)
        del vec
        med = sorted(mrs)[SEEDS // 2] if mrs else -1
        tag = '  <- baseline' if name == 'NONE' else ''
        print(f'  {name:>14} {R:3d}/{SEEDS:<4} {med:8d}{tag}', flush=True)
    only = [x.strip() for x in os.environ.get('COND', '').split(',') if x.strip()]
    names = only if only else ['NONE', 'top1', 'top5', 'top10', 'top20', 'top50',
                               'full200', 'full_minus_t1', 'full_minus_rd', 'rand50', 'top50_shuf']
    for name in names:
        R = 0; mrs = []; per_seed = []
        vec = cond_vec(name)   # lazy: only one full-dim vector live at a time
        for s in range(SEEDS):
            ok, mr = gen(vec, SEEDBASE + s)
            R += int(ok); mrs.append(mr); per_seed.append((SEEDBASE+s, ok, mr))
        del vec
        med = sorted(mrs)[SEEDS // 2] if mrs else -1
        mean = sum(mrs)/max(1,len(mrs))
        tag = '  <- baseline' if name == 'NONE' else ''
        print(f'  {name:>14} {R:3d}/{SEEDS:<4} {med:8d}  meanR {mean:7.1f}{tag}', flush=True)
        if only:
            print(f'    per-seed (seed, transport, minRank):', flush=True)
            for (sd, ok_, mr_) in per_seed:
                print(f'      {sd:>3}  {int(ok_)}  {mr_:>8}', flush=True)


if __name__ == '__main__':
    main()