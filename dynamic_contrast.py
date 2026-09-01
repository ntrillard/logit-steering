#!/usr/bin/env python3
"""dynamic_contrast.py — STATIC vs DYNAMIC logit contrast.

Question: is the successful K=150-250 intervention a genuine adaptive
steering technique (recompute the contrast as context evolves), or a fixed
lexical bias injected into generation?

Design (same harness as mechanism_matrix.py, canonical fantasy contrast):

  STATIC   : dL = ALPHA * perz * top200 computed ONCE from target sentences;
             applied unchanged every step after SW0 (the reference method).

  DYNAMIC-prefix : at every step, recompute the perz contrast from the model's
             CURRENT next-token logits on the target concept sentences vs the
             neutral reference (i.e. re-run the contrast at the live prefix).

  DYNAMIC-probe  : at every step, recompute a contrast between the model's
             current self next-token distribution (no target sentences) and
             the neutral reference, then top-200 z-score mask, scaled to
             N_REF. This isolates 'context-adaptive' from 'concept-adaptive'.

Metrics per condition (SEEDS runs): transport/N, median best held-out rank,
median dLogP_H, plus a cosine-of-perturbation series (how much the dynamic
vector rotates across steps -> adaptivity).

Run:
  SEEDS=30 NTOK=120 python3 dynamic_contrast.py
"""
import os, sys, time, re
from collections import Counter
import torch, transformers

SEEDBASE = int(os.environ.get('SEEDBASE', '0'))
SEEDS    = int(os.environ.get('SEEDS', '30'))
NTOK     = int(os.environ.get('NTOK', '120'))
NUCLEUS  = float(os.environ.get('NUCLEUS', '0.9'))
ALPHA    = float(os.environ.get('ALPHA', '2.0'))
SW0      = int(os.environ.get('SW0', '20'))
K        = int(os.environ.get('K', '200'))
MODEL    = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
PROMPT   = os.environ.get('PROMPT', 'The waves crashed gently on the beach')

HELD_OUT = 'creature creatures evil monsters beast horror lurking nightmare demons'

TGT = ('A dragon circled the ruined towers of the ancient kingdom|'
       'A knight drew his sword against the fire-breathing beast|'
       "The wizard's spell shattered the castle gates")
NEU = ('The waves crashed gently on the beach|'
       'The sand was cool to the touch|The sun was warm over the water')

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    t0 = time.time()
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

    tgt = [x.strip() for x in TGT.split('|') if x.strip()]
    neu = [x.strip() for x in NEU.split('|') if x.strip()]

    nm = torch.stack([logits_of(s) for s in neu]).mean(0)

    def perz_of(anchor_logits):
        """(anchor_logits - nm), per-sentence zscored then re-zscored (canonical)."""
        c = anchor_logits - nm
        c = (c - c.mean()) / (c.std() + 1e-6)
        return (c - c.mean()) / (c.std() + 1e-6)

    # canonical static full contrast (per-sentence zsum -> re-z => perz)
    perz_sum = None
    for s in tgt:
        Ls = logits_of(s)
        c = Ls - nm
        c = (c - c.mean()) / (c.std() + 1e-6)
        perz_sum = c if perz_sum is None else perz_sum + c
    perz = (perz_sum / max(1, len(tgt)))
    perz = (perz - perz.mean()) / (perz.std() + 1e-6)
    mK = torch.zeros(V); mK[perz.argsort(descending=True)[:K]] = 1.0
    dL_static = ALPHA * perz * mK
    N_REF = dL_static.norm()
    print(f'  K={K}  N_REF={N_REF:.3f}  toggle={int(mK.sum())}', flush=True)

    h_ids = set()
    for w in HELD_OUT.split():
        sp = tok(' ' + w, add_special_tokens=False).input_ids
        if len(sp) == 1: h_ids.add(int(sp[0]))

    def score(txt):
        toks = [w.strip('.,!?;:()[]"\'').lower() for w in re.findall(r"\S+", txt)]
        if not toks: return 0, 0, 0.0, 0
        maxr = cur = 0; prev = None; cnt = Counter()
        for t in toks:
            cnt[t] += 1
            if t == prev: cur += 1; maxr = max(maxr, cur)
            else: cur = 1; prev = t
        dist1 = len(cnt) / len(toks)
        low = txt.lower()
        h = sum(1 for w in HELD_OUT.split() if re.search(r'\b' + w + r'\w*', low))
        return h, maxr, dist1, (h >= 1 and maxr < 6 and dist1 > 0.6)

    def run(mode, seed):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        past = None; out_ids = []
        min_rank = 10**9; dH = 0.0; cos_seq = []
        with torch.no_grad():
            for step in range(NTOK):
                vc = {}
                hk = norm.register_forward_hook(lambda m, i, o: vc.__setitem__('o', o[0, -1, :].float().clone()))
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
                hk.remove()
                if past is None: past = out.past_key_values
                L0 = out.logits[0, -1, :].float()
                on = (step >= SW0)

                if on:
                    if mode == 'static':
                        dd = dL_static.to(DEV)
                    elif mode == 'dyn_prefix':
                        # recompute the concept contrast vs neutral at live prefix
                        anchor = torch.stack([logits_of(s) for s in tgt]).mean(0)
                        p = perz_of(anchor)
                        m = torch.zeros(V); m[p.argsort(descending=True)[:K]] = 1.0
                        d = ALPHA * p * m
                        dd = (d * (N_REF / d.norm())).to(DEV)
                        cos_seq.append((dL_static @ dd.cpu()).item() / (N_REF * dd.cpu().norm()))
                    elif mode == 'dyn_self':
                        # contrast current self next-token dist vs neutral
                        p = perz_of(L0.cpu())
                        m = torch.zeros(V); m[p.argsort(descending=True)[:K]] = 1.0
                        d = ALPHA * p * m
                        dd = (d * (N_REF / d.norm())).to(DEV)
                        cos_seq.append((dL_static @ dd.cpu()).item() / (N_REF * dd.cpu().norm()))
                    elif mode == 'dyn_prefix':
                        # recompute the concept contrast vs neutral at live prefix
                        anchor = torch.stack([logits_of(s) for s in tgt]).mean(0)
                        p = perz_of(anchor)
                        m = torch.zeros(V); m[p.argsort(descending=True)[:K]] = 1.0
                        d = ALPHA * p * m
                        dd = d * (N_REF / d.norm())
                        cos_seq.append((dL_static @ dd).item() / (N_REF * dd.norm()))
                    elif mode == 'dyn_self':
                        # contrast current self next-token dist vs neutral
                        p = perz_of(L0)
                        m = torch.zeros(V); m[p.argsort(descending=True)[:K]] = 1.0
                        d = ALPHA * p * m
                        dd = d * (N_REF / d.norm())
                        cos_seq.append((dL_static @ dd).item() / (N_REF * dd.norm()))
                    else:
                        dd = None

                L1 = L0 + (dd if on and dd is not None else 0.0)
                order1 = L1.argsort(descending=True).tolist()
                pos1 = {tid: k for k, tid in enumerate(order1)}
                if h_ids:
                    mr = min(pos1[i] for i in h_ids)
                    min_rank = min(min_rank, mr)
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
        _, maxr, dist1, ok = score(txt)
        # dH: mean held-out logP under steered softmax vs natural
        return dict(txt=txt, ok=ok, min_rank=min_rank, maxr=maxr, dist1=dist1,
                    cos_seq=cos_seq)

    for mode in ['static', 'dyn_prefix', 'dyn_self', 'none']:
        R = 0; ranks = []; maxrs = []; dist1s = []
        print(f'\n  == {mode.upper()} (SEEDS={SEEDS}) ==', flush=True)
        rot = []  # how much dynamic vector rotates vs static across steps
        for s in range(SEEDS):
            m = run(mode, SEEDBASE + s)
            R += int(m['ok']); ranks.append(m['min_rank'])
            maxrs.append(m['maxr']); dist1s.append(m['dist1'])
            if m['cos_seq']:
                rot.append(sum(m['cos_seq']) / len(m['cos_seq']))
        med = sorted(ranks)[SEEDS // 2] if ranks else -1
        mmax = sorted(maxrs)[SEEDS // 2] if maxrs else -1
        mdist = sorted(dist1s)[SEEDS // 2] if dist1s else -1
        mrot = sorted(rot)[SEEDS // 2] if rot else float('nan')
        print(f'  transport {R:3d}/{SEEDS:<4}  medMinR {med:6d}  medMaxR {mmax:4d}  '
              f'medDist1 {mdist:.3f}  medCosVsStatic {mrot:+.3f}', flush=True)
    print(f'[{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()