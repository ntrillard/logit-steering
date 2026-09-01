#!/usr/bin/env python3
"""generalize.py — Reproducibility / generalization of sparse ranked logit
steering across concepts, prompts, and target/neutral sets.

Question: is the K=150-250 operating window (and the ~10% transport rate) a
property of ONE vocabulary contrast (dragon/fantasy) and ONE prompt, or does
the technique survive across concepts, prompts, and target/neutral sets?

Design: for each (concept, prompt) cell, run the SWEEP machinery at
K in {100,150,200,250,300}, SEEDS seeds, NTOK tokens; report transport/N and
median best held-out rank. This both measures effect size (not just binary)
and tests window survival.

Concepts (target | neutral | held-out):
  FANTASY  dragon/knight/wizard vs beach        -> held: creature/evil/...
  SPACE    astronaut/rocket/planet vs kitchen   -> held: orbit/...
  PIRATE   ship/captain/loot vs meadow          -> held: ...

Run:
  SEEDS=6 NTOK=60 python3 generalize.py
  # or subset via CONCEPTS=FANTASY,SPACE PROMPTS=0,1
"""
import os, sys, time, re, importlib.util
from collections import Counter
import torch, transformers

SEEDS = int(os.environ.get('SEEDS', '6'))
NTOK  = int(os.environ.get('NTOK', '60'))
ALPHA = float(os.environ.get('ALPHA', '2.0'))
SW0   = int(os.environ.get('SW0', '20'))
K     = int(os.environ.get('K', '200'))
KSET  = [int(x) for x in os.environ.get('KSET', '100,150,200,250,300').split(',')]
MODEL = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
NUCLEUS = float(os.environ.get('NUCLEUS', '0.9'))
SEEDBASE = int(os.environ.get('SEEDBASE', '0'))

# concept cells: name -> (TGT, NEU, HELD_OUT, ANCHORS, UNREL)
CONCEPTS = {
    'FANTASY': (
        'A dragon circled the ruined towers of the ancient kingdom|'
        'A knight drew his sword against the fire-breathing beast|'
        "The wizard's spell shattered the castle gates",
        'The waves crashed gently on the beach|The sand was cool to the touch|'
        'The sun was warm over the water',
        'creature creatures evil monsters beast horror lurking nightmare demons',
        '', 'sand wave sea swim ocean surf beach tide shore shell'),
    'SPACE': (
        'The astronaut floated outside the space station|'
        'The rocket launched toward the distant planet|'
        'The crew explored the surface of the new world',
        'The cook stirred the soup in the kitchen|The farmer fed the chickens|'
        'The teacher wrote on the board',
        'orbit spaceship galaxy stars nebula comet asteroid mission astronaut',
        '', 'kitchen soup farmer chicken teacher board bread window'),
    'PIRATE': (
        'The pirate captain sailed his ship across the sea|'
        'The crew buried the chest of gold on the island|'
        'The sailor spotted a ship on the horizon',
        'The sheep grazed in the meadow|The cows walked to the barn|'
        'The farmer harvested the wheat',
        'ship captain crew treasure island gold sail sea loot',
        '', 'sheep cow meadow barn wheat farmer harvest grain'),
}

PROMPTS = [
    'The waves crashed gently on the beach',          # default prompt
    'The sun rose over the quiet town',                 # unrelated context
]

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

    want = [x.strip() for x in os.environ.get('CONCEPTS', 'FANTASY,SPACE,PIRATE').split(',')]
    want = [w for w in want if w in CONCEPTS]

    # shared lazy row(W) projector for DIAG mode (built once, not per concept)
    WtW_G = None
    W_lm = None
    for cname in want:
        TGT, NEU, HELD_OUT, ANCHORS, UNREL = CONCEPTS[cname]
        tgt = [x.strip() for x in TGT.split('|') if x.strip()]
        neu = [x.strip() for x in NEU.split('|') if x.strip()]

        nm = torch.stack([logits_of(s) for s in neu]).mean(0)
        perz_sum = None
        for s in tgt:
            Ls = logits_of(s)
            cc = Ls - nm
            cc = (cc - cc.mean()) / (cc.std() + 1e-6)
            perz_sum = cc if perz_sum is None else perz_sum + cc
        perz = perz_sum / max(1, len(tgt))
        perz = (perz - perz.mean()) / (perz.std() + 1e-6)

        if os.environ.get('DIAG') == '1':
            # vector-space diagnostics per concept (no generation). WtW built
            # lazily ONCE (the 152k GEMM is expensive; never repeat per concept).
            if WtW_G is None:
                W_lm = model.lm_head.weight.detach().float().cpu()
                WtW_G = W_lm.t() @ W_lm
            def proj_rowW(v):
                x = torch.linalg.solve(WtW_G, W_lm.t() @ v)
                return W_lm @ x
            print(f'\n  == DIAG {cname}: contrast concentration / rank / row-frac ==', flush=True)
            print(f'  {"K":>5} {"R_row":>7} {"cos(K,200)":>10} {"topPosFrac":>10} {"maxAbsFrac":>10}')
            m200 = torch.zeros(V); m200[perz.argsort(descending=True)[:200]] = 1.0
            d200 = ALPHA * perz * m200
            for kk in [25, 50, 100, 150, 200, 300, 500]:
                mk = torch.zeros(V); mk[perz.argsort(descending=True)[:kk]] = 1.0
                d = ALPHA * perz * mk
                rf = (proj_rowW(d).norm()**2 / (d.norm()**2 + 1e-9)).item()
                cs = (d @ d200).item() / (d.norm() * d200.norm() + 1e-9)
                top_abs = perz[perz.argsort(descending=True)[:kk]].abs().sum()
                tpf = (top_abs / perz.abs().sum()).item()
                maf = (perz[perz.argsort(descending=True)[:kk]].abs().max() / perz.abs().max()).item()
                print(f'  {kk:5d} {rf:7.3f} {cs:10.3f} {tpf:10.3f} {maf:10.3f}', flush=True)
            continue

        h_ids = set()

        h_ids = set()
        for w in HELD_OUT.split():
            sp = tok(' ' + w, add_special_tokens=False).input_ids
            if len(sp) == 1: h_ids.add(int(sp[0]))

        def score(txt):
            toks = [w.strip('.,!?;:()[]"\'').lower() for w in re.findall(r"\S+", txt)]
            if not toks: return 0, 0.0, 0
            maxr = cur = 0; prev = None; cnt = Counter()
            for t in toks:
                cnt[t] += 1
                if t == prev: cur += 1; maxr = max(maxr, cur)
                else: cur = 1; prev = t
            dist1 = len(cnt) / len(toks)
            low = txt.lower()
            h = sum(1 for w in HELD_OUT.split() if re.search(r'\b' + w + r'\w*', low))
            return h, maxr, dist1

        dK_cache = {}

        # N_REF = the known working dL at K=200
        m200 = torch.zeros(V); m200[perz.argsort(descending=True)[:200]] = 1.0
        N_REF = (ALPHA * perz * m200).norm()
        for k in KSET:
            m = torch.zeros(V); m[perz.argsort(descending=True)[:k]] = 1.0
            dK_cache[k] = (ALPHA * perz * m) * (N_REF / ((ALPHA * perz * m).norm() + 1e-9))

        def genK(dd, seed):
            torch.manual_seed(seed)
            ids = tok(PROMPT_use, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
            past = None; out_ids = []; minr = 10**9
            with torch.no_grad():
                for step in range(NTOK):
                    vc = {}
                    hk = norm.register_forward_hook(lambda m, i, o: vc.__setitem__('o', o[0, -1, :].float().clone()))
                    out = model(input_ids=ids, past_key_values=past, use_cache=True)
                    hk.remove()
                    if past is None: past = out.past_key_values
                    L0 = out.logits[0, -1, :].float()
                    on = (step >= SW0)
                    L1 = L0 + (dd.to(DEV) if (on and dd is not None) else 0.0)
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
            h, maxr, dist1 = score(txt)
            ok = (h >= 1) and (maxr < 6) and (dist1 > 0.6)
            return ok, minr

        for pname, PROMPT_use in enumerate(PROMPTS):
            print(f'\n  == CONCEPT {cname} / prompt[{pname}] = {PROMPT_use!r} ==', flush=True)
            # baseline (unsteered) row - no dd
            R0 = 0; mrs0 = []
            for s in range(SEEDS):
                ok0, mr0 = genK(None, SEEDBASE + s)
                R0 += int(ok0); mrs0.append(mr0)
            med0 = sorted(mrs0)[SEEDS // 2] if mrs0 else -1
            print(f'  {"NONE":>5} {R0:3d}/{SEEDS:<4} {med0:8d}   <- unsteered baseline', flush=True)
            for k in KSET:
                R = 0; mrs = []
                for s in range(SEEDS):
                    ok, mr = genK(dK_cache[k], SEEDBASE + s)
                    R += int(ok); mrs.append(mr)
                med = sorted(mrs)[SEEDS // 2] if mrs else -1
                print(f'  {k:5d} {R:3d}/{SEEDS:<4} {med:8d}', flush=True)
    print(f'[{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()