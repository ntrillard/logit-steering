#!/usr/bin/env python3
"""Quick metric reconciliation: compute BOTH concentration metrics for the
three concept contrasts and see which (if any) discriminates working (FANTASY)
from dead/contaminated (SPACE/PIRATE).

m1 = max|topK| / sum|topK|   (share of top-K mass in the single largest coord)
m2 = max|topK| / max|all|    (DIAG mode's column: is the global argmax inside?)
m3 = sum|topK| / sum|all|    (topPosFrac: how much of total contrast mass is
                              covered by the selected sparse set)
"""
import os
import torch, transformers

MODEL = 'Qwen/Qwen2-1.5B'
K = 200
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

CONCEPTS = {
    'FANTASY': (
        'A dragon circled the ruined towers of the ancient kingdom|'
        'A knight drew his sword against the fire-breathing beast|'
        "The wizard's spell shattered the castle gates",
        'The waves crashed gently on the beach|The sand was cool to the touch|'
        'The sun was warm over the water'),
    'SPACE': (
        'The astronaut floated outside the space station|'
        'The rocket launched toward the distant planet|'
        'The crew explored the surface of the new world',
        'The cook stirred the soup in the kitchen|The farmer fed the chickens|'
        'The teacher wrote on the board'),
    'PIRATE': (
        'The pirate captain sailed his ship across the sea|'
        'The crew buried the chest of gold on the island|'
        'The sailor spotted a ship on the horizon',
        'The sheep grazed in the meadow|The cows walked to the barn|'
        'The farmer harvested the wheat'),
}


def main():
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()

    def logits_of(s):
        t = tok(s, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        with torch.no_grad():
            return model(t).logits[0, -1, :].float().cpu()

    print(f'  {"concept":>8} {"m1 max/sumK":>11} {"m2 max/maxA":>11} {"m3 sumK/sumA":>11}')
    for cname, (TGT, NEU) in CONCEPTS.items():
        tgt = [x.strip() for x in TGT.split('|') if x.strip()]
        neu = [x.strip() for x in NEU.split('|') if x.strip()]
        nm = torch.stack([logits_of(s) for s in neu]).mean(0)
        perz_sum = None
        for s in tgt:
            Ls = logits_of(s)
            c = Ls - nm
            c = (c - c.mean()) / (c.std() + 1e-6)
            perz_sum = c if perz_sum is None else perz_sum + c
        perz = (perz_sum / len(tgt))
        perz = (perz - perz.mean()) / (perz.std() + 1e-6)
        topK = perz.argsort(descending=True)[:K]
        aK = perz[topK].abs()
        m1 = (aK.max() / (aK.sum() + 1e-9)).item()
        m2 = (aK.max() / (perz.abs().max() + 1e-9)).item()
        m3 = (aK.sum() / (perz.abs().sum() + 1e-9)).item()
        print(f'  {cname:>8} {m1:11.4f} {m2:11.4f} {m3:11.4f}')


if __name__ == '__main__':
    main()