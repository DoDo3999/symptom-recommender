# -*- coding: utf-8 -*-
import yaml, itertools, random
from collections import Counter
from .data_prep import build_transactions
from .train_recommender import lift

def evaluate(cfg_path: str, K: int = 10):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    tx, _ = build_transactions(cfg["data_path"], cfg)
    counts = Counter(itertools.chain.from_iterable([set(s) for s in tx["symptoms"].tolist()]))
    vocab = set([s for s,c in counts.items() if c >= int(cfg.get("min_support", 5))])
    baskets = [list(set([s for s in row if s in vocab])) for row in tx["symptoms"].tolist() if len(row)>=2]

    co = Counter()
    for s in baskets:
        s = sorted(set(s))
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                a,b = s[i], s[j]
                co[(a,b)] += 1; co[(b,a)] += 1

    N = max(1, len(baskets))
    hits = 0; ap = 0.0; total = 0

    for s in baskets:
        target = random.choice(s)
        context = [x for x in s if x != target]
        scores = {}
        for b in vocab:
            if b in context: continue
            sc = 0.0
            for a in context:
                sc += lift(co.get((a,b),0), counts.get(a,0), counts.get(b,0), N)
            if sc>0: scores[b]=sc
        ranked = [x for x,_ in sorted(scores.items(), key=lambda x: -x[1])][:K]
        if target in ranked:
            hits += 1
            ap += 1.0/(ranked.index(target)+1)
        total += 1

    hitk = hits/total if total else 0.0
    mapk = ap/total if total else 0.0
    print(f"[EVAL] N={N} | Hit@{K}={hitk:.4f} | MAP@{K}={mapk:.4f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    evaluate(a.config, a.k)