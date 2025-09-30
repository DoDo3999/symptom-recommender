# -*- coding: utf-8 -*-
import json, yaml, itertools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any
from .data_prep import build_transactions

def lift(co, cx, cy, N):
    if cx == 0 or cy == 0: return 0.0
    return (co * N) / (cx * cy)

def train_model(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    tx, detected = build_transactions(cfg["data_path"], cfg)
    min_support = int(cfg.get("min_support", 5))
    min_pair = int(cfg.get("min_pair_coocc", 3))
    vocab_counts = Counter(itertools.chain.from_iterable(tx["symptoms"].tolist()))
    vocab = sorted([s for s,c in vocab_counts.items() if c >= min_support])
    vset = set(vocab)

    filtered = []
    for _, row in tx.iterrows():
        sx = [s for s in row["symptoms"] if s in vset]
        if sx: filtered.append({"gender": row["gender"], "age_group": row["age_group"], "symptoms": sx})
    tx = filtered; N = len(tx)

    count = Counter()
    co = Counter()
    p_count = defaultdict(Counter)
    p_co = defaultdict(Counter)

    for r in tx:
        s = sorted(set(r["symptoms"]))
        g,a = r["gender"], r["age_group"]
        count.update(s)
        p_count[(g,a)].update(s)
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                a_s, b_s = s[i], s[j]
                co[(a_s,b_s)] += 1; co[(b_s,a_s)] += 1
                p_co[(g,a)][(a_s,b_s)] += 1; p_co[(g,a)][(b_s,a_s)] += 1

    co = Counter({k:v for k,v in co.items() if v >= min_pair})

    Path("model").mkdir(parents=True, exist_ok=True)
    with open("model/vocab.json", "w", encoding="utf-8") as f:
        json.dump({"id2sym": vocab}, f, ensure_ascii=False, indent=2)
    model = {
        "N": N,
        "profile_weight": float(cfg.get("profile_weight", 0.5)),
        "topk_default": int(cfg.get("topk_recommend", 10)),
        "count": dict(count),
        "co": {f"{a}|||{b}": int(v) for (a,b),v in co.items()},
        "p_count": {f"{g}|||{ag}": dict(c) for (g,ag),c in p_count.items()},
        "p_co": {f"{g}|||{ag}": {f"{a}|||{b}": int(v) for (a,b),v in c.items()} for (g,ag),c in p_co.items()},
    }
    with open("model/model.json", "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)
    print(f"[OK] Trained N={N} | vocab={len(vocab)} | pairs={len(co)}")
    print(f"[HINT] Detected: {detected}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    a = ap.parse_args()
    train_model(a.config)