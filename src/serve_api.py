# -*- coding: utf-8 -*-
import json
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="Agnos Symptom Recommender", version="1.0")

MODEL = json.loads(Path("model/model.json").read_text(encoding="utf-8"))
VOCAB = set(json.loads(Path("model/vocab.json").read_text(encoding="utf-8"))["id2sym"])

def lift(co, cx, cy, N):
    if cx == 0 or cy == 0: 
        return 0.0
    return (co * N) / (cx * cy)

class RecommendRequest(BaseModel):
    gender: str = "ไม่ระบุ"
    age: int = 30
    symptoms: List[str]
    top_k: int | None = None

@app.post("/recommend")
def recommend(req: RecommendRequest):
    gender = (req.gender or "ไม่ระบุ").strip()
    age = int(req.age) if req.age is not None else 0
    bins = [(0,12),(13,19),(20,39),(40,59),(60,200)]
    age_group = next((f"{lo}-{hi}" for lo,hi in bins if lo<=age<=hi), "NA")

    N = MODEL["N"]; pw = MODEL["profile_weight"]
    count = MODEL["count"]; co = MODEL["co"]; p_count = MODEL["p_count"]; p_co = MODEL["p_co"]
    key = f"{gender}|||{age_group}"
    cntp = p_count.get(key, {}); cop = p_co.get(key, {})

    inp = [s for s in req.symptoms if s in VOCAB]
    scores = {}
    for b in VOCAB:
        if b in inp: continue
        sc=0.0
        for a in inp:
            from_g = co.get(f"{a}|||{b}", 0) * N / max(1, count.get(a,0)*count.get(b,0))
            from_p = 0.0
            if cntp:
                from_p = cop.get(f"{a}|||{b}", 0) * N / max(1, cntp.get(a,0)*cntp.get(b,0))
            sc += (1-pw)*from_g + pw*from_p
        if sc>0: scores[b]=sc
    top_k = req.top_k or MODEL["topk_default"]
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return {"bucket": key, "items": [{"symptom": s, "score": float(v)} for s,v in ranked]}